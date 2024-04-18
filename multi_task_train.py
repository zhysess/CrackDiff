import os
import sys
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import tempfile
from distribute_utils import init_distributed_mode, dist, is_main_process, reduce_value
from multi_task.mltdiff import GaussianDiffusionSampler, GaussianDiffusionTrainer
from multi_task.mlt_unet import UNet
from crack500Dataloader import CrackDataset
from Scheduler import GradualWarmupScheduler


# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # if train on windows

def train(modelConfig: Dict, args):
    
    init_distributed_mode(args)
    rank = args.rank
    device = torch.device(args.device)


    with open(os.path.join(modelConfig["dataPath"], "train.txt"), 'r') as f:
            train_areas = [line.split()[0] for line in f.readlines()]
    train_dataset = CrackDataset(modelConfig["dataPath"], train_areas, _augment=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    
    nw = min([os.cpu_count(), modelConfig["batch_size"] if modelConfig["batch_size"] > 1 else 0, 8])  # number of workers
    if rank == 0:
        print(args)
        print('Using {} dataloader workers every process'.format(nw))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=modelConfig["batch_size"],
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=nw
    )


    # model setup
    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    if modelConfig["training_load_weight"] is not None:

        model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    else:

        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    
    trainer = GaussianDiffusionTrainer(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    # 转为DDP模型
    trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[args.gpu])
    loss_file = open("loss.txt", "w")

    # start training
    for e in range(modelConfig["epoch"]):
        train_sampler.set_epoch(e)
        train_one_epoch(model=trainer, optimizer=optimizer, 
                        data_loader=train_loader, device=device, 
                        epoch=e, modelConfig=modelConfig, f=loss_file)

        if (e + 1) % modelConfig["weight_save_iterval"] == 0:
            torch.save(model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        warmUpScheduler.step()
    
    if modelConfig["training_load_weight"] is None and rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    loss_file.close()
    dist.destroy_process_group()

def train_one_epoch(model, optimizer, data_loader, device, epoch, modelConfig, f):
    model.train()
    optimizer.zero_grad()
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    epoch_x_loss = 0.0
    epoch_img_loss = 0.0
    for step, images in enumerate(data_loader):
        x_0 = images["mask"].to(device)
        feature = images["crack"].to(device)
        x_loss, img_loss = model(x_0, feature)
        
        epoch_x_loss += x_loss
        epoch_img_loss += img_loss
        loss = x_loss + img_loss
        loss.backward()
        loss = reduce_value(loss, average=True)

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] noise loss {} img loss {}".format(epoch, 
                                                                    # round(x_loss.item(), 6), 
                                                                    # round(img_loss.item(), 6))
                                                                round(epoch_x_loss.item()/(step+1), 6), 
                                                                round(epoch_img_loss.item()/(step+1), 6))
            f.write(f"{epoch},{step},{x_loss}, {img_loss}\n")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), modelConfig["grad_clip"])
        optimizer.step()
        optimizer.zero_grad()
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
       


if __name__ == '__main__':
    
    modelConfig = {
        "epoch": 100,
        "weight_save_iterval": 20,
        "batch_size": 5,
        "T": 500,
        "channel": 64,
        "channel_mult": [1, 2, 2, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1,
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "dataPath": "./data/crack500"
    }
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    parser = argparse.ArgumentParser()
    # do not change
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # do not change
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    arg = parser.parse_args()

    train(modelConfig, arg)

