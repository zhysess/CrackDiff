import os
import sys
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import tempfile
from distribute_utils import init_distributed_mode, dist, is_main_process, reduce_value
from multi_task.mltdiff import GaussianDiffusionSampler, GaussianDiffusionTrainer, FocalTverskyLoss
from multi_task.mlt_unet import UNet
from crack500Dataloader import CrackDataset
from Scheduler import GradualWarmupScheduler
import time

def test(modelConfig: Dict, args):

    init_distributed_mode(args)

    device = torch.device(args.device)
    # load model and evaluate
    with torch.no_grad():
        
        with open(os.path.join(modelConfig["dataPath"], "test.txt"), 'r') as f:
                test_areas = [line.split()[0] for line in f.readlines()]

        test_dataset = CrackDataset(modelConfig["dataPath"], test_areas, _augment=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        nw = min([os.cpu_count(), modelConfig["batch_size"] if modelConfig["batch_size"] > 1 else 0, 8])  # number of workers
        test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=modelConfig["batch_size"],
                        sampler=test_sampler,
                        pin_memory=True,
                        num_workers=nw
        )
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), 
            map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["sampled_dir"]).to(device)

        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[args.gpu])
        # for r in range(modelConfig["val_epoch"]):
        for r in range(modelConfig["val_start_epoch"], modelConfig["val_epoch"]):
            
            sample_one_epoch(model=sampler, data_loader=test_loader, 
                             device=device, epoch=r, modelConfig=modelConfig)

def sample_one_epoch(model, data_loader, device, epoch, modelConfig):
    model.eval()
    batch = len(data_loader)
    loss = FocalTverskyLoss()
    start = time.time()
    for i, images in enumerate(data_loader):
        if i < modelConfig["val_start_loader"]:
            continue
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[len(images["name"]), 1, 256, 256], device=device)

        feature = images["crack"].to(device)
        name = images["name"]
        info = {
            "epoch": epoch,
            "total_batch": batch,
            "current_batch": i
        }
        sampledImgs, segImgs, logits = model(noisyImage, feature, info)
        
        l = loss(logits, images["mask"].to(device)).item()
        
        end = time.time()
        if is_main_process():
            print(l, end - start)
            with open("val_loss.txt", "a") as f:
                f.write(str(l) + ',' + str(end-start) + '\n')
        break

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
                    

if __name__ == '__main__':


    modelConfig = {
        "batch_size": 20,
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
        "test_load_weight": "ckpt_99_.pt",
        "save_weight_dir": "./Checkpoints/",
        "sampled_dir": "./generation/",
        "dataPath": "./data/crack500",
        "val_epoch": 1,  # validation numbers
        "val_start_epoch": 0,
        # "val_end_epoch": 5,
        "val_start_loader": 0

    }
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--val_start_e', default=0, type=int, help="which number to start test ")
    # parser.add_argument('--val_end_e', default=modelConfig["val_epoch"], type=int, help='which number to end test')
    arg = parser.parse_args()
    # modelConfig["val_start_epoch"] = arg.val_start_e
    # modelConfig["val_end_epoch"] = arg.val_end_e
    test(modelConfig, arg)
