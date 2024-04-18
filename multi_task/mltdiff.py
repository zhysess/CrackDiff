
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import os
from distribute_utils import is_main_process
from tqdm import tqdm

ALPHA = 0.3
BETA =0.7
GAMMA = 0.75
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        self.image_loss = FocalTverskyLoss()
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, image):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        seg_pred, noisy_pred = self.model(x_t, t, image) ###################################
        x_loss = F.mse_loss(noisy_pred, noise)
        img_loss = self.image_loss(seg_pred, x_0)
        return x_loss, img_loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, sampled_dir):
        super().__init__()

        self.model = model
        self.T = T
        self.sampled_dir = sampled_dir
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, image):
        # below: only log_variance is used in the KL computations
        # print(x_t.shape, image.shape)
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        seg_result, eps = self.model(x_t, t, image) #################################
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var, seg_result

    def forward(self, x_T, image, info):
        """
        Algorithm 2.
        """
        x_t = x_T
        tqdmT = list(reversed(range(self.T)))
        if is_main_process():
            tqdmT = tqdm(tqdmT) 
        for _, time_step in enumerate(tqdmT):
            # 在进程0中打印平均loss
            if is_main_process():
                tqdmT.desc = "[epoch {}] {}/{}".format(info["epoch"], info["current_batch"], info["total_batch"])
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var, seg_result = self.p_mean_variance(x_t=x_t, t=t, image=image)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            # if (time_step + 1) % 50 ==0 or time_step == 0 or(time_step<50 and time_step%10==0):
            #     torch.save(x_t, 
            #             os.path.join("./choose_crack500/diffusion_step", f"noise_{time_step}.pt"))
            #     torch.save(seg_result, 
            #             os.path.join("./choose_crack500/diffusion_step", f"seg_{time_step}.pt"))

        x_0 = x_t
        return (torch.clip(x_0, -1, 1) + 1) * 0.5, seg_result, x_0




