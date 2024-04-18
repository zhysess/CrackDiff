import torch.utils.data as data
from PIL import Image,ImageEnhance
import os
import torchvision.transforms as transforms
import torch
import numpy as np
import random
import torchvision.transforms.functional as F


class CrackDataset(data.Dataset):
    def __init__(self, data_path, names, _augment):
        super(CrackDataset, self).__init__()
        self.data_path = data_path
        self.names = names
        self.augment = _augment

    @staticmethod
    def augmentate(image, mask):
        # it is expected to be in [..., H, W] format
        image = torch.from_numpy(
            np.array(image, dtype=np.uint8)).permute(2, 0, 1)
        mask = torch.unsqueeze(torch.from_numpy(
            np.array(mask, dtype=np.uint8)), dim=0)
        image = F.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))
        image = F.adjust_contrast(
            image, contrast_factor=random.uniform(0.8, 1.2))
        image = F.adjust_brightness(
            image, brightness_factor=random.uniform(0.8, 1.2))
        image = F.adjust_saturation(
            image, saturation_factor=random.uniform(0.8, 1.2))
        image = F.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))
        image_mask = torch.cat([image, mask], dim=0)

        if random.uniform(0, 1) > 0.5:
            image_mask = F.hflip(image_mask)
        if random.uniform(0, 1) > 0.5:
            image_mask = F.vflip(image_mask)
        if random.uniform(0, 1) > 0.5:
            image_mask = F.rotate(image_mask, angle=90)

        image = image_mask[0:3, ...]
        mask = image_mask[3, ...].unsqueeze(dim=0)

        return F.to_pil_image(image), F.to_pil_image(mask)
    
    def __getitem__(self, index):
        name = self.names[index]
        img_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            # transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])
        crack = Image.open(os.path.join(self.data_path, "Images", name))
        mask = Image.open(os.path.join(self.data_path, "Labels", name)).convert('L')

        if self.augment:
            crack, mask = self.augmentate(crack, mask)
        
        crack = img_transform(crack)
        mask = img_transform(mask)  # [1, h, w]

        return {
            "name": name,
            "crack": crack,
            "mask": mask
        }

    def __len__(self):
        return len(self.names)

if __name__ == "__main__":
    with open(os.path.join("./data/crack500", "train.txt"), 'r') as f:
        train_areas = [line.split()[0] for line in f.readlines()]
    train_dataset = CrackDataset("./data/crack500", train_areas, _augment=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               num_workers=4
    )
    c = next(iter(train_loader))["crack"]
    m = next(iter(train_loader))["mask"]
    print(c.shape, m.shape)