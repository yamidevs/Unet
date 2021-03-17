from tqdm import tqdm
from CarvanaDataset import CarvanaDataset
from Utils import (
    get_loaders,
    save,
    load,
    check_accuracy,
    save_validation_image
)

from UNet import UNet
import torch
import torch.nn as nn

import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2
TRAIN_PATH = "data/train"
TRAIN_MASKS_PATH = "data/train_masks"
VALID_PATH = "data/valid"
VALID_MASKS_PATH = "data/valid_masks"

def train_fn(loader,model,optimizer,loss_fn,scalar):
    
    loop = tqdm(loader)
    for index,(data,targets) in enumerate(loop):
        data = data.to(device = Config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device = Config.DEVICE)
    
        # the use of autocast is used to accelerate the training
        with torch.cuda.amp.autocast():
            result = model(data)
            loss = loss_fn(result,targets)
        
        optimizer.zero_grad()
        # using the gradient scaling for avoid underflow gradient, for more information refer to https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
        scalar.scale(loss).backward()
        # scalar step include the unscale function
        scalar.step(optimizer)
        scalar.update()

        loop.set_postfix(loss = loss.item())

def main():

    print("STRATING TRAINING")
    train_transform = A.Compose(
        [
            A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
            A.Normalize(
                mean=0.5,
                std = 0.5,
            ),
            A.GaussNoise(3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(0.05,0.05,0.6),
            ToTensorV2(),
            
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
            A.Normalize(
                mean=0.5,
                std = 0.5,
            ),
            ToTensorV2(),
        ],
    )

    train_loader,valid_loader = get_loaders(TRAIN_PATH,TRAIN_MASKS_PATH,
    VALID_PATH,VALID_MASKS_PATH,Config.BATCH_SIZE,train_transform,val_transforms)

    model = UNet(1,1).to(device=Config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=Config.LEARNING_RATE)
    scalar = torch.cuda.amp.GradScaler()
    check_accuracy(valid_loader,model)
    for epoch in range(Config.NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn,scalar)

        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer"  : optimizer.state_dict()
        }

        save(checkpoint,"my_checkpoint.pth.tar")

        check_accuracy(valid_loader,model)
        save_validation_image(valid_loader,model)

if __name__ == "__main__":
    main()
    

        
