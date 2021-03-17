import torch
import torchvision
from CarvanaDataset import CarvanaDataset
from torch.utils.data import DataLoader
import Config
def get_loaders(
    train_dir,
    train_maskdir,
    valid_dir,
    valid_maskdir,
    batch_size,
    train_transformer,
    val_transformer,
    num_worker=2,
    pin_memory = True
):
    train_data = CarvanaDataset(train_dir,train_maskdir,train_transformer)
    train_loader = DataLoader(
        dataset = train_data,
        batch_size= batch_size,
        pin_memory=pin_memory,
        shuffle=True
    )

    valid_data = CarvanaDataset(valid_dir,valid_maskdir,val_transformer)
    valid_loader = DataLoader(
        dataset = valid_data,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=pin_memory

    )

    return train_loader,valid_loader


def save(state,file_name):
    print("Saving check-point")
    torch.save(state,file_name)

def load(checkpoint,model):
    print("Loading check-point")
    model.load_state_dict(checkpoint["state_dict"])

def save_validation_image(loader,model):
    model.eval()

    for index,(data,target) in enumerate(loader):
        data = data.to(device = Config.DEVICE)
        with torch.no_grad():
            result = torch.sigmoid(model(data))
            result = (result > 0.85).float()
            torchvision.utils.save_image(result,
            f"save/{index}_pred.png")
            torchvision.utils.save_image(target.unsqueeze(1),
            f"save/{index}.png")
        
    model.train()

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(Config.DEVICE)
            y = y.to(Config.DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.85).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()