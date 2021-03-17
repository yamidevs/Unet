import os
from PIL import Image
from  torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    
    def __init__(self,img_path,mask_path,transform = None):
        super(CarvanaDataset,self).__init__()
        self.img_dir = img_path
        self.mask_dir = mask_path
        self.transform = transform
        self.images = os.listdir(img_path)
    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img_path     = os.path.join(self.img_dir,self.images[index])
        mask_path    = os.path.join(self.mask_dir,self.images[index].replace('.jpg','.jpg'))
        image        = np.array(Image.open(img_path).convert("L"),dtype=np.float32)
        mask         = np.array(Image.open(mask_path).convert('L'),dtype=np.float32)
        mask[mask == 255.0] = 1.0

        # data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask  = augmentations["mask"]
        
        return image,mask

if __name__ == "__main__":
    carvana = CarvanaDataset("data/train","data/train_masks")
    print(carvana.__getitem__(5)[0])
    