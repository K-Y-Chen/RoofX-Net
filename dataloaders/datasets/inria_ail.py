import os
import sys
sys.path.append("/workspace/10ka/backup/pytorch-deeplab-xception")
import numpy as np
from glob import glob
from PIL import Image
from torch.utils import data
from config import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import re


# Implemented dataset for train, val.
# Test set missing labels.

class InriaAIL(data.Dataset):
    """ Inria Aerial Image Labeling dataset from 'Can semantic labeling methods
    generalize to any city? the inria aerial image labeling benchmark', Maggiori et al. (2017)
    https://ieeexplore.ieee.org/document/8127684

    'The training set contains 180 color image tiles of size 5000x5000, covering a surface of 1500mx1500m
    each (at a 30 cm resolution). There are 36 tiles for each of the following regions (Austin, Chicago, Kitsap County, Western Tyrol, Vienna)
    The format is GeoTIFF. Files are named by a prefix associated to the region (e.g., austin- or vienna-)
    followed by the tile number (1-36). The reference data is in a different folder and the file names
    correspond exactly to those of the color images. In the case of the reference data, the tiles are
    single-channel images with values 255 for the building class and 0 for the not building class.'
    """
    splits = ["train", "val", "test"]
    NUM_CLASSES = 2

    def __init__(self, args, root=Path.db_root_dir('inria_ail'), split="train"):
        self.split = split
        self.images = self.load_images(root, split)
        self.regions = sorted(list(set(image["region"] for image in self.images)))
        self.args = args
    
    def load_images(self, path, split):
        if split in ["train", "val"]:
            # print(split)
            with open(os.path.join(path, 'train', f'{split}_list.txt'),'r') as f:
                img_ids = f.readlines()
            images = [os.path.join(path, 'train', 'images', img_id[:-1]) for img_id in img_ids]
            targets = [os.path.join(path, 'train', 'gt', img_id[:-1]) for img_id in img_ids]
            # print(images)
            # print(targets)
            pattern = re.compile("[a-zA-Z]+")
            regions = [re.findall(pattern, os.path.basename(image))[0] for image in images]
        elif split == 'test':
            images = sorted(glob(os.path.join(path, split, "images", "*.tif")))
            images = sorted()
            pattern = re.compile("[a-zA-Z]+")
            regions = [re.findall(pattern, os.path.basename(image))[0] for image in images]

        else:
            targets = [None] * len(images)

        files = [
            dict(image=image, target=target, region=region)
            for image, target, region in zip(images, targets, regions)
        ]
        return files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, target_path = self.images[idx]["image"], self.images[idx]["target"]   
        image = Image.open(image_path)
        if self.split == "train" or self.split == "val":
            _mask = np.array(Image.open(target_path))
            _mask = np.clip(_mask, a_min=0, a_max=1)
            mask = mask = Image.fromarray(_mask)
        else:
            mask = None
            
        sample = {'image': image, 'label': mask}
        
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
        tr.RandomGaussianBlur(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
        
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Inria_ail(split=' + str(self.splits) + ')'
    
    
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.dataset = 'inria_ail'

    inria_ail_train = InriaAIL(args, split='val')

    dataloader = DataLoader(inria_ail_train, batch_size=2, shuffle=True, num_workers=2)
    
    for ii, sample in enumerate(dataloader):
        
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset=args.dataset)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.savefig(f'display.{args.dataset}.png')