import os
import sys
sys.path.append("/mnt/disk/cv/KeyuChen/pytorch-deeplab-xception")
import numpy as np
from glob import glob
import scipy.misc as m
from PIL import Image
from torch.utils import data
from config import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


# Implemented dataset for train, val.
# Test set missing labels.

class GID15(data.Dataset):
    """ Gaofen Image Dataset (GID-15) from 'Land-Cover Classification with High-Resolution
    Remote Sensing Images Using Transferable Deep Models', Tong et al. (2018)
    https://arxiv.org/abs/1807.05713

    'We construct a new large-scale land-cover dataset with Gaofen-2 (GF-2) satellite
    images. This new dataset, which is named as Gaofen Image Dataset with 15 categories
    (GID-15), has superiorities over the existing land-cover dataset because of its
    large coverage, wide distribution, and high spatial resolution. The large-scale
    remote sensing semantic segmentation set contains 150 pixel-level annotated GF-2
    images, which is labeled in 15 categories.'
    """
    classes = [
        "background",
        "industrial_land",
        "urban_residential",
        "rural_residential",
        "traffic_land",
        "paddy_field",
        "irrigated_land",
        "dry_cropland",
        "garden_plot",
        "arbor_woodland",
        "shrub_land",
        "natural_grassland",
        "artificial_grassland",
        "river",
        "lake",
        "pond",
    ]
    NUM_CLASSES = 15 + 1
    def __init__(self, args, root=Path.db_root_dir('gid15'), split="train"):
        self.split = split        
        self.args = args
        self.images = self.load_images(root)
    
    def load_images(self, path):
        
        images = sorted(glob(os.path.join(path, "img_dir", self.split, "*.tif")))
        if self.split in ["train", "val"]:
            masks = [
                image.replace("img_dir", "ann_dir").replace(".tif", "_15label.png")
                for image in images
            ]
        else:
            masks = [None] * len(images)

        files = [dict(image=image, mask=mask) for image, mask in zip(images, masks)]
        return files    
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, mask_path = self.images[idx]["image"], self.images[idx]["mask"]
        image = Image.open(image_path)        
        if self.split in ["train", "val"]:
            mask = Image.open(mask_path)            
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
        return 'GID15(split=' + str(self.split) + ')'
    
    
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.dataset = 'gid15'

    dubai_train = GID15(args, split='val')
    # print(dubai_train[0])

    dataloader = DataLoader(dubai_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset=args.dataset)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            print(img[jj].shape)
            print(img_tmp.shape)
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
