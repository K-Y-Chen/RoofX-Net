import os
# import sys
# sys.path.append("/mnt/disk/cv/KeyuChen/pytorch-deeplab-xception")
import numpy as np
from PIL import Image
from torch.utils import data
from config import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

# Implemented dataset for train, val.
# No test set.

class DubaiSegmentation(data.Dataset):
    """ Semantic segmentation dataset of Dubai imagery taken by MBRSC satellites
    https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset/

    """
    classes = {
        "Unlabeled":            {"rgb": (155, 155, 155), "color": "#9B9B9B"},
        "Water":                {"rgb": (226, 169, 41),  "color": "#E2A929"},
        "Land (unpaved area)":  {"rgb": (132, 41, 246),  "color": "#8429F6"},
        "Road":                 {"rgb": (110, 193, 228), "color": "#6EC1E4"},
        "Building":             {"rgb": (60, 16, 152),   "color": "#3C1098"},
        "Vegetation":           {"rgb": (254, 221, 58),  "color": "#FEDD3A"}
    }
    colors = [v["rgb"] for k, v in classes.items()]
    NUM_CLASSES = 6

    def __init__(self, args, root=Path.db_root_dir('dubai'), split="train"):
        self.split = split
        self.images = self.load_images(root)
        self.regions = list(set([image["region"] for image in self.images]))
        self.args = args
    
    def load_images(self, path):
        with open(os.path.join(path, f'{self.split}_list.txt'),'r') as f:
                img_list = f.readlines()
        images = [os.path.join(path, img[:-1]) for img in img_list]
        masks = [image.replace("images", "masks").replace("jpg", "png") for image in images]
        # images = sorted(glob(os.path.join(path, "**", "images", "*.jpg"), recursive=True))
        # masks = sorted(glob(os.path.join(path, "**", "masks", "*.png"), recursive=True))
        
        regions = [image.split(os.sep)[-3] for image in images]
        files = [
            dict(image=image, mask=mask, region=region)
            for image, mask, region in zip(images, masks, regions)
        ]
        return files

    
    def rgb_to_mask(self, rgb, colors):
        h, w = rgb.shape[:2]
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        for i, c in enumerate(colors):
            cmask = (rgb == c)
            if isinstance(cmask, np.ndarray):
                mask[cmask.all(axis=-1)] = i

        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, target_path = self.images[idx]["image"], self.images[idx]["mask"]
        image = Image.open(image_path).convert("RGB")
        _mask = np.array(Image.open(target_path).convert("RGB"))
        _mask = self.rgb_to_mask(_mask, self.colors)
        mask = Image.fromarray(_mask, mode='P')
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
        return 'Dubai_segmentation(split=' + str(self.splits) + ')'
    
    
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.dataset = 'dubai'

    dubai_train = DubaiSegmentation(args, split='val')
    # print(dubai_train[0])

    dataloader = DataLoader(dubai_train, batch_size=2, shuffle=True, num_workers=2)

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
            print(img_tmp.shape)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.savefig(f'display.{args.dataset}.png')