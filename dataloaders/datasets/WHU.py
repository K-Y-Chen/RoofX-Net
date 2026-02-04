import os
import sys

sys.path.append("/workspace/wangtianlei/Codes/segmentation/pytorch-deeplab-xception")
from glob import glob

import numpy as np
from config import Path
from dataloaders import custom_transforms as tr
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WHUSegmentation(Dataset):
    """
    WHU dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('WHU'),
                 split='train',
                 ):
        """
        :param base_dir: path to WHU dataset directory
        :param split: train/test/val
        :param transform: transform to apply
        """
        super().__init__()

        self.split = split   

        if args.use_small:
            base_dir=Path.db_root_dir('WHUsmall')
        self._base_dir = base_dir

        self.args = args
        self.images = sorted(glob(os.path.join(base_dir, split, "image", "*.tif")))
        self.labels = sorted(glob(os.path.join(base_dir, split, "label", "*.tif")))
        assert (len(self.images) == len(self.labels))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_path, target_path = self.images[index], self.labels[index]
        # print(img_path)
        _img = Image.open(img_path).convert('RGB')
        _target = Image.open(target_path).convert('L')
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            sample = self.transform_tr(sample)
            sample['label'][sample['label'] == 255] = 1
            return sample
        elif self.split == 'val':
            sample = self.transform_val(sample)
            sample['label'][sample['label'] == 255] = 1
            return sample
        elif self.split == 'test': 
            sample = self.transform_test(sample)
            sample['label'][sample['label'] == 255] = 1
            return sample

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

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)    

    def __str__(self):
        return 'WHU(split=' + str(self.split) + ')'


if __name__ == '__main__':
    import argparse

    import matplotlib.pyplot as plt
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader    
    

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.dataset = 'WHU'
    args.use_small = False

    train = WHUSegmentation(args, split='val')
    print('NUM_CLASSES = ', train.NUM_CLASSES)
    dataloader = DataLoader(train, batch_size=5, shuffle=False, num_workers=0)

    for ii, sample in enumerate(dataloader):
        
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            # print(np.unique(gt))
            tmp = np.array(gt[jj]).astype(np.uint8)
            # print(np.unique(gt[jj]))
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
    # plt.show()
    plt.savefig(f'display.{args.dataset}.png')
    print(f'display.{args.dataset}.png')


