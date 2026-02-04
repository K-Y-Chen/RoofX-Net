import os
import numpy as np
from config import Path
from dataloaders import custom_transforms as tr
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RooftopPlusSegmentation(Dataset):
    """
    RooftopPlus dataset
    """    
    classes = {
        "background":           {"rgb": (0, 0, 0)},
        "color steel":          {"rgb": (128, 128, 0)},
        "normal":               {"rgb": (128, 0, 0)},
        "slope":                {"rgb": (0, 128, 0)},           
    }
    
    colors = [v["rgb"] for k, v in classes.items()]

    def __init__(self, args, base_dir=Path.db_root_dir('rooftop'), split='train'):
        """
        :param base_dir: path to dataset directory
        :param split: train/val/custom
        :param transform: transform to apply
        """        
        super().__init__()
        self.args = args
        self.binary_cls = not self.args.no_binary_cls
        self.NUM_CLASSES = 2+1 if self.binary_cls else 3+1
        # print(self.NUM_CLASSES)
        self.base_dir = base_dir  
        if args.use_small:
            self.base_dir = base_dir.replace('splits512', 'cropped_region')

        self.split = split

        self.im_ids = []
        self.images = []
        self.categories = []

        # with open(os.path.join(os.path.join(self.base_dir, self.split + 'list_fewer_0.4.txt')), "r") as f:
        #     lines = f.read().splitlines()
        
        if self.split == 'train':
            with open(os.path.join(os.path.join(self.base_dir, self.split + 'list.txt')), "r") as f:
                lines = f.read().splitlines()
        elif self.split == 'val':
            with open(os.path.join(os.path.join(self.base_dir, self.split + 'list.txt')), "r") as f:
                lines = f.read().splitlines()        
        else:
            with open(self.split, "r") as f:
                lines = f.read().splitlines()
            
        for _, line in enumerate(lines):
            _image = os.path.join(self.base_dir, "img_" + line + ".png")
            _cat = os.path.join(self.base_dir, "label_" + line + ".png")
            # print(_image)
            # print(_cat)
            # exit()
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(self.split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        # print(sample['image'].size)
        # print(sample['label'].size)
        if self.split == "train":
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)
    
    def rgb_to_mask(self, rgb, colors):
        h, w = rgb.shape[:2]
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        for i, c in enumerate(colors):
            cmask = (rgb == c)
            if isinstance(cmask, np.ndarray):
                mask[cmask.all(axis=-1)] = i

        return mask

    def _make_img_gt_point_pair(self, index):
        # 读取PNG图片
        img = Image.open(self.images[index]).convert('RGB')
        # _target = Image.open(self.categories[index]).convert('L')
        _target = np.array(Image.open(self.categories[index]).convert('RGB'))
        _target = self.rgb_to_mask(_target, self.colors)
        if self.binary_cls:
            _target[_target > 2] = 2
        target = Image.fromarray(_target)
       
        return img, target

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
        return 'rooftop(split=' + str(self.split) + ')'


if __name__ == '__main__':
    import argparse

    import matplotlib.pyplot as plt
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader    
    

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 448
    args.dataset = 'rooftop'

    train = RooftopPlusSegmentation(args, split='train', binary_cls=True)
    print('NUM_CLASSES = ', train.NUM_CLASSES)
    dataloader = DataLoader(train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            # print(np.unique(gt))
            # print(gt.shape)
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
    # plt.show()
    plt.savefig(f'display.{args.dataset}.png')


