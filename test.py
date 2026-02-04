import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import make_data_loader
from dataloaders.utils import decode_segmap
from modeling.roofxnet import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from tqdm import tqdm
from utils.metrics import Evaluator

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


def load_training_config(conf_path):
    dic = {}
    with open(conf_path, 'r') as f:
        confs = f.readlines()
        for conf in confs:
            k, v = conf.strip().split(':')
            dic[k] = v
    return dic
            

class Tester(object):
    def __init__(self, args):
        self.args = args
              
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        _, self.test_loader, _, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = RoofXNet(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        
        self.model = model        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        # load checkpoint
        if not os.path.isfile(args.checkpoint):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.checkpoint)
        if args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])                    
        info = "=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch'])
        
        print(info)


    def validation(self, args):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, x1, x2 = self.model(image)
              
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)            
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            
            if args.visualization:
                tgt = target[0].astype(np.uint8)
                tgt_segmap = decode_segmap(tgt, dataset=args.dataset)
                pred = pred[0].astype(np.uint8)
                pred_segmap = decode_segmap(pred, dataset=args.dataset)            
                plt.figure()
                plt.title('display')
                plt.subplot(211)
                plt.imshow(pred_segmap)
                plt.subplot(212)
                plt.imshow(tgt_segmap)
                plt.savefig(f'test.{args.dataset}.png')
    
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # 
        Recall = self.evaluator.Recall()
        Precision = self.evaluator.Precision()
        F1 = self.evaluator.F_score()
                
        print('Test:')
        print('numImages: %5d]' % (i * self.args.test_batch_size + image.data.shape[0]))
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Acc:{}, mIoU:{}, fwIoU: {}".format(Acc, mIoU, FWIoU))
        print("Recall:{}, Precision:{}, F1-score:{}".format(Recall, Precision, F1))


def main():
    parser = argparse.ArgumentParser(description="RoofX-Net Testing")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='rooftop',
                        choices=['rooftop', 'WHU'],
                        help='dataset name (default: rooftop)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--no-binary-cls', action='store_true', default=False,
                        help='Only for rooftop dataset, if --no-binary-cls means use all 4 cls, else use 3 cls')
    parser.add_argument('--use-small', action='store_true', default=False,
                        help='Test on Small Scale Rooftops')    
    
    # testing hyper params
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    parser.add_argument('--training_config', type=str, default=None)
    parser.add_argument('--expname', type=str, default='test-default',
                        help='set the subname of the experiment')
    parser.add_argument('--visualization', '-vis', action='store_true', default=False)

    args = parser.parse_args()
    
    if args.training_config is not None:
        training_config = load_training_config(args.training_config)
        # print(training_config)        
        if 'dataset' in training_config:
            args.dataset = training_config['dataset']
        if 'backbone' in training_config:
            args.backbone = training_config['backbone']
        if 'out_stride' in training_config:
            args.out_stride = int(training_config['out_stride'])   
        if 'base_size' in training_config:
            args.base_size = int(training_config['base_size'])
        if 'crop_size' in training_config:
            args.crop_size = int(training_config['crop_size'])
        if 'seed' in training_config:
            args.seed = int(training_config['seed'])
        
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    
    if args.test_batch_size is None:
        args.test_batch_size = 4 * len(args.gpu_ids)
    args.batch_size = args.test_batch_size
    
    print(args)
    torch.manual_seed(args.seed)
    
    tester = Tester(args)
    print('Starting Testing:')        
    tester.validation(args)

if __name__ == "__main__":
   main()
