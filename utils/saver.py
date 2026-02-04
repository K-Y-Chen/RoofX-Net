import os
import torch
from collections import OrderedDict
from time import asctime


class Saver(object):

    def __init__(self, args):
        self.args = args        
        self.logger_time = asctime()
        self.experiment_dir = os.path.join('checkpoints', args.dataset, args.backbone, args.expname, self.logger_time)
        
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, suffix='_checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        
        epoch = state['epoch']
        best_pred = state['best_pred']
        loss = state['loss']
        
        if state['is_train']:
            save_file = f'[Epoch {epoch}]_[Train]_loss[{loss:.5f}]_bestMiou[{best_pred:.5f}]' + suffix
        else:
            save_file = f'[Epoch {epoch}]_[Eval]_loss[{loss:.5f}]_bestMiou[{best_pred:.5f}]' + suffix
            
        save_path = os.path.join(self.experiment_dir, save_file)        
        torch.save(state, save_path)
        
        if is_best:
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a+') as f:
                f.write(f'[Epoch {epoch}]: {best_pred}\n')
                
                
    def save_last_checkpoint(self, state):
        """Saves checkpoint to disk"""        
        save_file = 'last_checkpoint.pth.tar'              
        save_path = os.path.join(self.experiment_dir, save_file)        
        torch.save(state, save_path)
        

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epochs'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size
        p['seed'] = self.args.seed
        # p['edge'] = self.args.edge
        p['alpha'] = self.args.alpha
        p['beta'] = self.args.beta
        

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
        
    def save_log(self, info):
        logfile = os.path.join(self.experiment_dir, 'logger.txt')
        logfile = open(logfile, 'a+')
        logfile.write(info + '\n')