from dataloaders.datasets import (WHU, Massachusetts, cityscapes, coco,
                                  combine_dbs, dubai, gid15, inria_ail, pascal, rooftop, sbd)
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'rooftop':
        train_set = rooftop.RooftopPlusSegmentation(args, split='train')
        val_set = rooftop.RooftopPlusSegmentation(args, split='val')
        test_set = None
        num_class = train_set.NUM_CLASSES
        if not args.use_small:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            train_loader = None
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'dubai':
        train_set = dubai.DubaiSegmentation(args, split='train')
        val_set = dubai.DubaiSegmentation(args, split='val')
        
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'gid15':
        train_set = gid15.GID15(args, split='train')
        val_set = gid15.GID15(args, split='val')
        
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'inria_ail':
        train_set = inria_ail.InriaAIL(args, split='train')
        val_set = inria_ail.InriaAIL(args, split='val')
        
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'WHU':
        train_set = WHU.WHUSegmentation(args, split='train')
        val_set = WHU.WHUSegmentation(args, split='val')
        test_set = WHU.WHUSegmentation(args, split='test')
        
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'Mass':
        train_set = Massachusetts.MassachusettsSegmentation(args, split='train')
        val_set = Massachusetts.MassachusettsSegmentation(args, split='val')
        test_set = Massachusetts.MassachusettsSegmentation(args, split='test')
        
        num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,  **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class

    else:
        print('Using dataset: ', args.dataset)
        raise NotImplementedError

