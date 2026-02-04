class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'rooftop':
            return '/path/to/datasets/rooftopplus/splits512/'
        elif dataset == 'guangfusmall':
            return '/path/to/datasets/rooftopplus/cropped_region/'
        elif dataset == 'WHU':
            return '/path/to/WHU/'
        elif dataset == 'WHUsmall':
            return '/path/to/WHU/cropped_region/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError