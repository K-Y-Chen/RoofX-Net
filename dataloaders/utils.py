import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'sbd':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'guangfu':
        n_classes = 3
        label_colours = get_guangfu_labels()
    elif dataset == 'dubai':
        n_classes = 6
        label_colours = get_dubai_labels()
    elif dataset == 'gid15':
        n_classes = 15+1
        label_colours = get_gid15_labels()
    elif dataset == 'inria_ail':
        n_classes = 1+1
        label_colours = get_inria_ail_labels()
    elif dataset == 'WHU':
        n_classes = 2
        label_colours = get_WHU_labels()
    elif dataset == 'Mass':
        n_classes = 2
        label_colours = get_Massachusetts_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])
    

def get_guangfu_labels():
    return np.array([
        [0, 0, 0],          # background
        [128, 128, 0],      # color steel        
        [128, 0, 0],        # normal
        [0, 128, 0],        # slope 
    ])
    
def get_dubai_labels():
    return np.array([
        [155, 155, 155],    # "Unlabeled"
        [226, 169, 41],     # "Water" 
        [132, 41, 246],     # Land (unpaved area) 
        [110, 193, 228],    # "Road"
        [60, 16, 152],      # "Building"
        [254, 221, 58],     # "Vegetation"
    ])
    
def get_gid15_labels():
    return np.array([
        [0, 0, 0],          # background
        [200, 0, 0],        # industrial land
        [250, 0, 150],      # urban residential 
        [200, 150, 150],    # rural residential 
        [250, 150, 150],    # traffic land
        [0, 200, 0],        # paddy field
        [150, 250, 0],      # irrigated land
        [150, 200, 150],    # dry cropland
        [200, 0, 200],      # garden plot 
        [150, 0, 250],      # arbor woodland 
        [150, 150, 250],    # shrub land
        [250, 200, 0],      # natural grassland
        [200, 200,  0],     # artificial grassland
        [0, 0, 200],        # river
        [0, 150, 200],      # lake 
        [0, 200, 250],      # pond        
    ])

def get_inria_ail_labels():
    return np.array([
        [0, 0, 0],          # background
        [255, 255, 255],    # building
    ])

def get_WHU_labels():
    return np.array([
        [0, 0, 0],          # background
        [255, 255, 255],    # building        
    ])

def get_Massachusetts_labels():
    return np.array([
        [0, 0, 0],          # background
        [255, 255, 255],    # building        
    ])