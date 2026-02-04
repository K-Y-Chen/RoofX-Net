import sys

import torch

sys.path.append("/mnt/disk/cv/KeyuChen/pytorch-deeplab-xception")
from swin_transformer import *

net = swin_custom(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=24,
    head_dim=32,
    window_size=7,
    downscaling_factors=(2, 2, 1, 1),
    relative_pos_embedding=True
)
dummy_x = torch.randn(1, 3, 448, 448)
logits = net(dummy_x)  # (1,3)
# print(net)
print(logits.shape)