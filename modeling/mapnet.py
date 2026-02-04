import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(input, filters, kernel_size=3, stride=1, padding='same'):
    return nn.Conv2d(input, filters, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

def bn(input):
    return nn.BatchNorm2d(input, momentum=0.99, eps=1e-3, affine=True)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Bottleneck, self).__init__()
        # print(in_channels, out_channels, stride, downsample)
        self.bn = bn(in_channels)
        self.conv1 = conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding='valid')
        self.bn1 = bn(out_channels)
        self.conv2 = conv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = bn(out_channels)
        self.conv3 = conv2d(out_channels, out_channels*4, kernel_size=1, padding='valid')
        self.bn3 = bn(out_channels*4)
        self.downsample = downsample
        if self.downsample:
            self.residual = nn.Sequential(
                bn(in_channels),
                nn.ReLU(),
                conv2d(in_channels, out_channels*4, kernel_size=1, padding='valid')
            )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)
        if self.downsample:
            residual = self.residual(x)
            # print(residual.shape)
        out = out + residual
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.bn = bn(in_channels)
        self.conv1 = conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.bn1 = bn(out_channels)
        self.conv2 = conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn2 = bn(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return out
    

class TransLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransLayer, self).__init__()
        self.num_in = len(in_channels)
        self.num_out = len(out_channels)
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_out):
            if i < self.num_in:
                conv_layer = nn.Sequential(
                    bn(in_channels[i]),
                    nn.ReLU(),
                    conv2d(in_channels[i], out_channels[i], kernel_size=3, padding=1)
                )
                self.conv_layers.append(conv_layer)
            else:
                conv_layer = nn.Sequential(
                    bn(in_channels[-1]),
                    nn.ReLU(),
                    conv2d(in_channels[-1], out_channels[i], kernel_size=3, stride=2, padding=1)
                )
                self.conv_layers.append(conv_layer)

    def forward(self, x):
        output = []
        for i in range(self.num_out):
            if i < self.num_in:
                out = self.conv_layers[i](x[i])
                output.append(out)
            else:
                out = self.conv_layers[i](x[-1])
                output.append(out)
        return output


class ConvB(nn.Module):
    def __init__(self, block_num, channels):
        super(ConvB, self).__init__()
        self.block_num = block_num
        self.channels = channels

        self.res_blocks = nn.ModuleList()
        for i in range(len(channels)):
            res_blocks_for_channel = nn.ModuleList()
            for j in range(block_num):
                res_blocks_for_channel.append(ResBlock(channels[i], channels[i]))
            self.res_blocks.append(res_blocks_for_channel)

    def forward(self, x):
        out = []
        for i in range(len(self.channels)):
            residual = x[i]
            for j in range(self.block_num):
                residual = self.res_blocks[i][j](residual)
            out.append(residual)
        return out



class FeatFuse(nn.Module):
    def __init__(self, channels, multi_scale_output=True):
        super(FeatFuse, self).__init__()
        self.channels = channels
        self.multi_scale_output = multi_scale_output

        # Initialize BatchNorm and Conv2d layers here
        self.bns = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])
        self.convs = nn.ModuleList([nn.Conv2d(ch, ch, 1) for ch in channels])

    def forward(self, x):
        out = []
        for i in range(len(self.channels) if self.multi_scale_output else 1):
            residual = x[i]
            for j in range(len(self.channels)):
                if j > i:
                    if not self.multi_scale_output:
                        y = self.bns[j](x[j])
                        y = F.relu(y)
                        y = self.convs[j](y)
                        y = F.interpolate(y, scale_factor=2 ** (j - i), mode='bilinear', align_corners=False)
                        out.append(y)
                    else:
                        y = self.bns[j](x[j])
                        y = F.relu(y)
                        y = self.convs[i](y)
                        y = F.interpolate(y, scale_factor=2 ** (j - i), mode='bilinear', align_corners=False)
                        residual = residual + y

                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            y = self.bns[j](y)
                            y = F.relu(y)
                            y = self.convs[i](y)
                            y = F.max_pool2d(y, 2, 2)

                        else:
                            y = self.bns[j](y)
                            y = F.relu(y)
                            y = self.convs[j](y)
                            y = F.max_pool2d(y, 2, 2)

                    residual = residual + y
            out.append(residual)
        return out





class ConvBlock(nn.Module):
    def __init__(self, channels, multi_scale_output=True):
        super(ConvBlock, self).__init__()
        self.channels = channels
        self.multi_scale_output = multi_scale_output
        self.residual_conv = ConvB(4, channels)  # Assuming convb is a custom convolutional block
        # print('ConvBlock', channels)
        self.feat_fuse = FeatFuse(channels, multi_scale_output)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.feat_fuse(residual)
        return out

class Stage(nn.Module):
    def __init__(self, num_modules, channels, multi_scale_output=True):
        super(Stage, self).__init__()
        self.num_modules = num_modules
        self.channels = channels
        self.multi_scale_output = multi_scale_output
        self.conv_blocks = nn.ModuleList()
        
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                self.conv_blocks.append(ConvBlock(channels, multi_scale_output=False))
            else:
                self.conv_blocks.append(ConvBlock(channels))

    def forward(self, x):
        out = x
        for i in range(self.num_modules):
            out = self.conv_blocks[i](out)
        return out



class PyramidPoolingBlock(nn.Module):
    def __init__(self, input_channels, bin_sizes):
        super(PyramidPoolingBlock, self).__init__()
        self.bin_sizes = bin_sizes
        self.input_channels = input_channels
        self.pooled_outputs = nn.ModuleList()
        for bin_size in bin_sizes:
            self.pooled_outputs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((bin_size, bin_size)),
                conv2d(input_channels, input_channels // 4, kernel_size=1)
            ))

    def forward(self, x):
        pooled_results = [x]
        _, _, h, _ = x.size()
        for pool_layer in self.pooled_outputs:
            pooled_feature = pool_layer(x)
            upsampled_feature = F.interpolate(pooled_feature, size=(h, h), mode='bilinear', align_corners=True)
            pooled_results.append(upsampled_feature)
        output = torch.cat(pooled_results, dim=1)
        return torch.add(x, output)


def spatial_pooling(input):
    h, w = input.shape[2], input.shape[3]
    p1 = F.interpolate(F.max_pool2d(input, 2, 2), size=(h, w), mode='bilinear', align_corners=True)
    p2 = F.interpolate(F.max_pool2d(input, 3, 3), size=(h, w), mode='bilinear', align_corners=True)
    p3 = F.interpolate(F.max_pool2d(input, 5, 5), size=(h, w), mode='bilinear', align_corners=True)
    p4 = F.interpolate(F.max_pool2d(input, 6, 6), size=(h, w), mode='bilinear', align_corners=True)
    p = torch.cat([p1, p2, p3, p4, input], dim=1)
    return p

class ChannelSqueeze(nn.Module):
    def __init__(self, filters, name=""):
        super(ChannelSqueeze, self).__init__()
        self.name = name
        self.filters = filters
        self.fc1 = nn.Linear(filters, filters)
        self.fc2 = nn.Linear(filters, filters)

    def forward(self, input):
        squeeze = torch.mean(input, dim=[2, 3], keepdim=True)
        fc1 = F.relu(self.fc1(squeeze.view(squeeze.size(0), -1)))
        fc2 = torch.sigmoid(self.fc2(fc1))
        result = fc2.view(-1, self.filters, 1, 1)
        return input * result
    
class Stage0(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(Stage0, self).__init__()
        self.bottleneck1 = Bottleneck(in_channels, out_channels, downsample=True)
        self.bottleneck2 = Bottleneck(64 * 4, out_channels)
        self.bottleneck3 = Bottleneck(64 * 4, out_channels)
        self.bottleneck4 = Bottleneck(64 * 4, out_channels)

    def forward(self, x):
        x = self.bottleneck1(x)
        # print(x.shape)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        return x
    
class MapNet(nn.Module):
    def __init__(self, num_cls=4):
        super(MapNet, self).__init__()
        self.channels_s2 = [64, 64]
        self.channels_s3 = [64, 64, 64]
        self.num_modules_s2 = 2
        self.num_modules_s3 = 3
        self.num_cls = num_cls

        # Define layers used in the network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # # Define stages and translayers
        self.stage0 = Stage0(64, 64)
        self.translayer1 = TransLayer([256], self.channels_s2)
        # def __init__(self, num_modules, channels, multi_scale_output=True):
        
        self.stage1 = Stage(self.num_modules_s2, self.channels_s2)
        self.translayer2 = TransLayer(self.channels_s2, self.channels_s3)
        self.stage2 = Stage(self.num_modules_s3, self.channels_s3, multi_scale_output=False)

        # Additional layers
        self.channel_squeeze = ChannelSqueeze(192)
        
        # Spatial pooling
        self.spatial_pooling = spatial_pooling
        self.bn4 = nn.BatchNorm2d(832)

        # # Final convolutions for output
        self.final_conv1 = nn.Conv2d(832, 128, kernel_size=1, padding=0)
        self.final_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        )
        self.final_upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )
        self.final_conv2 = nn.Conv2d(32, self.num_cls, kernel_size=1)

    def forward(self, x):
        conv1 = self.bn1(self.conv1(x))
        conv1 = F.relu(conv1)
        conv2 = self.bn2(self.conv2(conv1))
        conv2 = F.relu(conv2)
        conv3 = self.bn3(self.conv3(conv2))
        conv3 = F.relu(conv3)
        pool = self.pool(conv3)

        stage1 = self.stage0(pool)
        trans1 = self.translayer1([stage1])
        stage2 = self.stage1(trans1)
        
        trans2 = self.translayer2(stage2)
        stage3 = self.stage2(trans2)
        
        stg3 = torch.cat(stage3, dim=1)
        
        
        squeeze = self.channel_squeeze(stg3)
        

        spatial = torch.cat([stage3[0], stage3[1]], dim=1)
        spatial = self.spatial_pooling(spatial)
        

        new_feature = torch.cat([spatial, squeeze], dim=1)
        new_feature = self.bn4(new_feature)
        new_feature = F.relu(new_feature)
        
        result = self.final_conv1(new_feature)
        

        up1 = self.final_upsample1(result)
        up2 = self.final_upsample2(up1)
        final = self.final_conv2(up2)
        # print(final.shape)

        return final

if __name__ == '__main__':
    image = torch.randn(2, 3, 512, 512).cuda()
    net = MapNet().cuda()
    out = net(image)
    print(out.shape)