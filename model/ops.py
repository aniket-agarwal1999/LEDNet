import torch.nn as nn
import torch
import torch.nn.functional as F

'''
So here writing up the helper functions for the case of LEDNet
'''

class Channel_Split(nn.Module):
    '''
    Used to split the channels of a tensor
    '''
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
    
    def forward(self, x):
        x_left = x[:, 0:self.channels // 2, :, :]
        x_right = x[:, self.channels // 2 : , :, :]

        return x_left, x_right


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class SSnbt(nn.Module):
    '''
    The class for the implementation split-shuffle-non-bottleneck block
    '''

    def __init__(self, channels, kernel_size = 3, padding = 0, groups=4, dilation=1, bias= True, relu=False, drop_prob=0):
        '''
        groups: parameter for determining the shuffling order in the shuffleBlock
        '''
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        ### Now firstly there will be channel split and hence we have to write for left and right parts of
        ### our model

        ### Also in this model the convolutions are asymmetric in nature

        mid_channels = channels // 2

        self.split = Channel_Split(channels)

        self.l_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation = 1, bias=bias)

        self.l_conv2 = nn.Sequential(
                            nn.Conv2d(mid_channels, mid_channels, kernel_size = (1, kernel_size), stride=1, padding=(0, padding), dilation=1, bias=bias),
                            nn.BatchNorm2d(mid_channels),
                            activation
        )

        self.r_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size = (1, kernel_size), stride=1, padding=(0, padding), dilation=1, bias=bias)

        self.r_conv2 = nn.Sequential(
                            nn.Conv2d(mid_channels, mid_channels, kernel_size = (kernel_size, 1), stride=1, padding=(padding, 0), dilation=1, bias=bias),
                            nn.BatchNorm2d(mid_channels),
                            activation
        )

        self.l_conv3 = nn.Sequential(
                        nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding + dilation - 1, 0), dilation = dilation, bias=bias),
                        activation
        )

        self.l_conv4 = nn.Sequential(
                            nn.Conv2d(mid_channels, mid_channels, kernel_size = (1, kernel_size), stride=1, padding=(0, padding + dilation - 1), dilation=dilation, bias=bias),
                            nn.BatchNorm2d(mid_channels),
                            activation
        )

        self.r_conv3 = nn.Sequential(
                        nn.Conv2d(mid_channels, mid_channels, kernel_size = (1, kernel_size), stride=1, padding=(0, padding + dilation - 1), dilation=dilation, bias=bias),
                        activation
        )

        self.r_conv4 = nn.Sequential(
                            nn.Conv2d(mid_channels, mid_channels, kernel_size = (kernel_size, 1), stride=1, padding=(padding + dilation - 1, 0), dilation=dilation, bias=bias),
                            nn.BatchNorm2d(mid_channels),
                            activation
        )

        self.regularizer = nn.Dropout2d(p=drop_prob)
        
        self.out_activation = activation
        self.shuffle = ShuffleBlock(groups=groups)
    
    def forward(self, x):
        main = x
    
        l_input, r_input = self.split(x)

        l_input =  self.l_conv1(l_input)
        l_input = self.l_conv2(l_input)
        r_input = self.r_conv1(r_input)
        r_input = self.r_conv2(r_input)

        l_input =  self.l_conv3(l_input)
        l_input = self.l_conv4(l_input)
        r_input = self.r_conv3(r_input)
        r_input = self.r_conv4(r_input)
        
        ext = torch.cat((l_input, r_input), 1)

        ext = self.regularizer(ext)

        out = main + ext
        out = self.out_activation(out)
        out = self.shuffle(out)

        return out


class Downsample_Block_led(nn.Module):
    '''
    The downsampling block for the LEDNet

    So basically we will do two operations in parallel, conv and max pool and then concat both of them
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, bias=True, relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        
        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_activation = activation
    
    def forward(self, x):
        main = self.conv(x)
        ext = self.maxpool(x)

        out = torch.cat((main, ext), 1)
        out = self.out_activation(out)
        return out

class APN(nn.Module):
    '''
    This is the Attention Pyramid Network including the whole upsampling block
    '''

    def __init__(self, in_channels, out_channels, image_dim, bias=True, relu=True):
        '''
        image_dim = dimension of the incoming image
        '''
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.conv_33 = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                            nn.BatchNorm2d(in_channels),
                            activation
        )

        self.conv_55 = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, bias=bias),
                            nn.BatchNorm2d(in_channels),
                            activation
        )

        self.conv_77 = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=2, padding=3, bias=bias),
                            nn.BatchNorm2d(in_channels),
                            activation
        )

        self.conv_77_toC = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.conv_55_toC = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.conv_33_toC = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.conv_11 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.global_pool = nn.AvgPool2d(kernel_size = (image_dim, image_dim), stride=0, padding=0)
        self.pool_conv_toC = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.upsample_times2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_globalPool = nn.UpsamplingBilinear2d(size = (image_dim, image_dim))
    
    def forward(self, x):
        output_conv33 = self.conv_33(x)
        
        output_conv55 = self.conv_55(output_conv33)

        output_conv77 = self.conv_77(output_conv55)
        output_conv77_toC = self.conv_77_toC(output_conv77)
        output_conv77_upsample = self.upsample_times2(output_conv77_toC)

        output_conv55_toC = self.conv_55_toC(output_conv55)
        output_conv55_toC = output_conv55_toC + output_conv77_upsample
        output_conv55_upsample = self.upsample_times2(output_conv55_toC)

        output_conv33_toC = self.conv_33_toC(output_conv33)
        output_conv33_toC = output_conv33_toC + output_conv55_upsample
        output_conv33_upsample = self.upsample_times2(output_conv33_toC)

        output_conv11 = self.conv_11(x)

        output_global_pool = self.global_pool(x)
        output_pool_conv_toC = self.pool_conv_toC(output_global_pool)
        output_pool_upsample = self.upsample_globalPool(output_pool_conv_toC)

        output_conv11 = output_conv11 * output_conv33_upsample
        output_conv11 = output_conv11 + output_pool_upsample

        final_output = F.upsample_bilinear(output_conv11, scale_factor=8)

        return final_output 