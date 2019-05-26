import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import SSnbt, Downsample_Block_led, APN


class LEDNet(nn.Module):
    '''
    Implementation of the LEDNet network
    '''

    def __init__(self, in_channels=3, n_classes=22, encoder_relu=False, decoder_relu=True, image_dim=128):
        super().__init__()

        ### Encoder
        self.downsample_1 = Downsample_Block_led(in_channels, 32, kernel_size=3, padding=1, relu=encoder_relu)

        self.ssnbt_1_1 = SSnbt(32, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.ssnbt_1_2 = SSnbt(32, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.ssnbt_1_3 = SSnbt(32, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)

        self.downsample_2 = Downsample_Block_led(32, 64, kernel_size=3, padding=1, relu=encoder_relu)

        self.ssnbt_2_1 = SSnbt(64, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.ssnbt_2_2 = SSnbt(64, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)

        self.downsample_3 = Downsample_Block_led(64, 128, kernel_size=3, padding=1, relu=encoder_relu)

        self.ssnbt_3_1 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=1, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_2 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=2, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_3 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=5, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_4 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=9, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_5 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=2, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_6 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=5, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_7 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=9, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_8 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=17, relu=encoder_relu, drop_prob=0.1)

        ### Decoder
        self.apn = APN(128, n_classes, image_dim= image_dim // 8, relu=decoder_relu)
    
    def forward(self, x):

        ### Encoder
        x = self.downsample_1(x)
        x = self.ssnbt_1_1(x)
        x = self.ssnbt_1_2(x)
        x = self.ssnbt_1_3(x)

        x = self.downsample_2(x)
        x = self.ssnbt_2_1(x)
        x = self.ssnbt_2_2(x)

        x = self.downsample_3(x)
        x = self.ssnbt_3_1(x)
        x = self.ssnbt_3_2(x)
        x = self.ssnbt_3_3(x)
        x = self.ssnbt_3_4(x)
        x = self.ssnbt_3_5(x)
        x = self.ssnbt_3_6(x)
        x = self.ssnbt_3_7(x)
        x = self.ssnbt_3_8(x)

        ### Decoder
        out = self.apn(x)

        return out


def return_model(input_nc=3, output_nc=22, netG = 'lednet_128'):

    if netG == 'lednet_128':
        gen_net = LEDNet(in_channels=input_nc, n_classes= output_nc, encoder_relu=False, decoder_relu=True, image_dim=128)
    elif netG == 'lednet_256':
        gen_net = LEDNet(in_channels=input_nc, n_classes= output_nc, encoder_relu=False, decoder_relu=True, image_dim=256)

    return gen_net