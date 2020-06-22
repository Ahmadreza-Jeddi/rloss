
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math
from collections import OrderedDict
from modeling.selpsup.model_factory import model_factory



def conv3x3_relu(inplanes, planes, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, 
                                    stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class VGG16_feature(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_feature, self).__init__()

        self.features = self.make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'MN', 513, 513, 513, 'MN'])
        # self.features2 = nn.Sequential(conv3x3_relu(512, 512, rate=2),
        #                                conv3x3_relu(512, 512, rate=2),
        #                                conv3x3_relu(512, 512, rate=2),
        #                                nn.MaxPool2d(3, stride=1, padding=1))

        
        # if pretrained:
            # url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth' --> imagenet
            # # weight  = model_zoo.load_url(url)
            # url = 'modeling/model-checkpoints/vgg16-bn-deeperCluster-yfcc.pth'
            
            # weight = torch.load(url)['state_dict']
            # weight2 = OrderedDict()
            # for key in list(weight.keys()):
            #     if 'features' in key:
            #         print(key)
            #         weight2[key[21:]] = weight[key]

            ###              #ImageNet#                ###
            # url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
            # weight = model_zoo.load_url(url)
            
            # weight2 = OrderedDict()
            # for key in list(weight.keys()):
            #     if 'features' in key:
            #         print(key)
            #         weight2[key[9:]] = weight[key]


            # self.features.load_state_dict(weight2)

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'MN':
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            elif v == 513:
                conv2d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

        

    def forward(self, x):
        x = self.features(x)
        # x = self.features2(x)

        return x



class Atrous_module(nn.Module):
    def __init__(self, inplanes, num_classes, rate):
        super(Atrous_module, self).__init__()
        planes = inplanes
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.fc1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class DeepLab(nn.Module):
    def __init__(self, num_classes, small=False, pretrained=False):
        super(DeepLab, self).__init__()
        # self.vgg_feature = VGG16_feature(pretrained=True)
        self.vgg_feature = model_factory(sobel=False)

        # url = 'modeling/model-checkpoints/vgg16-rotnet_flickr.pth'
        # weight = torch.load(url)['state_dict']
        # weight2 = OrderedDict()
        # for key in list(weight.keys()):
            
        #     if 'body.features' in key:
        #         print(key)
        #         weight2[key[7:]] = weight[key]
        # self.vgg_feature.load_state_dict(weight2)


        if small:
            rates = [2, 4, 8, 12]
        else:
            rates = [6, 12, 18, 24]
        self.aspp1 = Atrous_module(512 , num_classes, rate=rates[0])
        self.aspp2 = Atrous_module(512 , num_classes, rate=rates[1])
        self.aspp3 = Atrous_module(512 , num_classes, rate=rates[2])
        self.aspp4 = Atrous_module(512 , num_classes, rate=rates[3])
        
    # def forward(self, x):
    #     x_size = x.size()
    #     x = self.vgg_feature(x)
    #     x = self.atrous(x)
    #     x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)

    #     return x 

    def get_1x_lr_params(self):
        modules = [self.vgg_feature]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp1, self.aspp2, self.aspp3, self.aspp4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

        
    def forward(self, x):
        x_size = x.size()
        x = self.vgg_feature(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = x1 + x2 + x3 + x4
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)

        return x 
