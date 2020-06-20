
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math

model_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, rate=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, rate=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_pretrained_model()
            # print('load the pre-trained model.')
            # resnet = models.resnet101(pretrained)
            # self.conv1 = resnet.conv1
            # self.bn1 = resnet.bn1
            # self.layer1 = resnet.layer1
            # self.layer2 = resnet.layer2

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, rate=rate))

        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        new_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        pretrain_dict = new_model.state_dict()
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                # print(k)
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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
    def __init__(self, num_classes, pretrained=True):
        super(DeepLab, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 6, 3], pretrained=True)

        self.atrous = Atrous_module(2048 , num_classes, rate=12)
        
    def forward(self, x):
        x_size = x.size()
        x = self.resnet_features(x)
        x = self.atrous(x)
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)

        return x 


    def get_1x_lr_params(self):
        modules = [self.resnet_features]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.atrous]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


