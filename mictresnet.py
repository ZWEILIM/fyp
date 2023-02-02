
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


__all__ = ['MiCTResNet', 'MiCTBlock', 'get_mictresnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def _to_4d_tensor(x, depth_stride=None):

    x = x.transpose(0, 2)  # swap batch and depth dimensions: BxCxDxHxW => DxCxBxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxBxHxW => BxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: BxDxCxHxW => B*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: B*[1xDxCxHxW] => 1x(B*D)xCxHxW
    x = x.squeeze(0)  # 1x(B*D)xCxHxW => (B*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):

    x = torch.split(x, depth)  # (B*D)xCxHxW => B*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: BxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: BxDxCxHxW => BxCxDxHxW
    return x


# create ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MiCTResNet(nn.Module):


    def __init__(self, block, layers, dropout, n_classes, no_top=False,
                 first_3d_conv_depth=5, mict_3d_conv_depth=5,
                 t_strides=(1, 1, 1, 1, 1), t_padding='center', **kwargs):

        super(MiCTResNet, self).__init__(**kwargs)

        if t_padding not in ('forward', 'center'):
            raise ValueError('Invalid value for parameter `t_padding`: {}'.format(t_padding))

        self.inplanes = 64
        self.dropout = dropout
        self.t_strides = t_strides
        self.first_3d_conv_depth = first_3d_conv_depth
        self.t_padding = t_padding
        self.n_classes = n_classes
        self.no_top = no_top

        #Create 2d conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)
        #Create 3d conv
        self.conv2 = nn.Conv3d(3, 64, kernel_size=(first_3d_conv_depth, 7, 7),
                               stride=(t_strides[0], 2, 2),
                               padding=(first_3d_conv_depth//2, 3, 3) if t_padding == 'center' else 0,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=(t_strides[1], 2, 2), padding=1)
        self.relu = nn.ReLU(inplace=True)

        #Adding layer in MiCTBlock
        self.layer1 = MiCTBlock(block, self.inplanes, 64, layers[0], stride=(1, 1),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)
        self.layer2 = MiCTBlock(block, self.layer1.inplanes, 128, layers[1], stride=(t_strides[2], 2),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)
        self.layer3 = MiCTBlock(block, self.layer2.inplanes, 256, layers[2], stride=(t_strides[3], 2),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)
        self.layer4 = MiCTBlock(block, self.layer3.inplanes, 512, layers[3], stride=(t_strides[4], 1),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)

        # Check the no_top parameters
        if not self.no_top:
            self.avgpool1 = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.avgpool2 = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout3d(dropout)
            self.fc = nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #Transfer ResNet weights
    def transfer_weights(self, state_dict):

        for key in state_dict.keys():
            if key.startswith('conv1') | key.startswith('bn1'):
                eval('self.' + key + '.data.copy_(state_dict[\'' + key + '\'])')
            if key.startswith('layer'):
                var = key.split('.')
                if var[2] == 'downsample':
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].downsample[' + var[3] + '].' +
                         var[4] + '.data.copy_(state_dict[\'' + key + '\'])')
                else:
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].' + var[2] + '.' + var[3] +
                         '.data.copy_(state_dict[\'' + key + '\'])')

    #forward pass
    def forward(self, x):
        x = x.transpose(1, 2)  # BxDxCxHxW => BxCxDxHxW
        if self.t_padding == 'forward':
            out1 = F.pad(x, [3, 3, 3, 3, 0, 2*(self.first_3d_conv_depth//2)], 'constant', 0)
            out1 = self.conv2(out1)
        else:
            out1 = self.conv2(x)  # center padding
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool2(out1)

        x, depth = _to_4d_tensor(x, depth_stride=self.t_strides[1])
        out2 = self.conv1(x)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        out2 = self.maxpool1(out2)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if not self.no_top:
            out = self.avgpool1(out)
            out = out.squeeze(4).squeeze(3)
            out = self.drop(out)
            out_fc = []
            for i in range(out.size()[-1]):
                out_fc.append(self.fc(out[:, :, i]).unsqueeze(2))
            out_fc = torch.cat(out_fc, 2)
            out = self.avgpool2(out_fc).squeeze(2)

        return out


class MiCTBlock(nn.Module):

    def __init__(self, block, inplanes, planes, blocks, stride=(1, 1),
                 mict_3d_conv_depth=5, t_padding='center'):

        super(MiCTBlock, self).__init__()

        if t_padding not in ('forward', 'center'):
            raise ValueError('Invalid value for parameter `t_padding`: {}'.format(t_padding))

        downsample = None
        if stride[1] != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.blocks = blocks
        self.stride = stride
        self.mict_3d_conv_depth = mict_3d_conv_depth
        self.t_padding = t_padding
        self.bottlenecks = nn.ModuleList()
        self.bottlenecks.append(block(inplanes, planes, stride[1],
                                      downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.bottlenecks.append(block(self.inplanes, planes))

        self.conv = nn.Conv3d(inplanes, planes, kernel_size=(mict_3d_conv_depth, 3, 3),
                              stride=(stride[0], stride[1], stride[1]),
                              padding=(mict_3d_conv_depth//2, 1, 1) if t_padding == 'center' else 0,
                              bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        if self.t_padding == 'forward':
            out1 = F.pad(x, [1, 1, 1, 1, 0, 2*(self.mict_3d_conv_depth//2)], 'constant', 0)
            out1 = self.conv(out1)
        else:
            out1 = self.conv(x)
        out1 = self.bn(out1)
        out1 = self.relu(out1)

        x, depth = _to_4d_tensor(x, depth_stride=self.stride[0])
        out2 = self.bottlenecks[0](x)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2

        out, depth = _to_4d_tensor(out)
        for i in range(1, self.blocks):
            out = self.bottlenecks[i](out)
        out = _to_5d_tensor(out, depth)

        return out


def get_mictresnet(backbone, dropout=0.5, n_classes=101,
                   no_top=False, pretrained=True, **kwargs):

    if backbone == 'resnet18':
        model = MiCTResNet(BasicBlock, [2, 2, 2, 2], dropout,
                           n_classes, no_top, **kwargs)
        if pretrained:
            model.transfer_weights(model_zoo.load_url(model_urls['resnet18']))
    elif backbone == 'resnet34':
        model = MiCTResNet(BasicBlock, [3, 4, 6, 3], dropout,
                           n_classes, no_top, **kwargs)
        if pretrained:
            model.transfer_weights(model_zoo.load_url(model_urls['resnet34']))
    else:
        raise ValueError('Unknown backbone: {}'.format(backbone))

    return model
