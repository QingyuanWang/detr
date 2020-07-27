import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from timm.models.resnest import resnest101e

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .position_encoding import PositionEmbeddingSine


def resnest_fpn_backbone(pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    # resnet_backbone = resnet.__dict__['resnet152'](pretrained=pretrained,norm_layer=norm_layer)
    backbone = resnest101e(pretrained=pretrained)
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


class ResnestBackBone(nn.Module):
    def __init__(self):
        super(ResnestBackBone, self).__init__()
        self.resnest = resnest_fpn_backbone(pretrained=True)
        self.weights = nn.Parameter(torch.ones(5))

        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p6_upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.p7_upsample = nn.Upsample(scale_factor=8, mode='nearest')

        self.p3_downsample = nn.MaxPool2d(kernel_size=2)
        # self.p4_downsample = nn.MaxPool2d(kernel_size=2)

        self.position_encoding = PositionEmbeddingSine(128, normalize=True)
        self.freeze_bn()

    def forward(self, inputs):
        # P5 (32*32) is the target.
        img_batch = inputs.tensors

        resnest_out = self.resnest(img_batch)
        print(resnest_out)
        # p3 = self.conv3(c3)
        # p4 = self.conv4(c4)
        # p5 = self.conv5(c5)
        # p6 = self.conv6(c5)
        # p7 = self.conv7(p6)

        # p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn([p3, p4, p5, p6, p7])

        # # weights = F.softmax(self.weights, dim=0)
        # weights = self.weights / torch.sum(self.weights)

        # features = weights[0] * self.p3_downsample(p3) + weights[1] * p4 + weights[2] * self.p5_upsample(
        #     p5) + weights[3] * self.p6_upsample(p6) + weights[4] * self.p7_upsample(p7)

        # features = nested_tensor_from_tensor_list(features)
        # pos = self.position_encoding(features).to(features.tensors.dtype)

        # return [features], [pos]
