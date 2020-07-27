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
        self.weights = nn.Parameter(torch.ones(4))

        self.p2_downsample = nn.MaxPool2d(kernel_size=2)
        self.p1_downsample = nn.MaxPool2d(kernel_size=4)
        self.p0_downsample = nn.MaxPool2d(kernel_size=8)
        # self.p4_downsample = nn.MaxPool2d(kernel_size=2)

        self.position_encoding = PositionEmbeddingSine(128, normalize=True)

    def forward(self, inputs):
        # P5 (32*32) is the target.
        img_batch = inputs.tensors

        resnest_out = self.resnest(img_batch)
        weights = self.weights / torch.sum(self.weights)

        features = weights[0] * self.p0_downsample(resnest_out['0']) + weights[1] * self.p1_downsample(
            resnest_out['1']) + weights[2] * self.p2_downsample(resnest_out['2']) + weights[3] * resnest_out['3']

        features = nested_tensor_from_tensor_list(features)
        pos = self.position_encoding(features).to(features.tensors.dtype)

        return [features], [pos]
