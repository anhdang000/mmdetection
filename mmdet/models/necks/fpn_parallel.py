import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPNParallel(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPNParallel, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.num_merge_stages = 3
        self.convs_after_merge_1 = [nn.ModuleList() for _ in range(len(self.num_merge_stages))]
        self.convs_after_merge_2 = [nn.ModuleList() for _ in range(len(self.num_merge_stages))]
        for s in range(len(self.num_merge_stages)):
            for _ in range(len(self.lateral_convs)):
                conv_1 = nn.Conv2d(2*self.out_channels, self.out_channels, kernel_size=3, padding=(1, 1)).cuda()
                nn.init.kaiming_uniform_(conv_1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                self.convs_after_merge_1[s].append(conv_1)

                conv_2 = nn.Conv2d(2*self.out_channels, self.out_channels, kernel_size=3, padding=(1, 1)).cuda()
                nn.init.kaiming_uniform_(conv_2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                self.convs_after_merge_2[s].append(conv_2)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs1, inputs2):
        """Forward function."""
        assert len(inputs1) == len(self.in_channels)
        assert len(inputs2) == len(self.in_channels)

        # build laterals
        laterals_1 = [
            lateral_conv(inputs1[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals_2 = [
            lateral_conv(inputs2[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Merge [stage 1]
        laterals_merge = [torch.cat((l1, l2), dim=1) for l1, l2 in zip(laterals_1, laterals_2)]
        laterals_1 = [
            self.convs_after_merge_1[0][i](laterals_merge[i]) 
            for i in range(len(laterals_1))
        ]
        laterals_2 = [
            self.convs_after_merge_2[0][i](laterals_merge[i]) 
            for i in range(len(laterals_2))
        ]

        # build top-down path
        used_backbone_levels = len(laterals_1)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals_1[i - 1] += F.interpolate(laterals_1[i],
                                                 **self.upsample_cfg)
                laterals_2[i - 1] += F.interpolate(laterals_2[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals_1[i - 1].shape[2:]
                laterals_1[i - 1] += F.interpolate(
                    laterals_1[i], size=prev_shape, **self.upsample_cfg)
                laterals_2[i - 1] += F.interpolate(
                    laterals_2[i], size=prev_shape, **self.upsample_cfg)

        # Merge [stage 2]
        laterals_merge = [torch.cat((l1, l2), dim=1) for l1, l2 in zip(laterals_1, laterals_2)]
        laterals_1 = [
            self.convs_after_merge_1[1][i](laterals_merge[i]) 
            for i in range(len(laterals_1))
        ]
        laterals_2 = [
            self.convs_after_merge_2[1][i](laterals_merge[i]) 
            for i in range(len(laterals_2))
        ]

        # build outputs
        # part 1: from original levels
        outs_1 = [
            self.fpn_convs[i](laterals_1[i]) for i in range(used_backbone_levels)
        ]
        outs_2 = [
            self.fpn_convs[i](laterals_2[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs_1):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs_1.append(F.max_pool2d(outs_1[-1], 1, stride=2))
                    outs_2.append(F.max_pool2d(outs_2[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs1[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals_1[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs_1[-1]
                else:
                    raise NotImplementedError
                outs_1.append(self.fpn_convs[used_backbone_levels](extra_source))
                outs_2.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs_1.append(self.fpn_convs[i](F.relu(outs_1[-1])))
                        outs_2.append(self.fpn_convs[i](F.relu(outs_2[-1])))
                    else:
                        outs_1.append(self.fpn_convs[i](outs_1[-1]))
                        outs_2.append(self.fpn_convs[i](outs_2[-1]))
        
        # Merge [stage 3]
        outs_merge = [torch.cat((l1, l2), dim=1) for l1, l2 in zip(outs_1, outs_2)]
        outs_1 = [
            self.convs_after_merge_1[2][i](outs_merge[i]) 
            for i in range(len(outs_1))
        ]
        outs_2 = [
            self.convs_after_merge_2[2][i](outs_merge[i]) 
            for i in range(len(outs_2))
        ]
        outs = [out_1 + out_2 for out_1, out_2 in zip(outs_1, outs_2)]
        return tuple(outs)
