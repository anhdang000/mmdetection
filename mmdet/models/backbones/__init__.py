from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .trident_resnet import TridentResNet
from .resnet_parallel_1 import ResNetParallel1
from .resnet_parallel_2 import ResNetParallel2
from .resnet_parallel_3 import ResNetParallel3
from .resnet_parallel_5 import ResNetParallel5
from .resnet_parallel_6 import ResNetParallel6

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 
    'ResNetParallel1', 'ResNetParallel2', 'ResNetParallel3', 'ResNetParallel5',
    'ResNetParallel6'
]
