from .c2d_sifa_resnet import *

from .swin_vit import *
from .c2d_swin_vit import *
from .c2d_sifa_swin import *

from .model_factory import get_model_by_name, transfer_weights, remove_fc, remove_defcor_weight