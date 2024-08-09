from .cifar100 import Cifar100
from .cifar100_osr import Cifar100_osr
from .tinyimagenet_osr import Tinyimagenet_osr
from .tinyimagenet import Tinyimagenet
from .flower import Flower
from .flower_osr import Flower_osr
from .voc_2007_merge1 import Voc2007_merge1
from .voc_2007_merge1_osr import Voc2007_merge1_osr


get_dataset_from_name = {
    "cifar100"                  : Cifar100,
    "cifar100_osr"              : Cifar100_osr,
    "tinyimagenet"              : Tinyimagenet,
    "tinyimagenet_osr"          : Tinyimagenet_osr,
    "flower"                    : Flower,
    "flower_osr"                : Flower_osr,
    "voc2007_merge1"            : Voc2007_merge1,
    "voc2007_merge1_osr"        : Voc2007_merge1_osr
}
