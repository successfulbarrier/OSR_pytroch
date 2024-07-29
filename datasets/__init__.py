from .cifar100 import Cifar100
from .cifar100_osr import Cifar100_osr

get_dataset_from_name = {
    "cifar100"                  : Cifar100,
    "cifar100_osr"              : Cifar100_osr,
}
