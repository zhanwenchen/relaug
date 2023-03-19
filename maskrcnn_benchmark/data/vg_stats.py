from dataclasses import dataclass
from typing import List
from torch import Tensor
from torch.utils.data import Dataset


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class VGStats(metaclass=Singleton):
    fg_matrix: Tensor = None
    pred_dist: Tensor = None
    obj_classes: List[str] = None
    rel_classes: List[str] = None
    att_classes: List[str] = None
    stats: List[list] = None

    def __post_init__(self):
        if self.fg_matrix is None or self.pred_dist is None or self.obj_classes is None or self.rel_classes is None:
            raise ValueError('None')
