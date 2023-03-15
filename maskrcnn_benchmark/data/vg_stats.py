from os import environ as os_environ
from os.path import join as os_path_join, exists as os_path_exists
from dataclasses import dataclass
from typing import List
from torch import Tensor, load as torch_load, save as torch_save


STATISTICS_FNAME = 'statistics.pth'
STATISTICS_FPATH = os_path_join(os_environ['MODEL_DIRPATH'], STATISTICS_FNAME)


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

    def __post_init__(self):
        if self.fg_matrix is None or self.pred_dist is None or self.obj_classes is None or self.rel_classes is None:
            raise ValueError('None')
        if not os_path_exists(STATISTICS_FPATH):
            statistics = {
                'fg_matrix': self.fg_matrix,
                'pred_dist': self.pred_dist,
                'obj_classes': self.obj_classes,
                'rel_classes': self.rel_classes,
                'att_classes': self.att_classes,
            }
            print(f'VGStats: Started saving statistics to ${STATISTICS_FPATH} for the first time')
            torch_save(statistics, STATISTICS_FPATH)
            print(f'VGStats: Finished saving statistics to ${STATISTICS_FPATH} for the first time')



    @classmethod
    def from_file(cls):
        statistics = torch_load(STATISTICS_FPATH)
        fg_matrix = statistics['fg_matrix']
        pred_dist = statistics['pred_dist']
        obj_classes = statistics['obj_classes']
        rel_classes = statistics['rel_classes']
        att_classes = statistics['att_classes']
        return cls(fg_matrix, pred_dist, obj_classes, rel_classes, att_classes)
