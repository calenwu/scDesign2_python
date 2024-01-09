import numpy as np
import pandas as pd

from typing import Union, List


class SimCellConfig:
    def __init__(
        self,
        random_seed: int = 0,
        n_genes: int = 5000,
        n_cells: int = 200,
        group_names: Union[np.ndarray, List[str]] = ["group1"],
        p_de_list: Union[float, np.ndarray] = 0.1,
        p_down_list: Union[float, np.ndarray] = 0.5,
        de_location_list: Union[float, np.ndarray] = 0.1,
        de_scale_list: Union[float, np.ndarray] = 0.4,
        batch_effect: bool = True,
        batch_names: Union[np.ndarray, List[str]] = ["patient1"],
        pb_de_list: Union[float, np.ndarray] = 0.1,
        pb_down_list: Union[float, np.ndarray] = 0.5,
        bde_location_list: Union[float, np.ndarray] = 0.1,
        bde_scale_list: Union[float, np.ndarray] = 0.4,
        shared_cnv: bool = False,
        mean_shape: float = 0.6,
        mean_scale: float = 3,
        p_outlier: float = 0.05,
        outlier_loc: int = 4,
        outlier_scale: float = 0.5,
        libsize_loc: int = 11,
        libsize_scale: float = 0.2,
        common_disp: float = 0.1,
        dof: int = 60,
        dropout_midpoint: float = 0,
        dropout_shape: float = -1,
    ):
        self.random_seed = random_seed
        self.n_genes = n_genes
        self.n_cells = n_cells
        self.group_names = group_names
        self.p_de_list = p_de_list
        self.p_down_list = p_down_list
        self.de_location_list = de_location_list
        self.de_scale_list = de_scale_list
        self.batch_names = batch_names
        self.batch_effect = batch_effect
        self.pb_de_list = pb_de_list
        self.pb_down_list = pb_down_list
        self.bde_location_list = bde_location_list
        self.bde_scale_list = bde_scale_list
        self.shared_cnv = shared_cnv
        self.mean_shape = mean_scale
        self.mean_scale = mean_shape
        self.p_outlier = p_outlier
        self.outlier_loc = outlier_loc
        self.outlier_scale = outlier_scale
        self.libsize_loc = libsize_loc
        self.libsize_scale = libsize_scale
        self.common_disp = common_disp
        self.dof = dof
        self.dropout_midpoint = dropout_midpoint
        self.dropout_shape = dropout_shape

    def to_dict(self):
        dict = vars(self).copy()
        for k, v in dict.items():
            if type(v) == np.ndarray:
                dict[k] = dict[k].tolist()
        return dict

    def create_rng(self):
        return np.random.default_rng(self.random_seed)
