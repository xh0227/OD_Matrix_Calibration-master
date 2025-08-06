# Improt necessary modules.
import pandas as pd
import numpy as np
from itertools import product
from od_cal_cls.spsa import spsa
from od_cal_fct.user_utill import *

# Class definition for FDSA implementation.
class fdsa (spsa):
    
    # Initialization.
    def __init__(self, 
                in_fl_step_ini: float = 2, in_fl_perturb_ini: float = 1, 
                in_fl_param_a: float = 25, in_fl_param_alpha: float = 0.3, 
                in_fl_param_gamma: float = 0.15, in_int_seg_size: int = 3, 
                in_int_iter_gradi: int = 3, in_int_iter_opti: int = 15, 
                in_lv_seg_step: bool = True, in_lv_seg_perturb: bool = True,
                in_lv_min_bound:bool = False,
        ) -> None:
        super().__init__(in_fl_step_ini, in_fl_perturb_ini, 
                        in_fl_param_a, in_fl_param_alpha, 
                        in_fl_param_gamma, in_int_seg_size, 
                        in_int_iter_gradi, in_int_iter_opti, 
                        in_lv_seg_step, in_lv_seg_perturb,
                        in_lv_min_bound
                )
        
    # Generator for perturbed OD matrix.
    def gen_pertub_dfs (self) -> list:
        
        # List of dimensioin of OD matrix.
        lst_dim_od = range(self.int_dim_od)
        
        # Product loop for each element in OD matrix.
        for tup_coord in product(lst_dim_od, lst_dim_od):
            # Coordinations for OD matrix.
            idx_tmp1 = tup_coord[0]
            col_tmp1 = tup_coord[1]
            # No perturbation on diagonal elements.
            if idx_tmp1 == col_tmp1:
                continue
            else:
                # Copy current OD matrix.
                df_pertub_p_tmp1 = self.df_od.copy()
                df_pertub_m_tmp1 = self.df_od.copy()
                # Perturbing can be done in either segmented or not segmented.
                if self.lv_seg_perturb:
                    # Pertubing in plus direction for just one element.
                    df_pertub_p_tmp1.iat[idx_tmp1, col_tmp1] += self.df_perturb_seg.iat[idx_tmp1, col_tmp1]
                    # Pertubing in minus direction for just one element.
                    df_pertub_m_tmp1.iat[idx_tmp1, col_tmp1] -= self.df_perturb_seg.iat[idx_tmp1, col_tmp1]
                else:
                    # Pertubing in plus direction for just one element.
                    df_pertub_p_tmp1.iat[idx_tmp1, col_tmp1] += self.fl_perturb
                    # Pertubing in minus direction for just one element.
                    df_pertub_m_tmp1.iat[idx_tmp1, col_tmp1] -= self.fl_perturb
                # Yield two dataframes as a list.
                # REMEMBER: First element is plus perturbed one.
                yield [df_pertub_p_tmp1, df_pertub_m_tmp1, idx_tmp1, col_tmp1]