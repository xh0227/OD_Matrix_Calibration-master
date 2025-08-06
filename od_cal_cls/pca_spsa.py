# Import necessary modules.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from od_cal_cls.spsa import spsa
from od_cal_fct.user_utill import *

# Class definition for PCA-SPSA implementation.
class pca_spsa (spsa):
    
    def __init__(self, 
                in_fl_step_ini: float = 2, in_fl_perturb_ini: float = 1, 
                in_fl_param_a: float = 25, in_fl_param_alpha: float = 0.3, 
                in_fl_param_gamma: float = 0.15, in_int_seg_size: int = 3, 
                in_int_iter_gradi: int = 3, in_int_iter_opti: int = 15, 
                in_lv_seg_step: bool = True, in_lv_seg_perturb: bool = True,
                in_lv_min_bound:bool = False,
        ) -> None:
        # Initialization of parent class spsa.
        super().__init__(in_fl_step_ini, in_fl_perturb_ini, 
                        in_fl_param_a, in_fl_param_alpha, 
                        in_fl_param_gamma, in_int_seg_size, 
                        in_int_iter_gradi, in_int_iter_opti, 
                        in_lv_seg_step, in_lv_seg_perturb,
                        in_lv_min_bound
                )
        # Further initialization for pca-spsa.
        self.lst_path_od_samples    = list()        
        self.arr_stacked_ods        = np.ndarray((1,1))
        self.obj_od_pca             = object()
        self.int_pc_size            = int(0)
        self.arr_od_reduct          = np.ndarray((1,1))
    
    # Function to read file paths for od samples.
    def read_od_samples (self, in_str_dir_od_samples:str):
        # Update list of paths of od samples.
        self.lst_path_od_samples = fileListCreator(
            strDirPath= in_str_dir_od_samples, lv_flt= True, ext_flt= "csv"
        )        
        # Read all od sample files. Extract array and flatten.
        lst_arr_ods_flat_tmp = []
        for od_sample in self.lst_path_od_samples:
            df_od_tmp1 = pd.read_csv(od_sample, index_col= 0)
            arr_od_tmp1 = df_od_tmp1.values
            arr_od_flat_tmp1 = arr_od_tmp1.flatten()
            lst_arr_ods_flat_tmp.append(arr_od_flat_tmp1)
        # Vertical stacking of all flattened arrays.
        self.arr_stacked_ods = np.stack(lst_arr_ods_flat_tmp, axis= 0)
        print(f"All OD samples are stacked for PCA module. Nr of Samples: {len(self.arr_stacked_ods)}")
        
    # Function to create PCA object fitted with od samples.
    def create_od_pca (self, in_fl_ex_var:float):
        # Instantiate PCA object with demanding explaining variance.
        self.obj_od_pca = PCA(n_components= in_fl_ex_var, svd_solver= "full")
        # Fitting with stacked od sample matrix.
        self.obj_od_pca.fit(self.arr_stacked_ods)
        self.int_pc_size = self.obj_od_pca.n_components_
        print(f"PCA object has been fit with {in_fl_ex_var} explaining ratio.")
        print(f"Reduced dimension: {self.int_pc_size}")
    
    # Function to convert current od to reducted z.
    def pca_transform (self):        
        # Make flat array from od dataframe.
        arr_od_flat_tmp = self.df_od.values.flatten()
        # Reshape it to use it for transformation. Row vector.
        arr_od_flat_tmp = arr_od_flat_tmp.reshape((1,-1))
        # Apply PCA transformation onto current od array.
        arr_od_reduct_tmp = self.obj_od_pca.transform(arr_od_flat_tmp)
        # Reshape it again and update dim reducted od array.
        self.arr_od_reduct = arr_od_reduct_tmp[0]        
        print(f"OD transformation successfully done. Transformation result saved.")
        
    # Function to reshape reducted od into normal od. z >> OD.
    def pca_inverse_transform (self, in_arr_od_reduct:np.ndarray) -> pd.DataFrame:
        # Reshape reducted array and inverse transformation.
        arr_od_reduct = in_arr_od_reduct.reshape((1,-1))
        arr_od_ivs = self.obj_od_pca.inverse_transform(arr_od_reduct)
        # Reshape output of inverse transformation.
        arr_od_ivs = arr_od_ivs[0]
        arr_od_ivs = arr_od_ivs.reshape(self.tup_dim_od)
        # Make diagonal elements zero.
        np.fill_diagonal(arr_od_ivs, 0)
        # Make Dataframe.
        idx_tmp = self.df_od.index
        cols_tmp = self.df_od.columns
        df_od_ivs = pd.DataFrame(arr_od_ivs, index= idx_tmp, columns= cols_tmp)
        df_od_ivs = df_od_ivs.clip(lower=0)
        return df_od_ivs
    
    # Method Override for randome vector cration.
    def bernoulli_delta (self, in_int_iter_cur_opti: int, in_int_iter_cur_gradi: int):
        # Bernoulli distribution with mean value 0.
        self.arr_bernoulli = 2*np.random.binomial(n=1, p=0.5, size= self.int_pc_size) - 1        
        print(f"        Bernoulli delta array for iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi} is ready.")
        
    # Method Override for Perturbation plus.
    def perturbation_plus (self, in_int_iter_cur_opti: int, in_int_iter_cur_gradi: int) -> np.ndarray:
        # z=z+z.*ck.*delta
        arr_od_reduct_plus = self.arr_od_reduct + (self.arr_od_reduct * self.fl_perturb * self.arr_bernoulli)
        print(f"        Perturbation vector in plus direction is synthesized for iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        return arr_od_reduct_plus
    
    # Method Override for Perturbation minus.
    def perturbation_minus (self, in_int_iter_cur_opti: int, in_int_iter_cur_gradi: int) -> np.ndarray:
        # z=z-z.*ck.*delta
        arr_od_reduct_minus = self.arr_od_reduct - (self.arr_od_reduct * self.fl_perturb * self.arr_bernoulli)
        print(f"        Perturbation vector in minus direction is synthesized for iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        return arr_od_reduct_minus
    
    # Method Override for Minimization_loss.
    def minimization_loss(self, in_int_iter_cur_opti: int):
        # Pick relevent gradient estimation. Reducted dimension.
        arr_red_gradi_est = self.lst_gradi[in_int_iter_cur_opti - 1]
        # Set fl_step_ini in case of first optimization loop.
        if in_int_iter_cur_opti == 1:
            fl_red_gradi = np.abs(arr_red_gradi_est).mean()
            self.fl_step = (0.7 * self.fl_perturb) / fl_red_gradi
            self.fl_step_ini = self.fl_step * ((1 + self.fl_param_a)**self.fl_param_alpha)
            self.lst_step.pop()
            self.lst_step.append(self.fl_step)
            print("    Minimization step initialized.")
        # Minimization. z=z-z.*ak.*ghat
        self.arr_od_reduct = self.arr_od_reduct - (self.arr_od_reduct * self.fl_step * arr_red_gradi_est)
        print(f"    Minimization is sucessfully done. Iteration_{in_int_iter_cur_opti}.")