# Import necessary modules.
from sklearn.metrics import mean_squared_error
from od_cal_fct.graph_gen import *
from od_cal_fct.user_utill import *
from od_cal_fct.flow_gen import *
from od_cal_cls.gnn_cls import gnn_GATv2_CONV_LIN
from torch_geometric.data import Data
from math import sqrt
import pandas as pd
import numpy as np
import torch
import os

# Class definition for SPSA implementation.
class spsa (object):
    
    def __init__(self,
            in_fl_step_ini:float = 2.0, in_fl_perturb_ini:float = 1.0,
            in_fl_param_a:float = 25.0, in_fl_param_alpha:float = 0.3, 
            in_fl_param_gamma:float = 0.15, in_int_seg_size:int = 3,
            in_int_iter_gradi:int = 3, in_int_iter_opti:int = 15,
            in_lv_seg_step:bool = True, in_lv_seg_perturb:bool = True,
            in_lv_min_bound:bool = False,            
        ) -> None:
        
        # Store inputs in instance.
        self.fl_step_ini    = in_fl_step_ini
        self.fl_perturb_ini = in_fl_perturb_ini
        self.fl_param_a     = in_fl_param_a
        self.fl_param_alpha = in_fl_param_alpha
        self.fl_param_gamma = in_fl_param_gamma
        self.int_seg_size   = in_int_seg_size
        self.int_iter_gradi = in_int_iter_gradi
        self.int_iter_opti  = in_int_iter_opti
        self.lv_seg_step    = in_lv_seg_step
        self.lv_seg_perturb = in_lv_seg_perturb
        self.lv_min_bound   = in_lv_min_bound
        
        # Attrs for OD matrix.
        self.df_od          = pd.DataFrame(np.zeros((2,2)))     # Updated along iter_opti.
        self.df_od_base     = pd.DataFrame(np.zeros((2,2)))     # Will not be updated.
        self.df_od_true     = pd.DataFrame(np.zeros((2,2)))
        self.df_od_best     = pd.DataFrame(np.zeros((2,2)))
        self.df_min_mem     = pd.DataFrame(np.zeros((2,2)))
        self.tup_dim_od     = tuple()
        self.int_dim_od     = int(0)
        
        # Attrs for link flows.
        self.df_true_flow   = pd.DataFrame(np.zeros((2,2)))     # True link flow. Won't be changed. 
        self.tup_dim_flow   = tuple()
        
        # Parameters.
        self.fl_step        = float(0)
        self.fl_perturb     = float(0)
        self.df_step_seg    = pd.DataFrame(np.zeros((2,2)))
        self.df_perturb_seg = pd.DataFrame(np.zeros((2,2)))
        self.df_flow_best   = pd.DataFrame(np.zeros((2,2)))
        self.arr_bernoulli  = np.ndarray((1,1))
        self.lst_step       = list()
        self.lst_perturb    = list()
        self.lst_gradi      = list()
        self.lst_nRmse      = list()
        self.lst_time_epoch = list()
        self.nRmse_best     = float(0)
        
        # Attrs for GNN.
        self.gnn_model      = object()
        self.ts_edge_idx    = torch.tensor(0)
        
    def read_od_base (self, in_str_path_od_csv:str):        
        # Read OD dataframe from the file path.
        self.df_od = pd.read_csv(in_str_path_od_csv, index_col= 0)
        self.df_min_mem = self.df_od.copy()
        self.df_od_base = self.df_od.copy()
        print("OD matrix has been imported.")
        print(f"dType: {str(self.df_od.values.dtype)}")
        # Store OD matrix dimension info.
        self.tup_dim_od = self.df_od.shape
        self.int_dim_od = self.df_od.shape[0]
        print(f"Dimension of OD matrix is {self.int_dim_od:05d} by {self.df_od.shape[1]:05d}.")
        print(self.df_od.head(15))
        
    def read_od_true (self, in_str_path_od_csv:str):        
        # Read OD dataframe from the file path.
        self.df_od_true = pd.read_csv(in_str_path_od_csv, index_col= 0)
        print("True OD matrix has been imported.")
        print(f"dType: {str(self.df_od_true.values.dtype)}")        
        print(self.df_od_true.head(15))
        
    def read_true_flow (self, in_str_path_true_flow_csv:str):
        # Read true flow dataframe from the file path.
        self.df_true_flow = pd.read_csv(in_str_path_true_flow_csv, index_col= 0)
        print("True flow has been imported.")
        print(f"dType: {str(self.df_true_flow.values.dtype)}")
        print(self.df_true_flow.head(15))
        self.tup_dim_flow = self.df_true_flow.shape
        print(f"Dimension of true flow is {self.tup_dim_flow[0]:05d} by {self.tup_dim_flow[1]:05d}.")
        
    def param_update (self, in_int_iter_cur_opti:int):
        # Filter out iteration counter.
        int_iter = in_int_iter_cur_opti
        int_iter = min(self.int_iter_opti, max(int_iter, 1))
        # Initial parameters with iter number.
        self.fl_step  = self.fl_step_ini / ((int_iter + self.fl_param_a)**self.fl_param_alpha)
        self.fl_perturb = self.fl_perturb_ini / (int_iter**self.fl_param_gamma)
        # Append list of parameters.
        self.lst_step.append(self.fl_step)
        self.lst_perturb.append(self.fl_perturb)
        # Segment level dataframe creation.
        df_od_seg = ((self.df_od//self.int_seg_size) + 1).copy()
        # OD matrix mean value.
        arr_df_od_tmp = self.df_od.values
        np.fill_diagonal(arr_df_od_tmp, 0)
        fl_od_mean = arr_df_od_tmp.sum() / (arr_df_od_tmp.size - arr_df_od_tmp.shape[0])
        # Semented step parameters.
        arr_step_seg_tmp = np.full(self.tup_dim_od, self.fl_step)
        np.fill_diagonal(arr_step_seg_tmp, 0)   # Make diagonal 0.                
        self.df_step_seg = pd.DataFrame(
            arr_step_seg_tmp,
            index= self.df_od.index, columns= self.df_od.columns
        )
        # Segmenting will be done only when user requests.
        if self.lv_seg_step:            
            self.df_step_seg = (self.df_step_seg*self.int_seg_size) / fl_od_mean
            self.df_step_seg = self.df_step_seg * df_od_seg
        # Notice to user.
        print(f"    Step size a_k for iteration_{int_iter} is ready.")
        
        # Segmented step perturb coe.
        arr_perturb_seg_tmp = np.full(self.tup_dim_od, self.fl_perturb)
        np.fill_diagonal(arr_perturb_seg_tmp, 0)    # Make diagonal 0.        
        self.df_perturb_seg = pd.DataFrame(
            arr_perturb_seg_tmp,
            index= self.df_od.index, columns= self.df_od.columns
        )
        # Segmenting will be done only when user requests.
        if self.lv_seg_perturb:            
            self.df_perturb_seg = (self.df_perturb_seg*self.int_seg_size) / fl_od_mean
            self.df_perturb_seg = self.df_perturb_seg * df_od_seg        
        # Notice to user.
        print(f"    Perturbation coefficient c_k for iteration_{int_iter} is ready.")
        
    def bernoulli_delta (self, in_int_iter_cur_opti:int, in_int_iter_cur_gradi:int):
        # Bernoulli distribution with mean value 0.
        self.arr_bernoulli = 2*np.random.binomial(n=1, p=0.5, size= self.tup_dim_od) - 1
        np.fill_diagonal(self.arr_bernoulli, 0)
        print(f"        Bernoulli delta array for iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi} is ready.")
    
    def perturbation_plus (self, in_int_iter_cur_opti:int, in_int_iter_cur_gradi:int) -> pd.DataFrame:
        # Perturbing OD matrix in positive direction.
        df_perturb_plus = self.df_od + (self.df_perturb_seg * self.arr_bernoulli)
        df_perturb_plus = df_perturb_plus.clip(lower=0)#.round(0).astype(int)
        print(f"        Perturbation vector in plus direction is synthesized for iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        return df_perturb_plus
    
    def perturbation_minus (self, in_int_iter_cur_opti:int, in_int_iter_cur_gradi:int) -> pd.DataFrame:
        # Perturbing OD matrix in minus direction.
        df_perturb_minus = self.df_od - (self.df_perturb_seg * self.arr_bernoulli)
        df_perturb_minus = df_perturb_minus.clip(lower=0)#.round(0).astype(int)
        print(f"        Perturbation vector in minus direction is synthesized for iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        return df_perturb_minus
    
    def minimization_loss (self, in_int_iter_cur_opti:int):
        
        # Pick relevent gradient estimation.
        arr_gradi_est = self.lst_gradi[in_int_iter_cur_opti - 1]
        df_gradi_est = pd.DataFrame(arr_gradi_est, index= self.df_od.index, columns= self.df_od.columns)
        
        # Lilmitations on update from previous OD matrix.
        # Set Min/Max of OD update. 35[%]
        df_od_max = self.df_min_mem * 1.35
        df_od_min = self.df_min_mem * 0.65   
        # Internal function for each element of OD matrix.
        self.df_min_mem = self.df_min_mem - (self.df_step_seg * df_gradi_est)
        # Masked Min/Max of OD update.
        df_od_max_msk = df_od_max[df_od_max < self.df_min_mem].copy()
        df_od_min_msk = df_od_min[df_od_min > self.df_min_mem].copy()
        # Clipping with masked Min/Max OD matrices.
        # In memorizing dataframe, float numbers will be stored.
        self.df_min_mem.update(df_od_max_msk)
        self.df_min_mem.update(df_od_min_msk)
        
        # Boundary of update from base OD matrix.
        if self.lv_min_bound:            
            # Set Upper/Loser bound of OD update from base OD matrix.
            df_od_bound_u = self.df_od_base * 1.7
            df_od_bound_l = self.df_od_base * 0.8
            # Masked boundary OD matrices.
            df_od_bound_u_msk = df_od_bound_u[df_od_bound_u < self.df_min_mem].copy()
            df_od_bound_l_msk = df_od_bound_l[df_od_bound_l > self.df_min_mem].copy()
            # Flipping with masked boundary OD matrices.
            self.df_min_mem.update(df_od_bound_u_msk)
            self.df_min_mem.update(df_od_bound_l_msk)        
        
        # Update attribute OD matrix. User may restric output value as integer.
        self.df_od = self.df_min_mem.clip(lower=0)#.round(0).astype(int)
        print(f"    Minimization is sucessfully done. Iteration_{in_int_iter_cur_opti}.")
    
    @staticmethod
    def sim_sumo_get_flow (
        in_int_iter_cur_opti:int, in_int_iter_cur_gradi:int, in_df_perturbed_od:pd.DataFrame,
        in_str_id_tazRel:str = "car", in_str_sim_st:str = "0", in_str_sim_end:str = "2:0:0",
        in_fl_sim_step:float = 0.2, in_str_path_net:str = "./data_sumo/toy_grid.net.xml",
        in_str_path_taz:str = "./data_sumo/taz_edges.taz.xml", in_str_path_vType:str = "./data_sumo/vehicleType.vType.xml",
        in_str_dir_sumo_tmp = "./data_sumo/tmp", in_str_dir_edgeInfo:str = "./data_sumo/tmp_edgeInfo"
    ) -> pd.DataFrame:
        
        # Initialize temporary folders.
        delAllInDir(in_str_dir_sumo_tmp)
        delAllInDir(in_str_dir_edgeInfo)
        
        # Adjust input argument. 
        if in_int_iter_cur_gradi == 999:
            in_int_iter_cur_gradi = "Minimization"
        
        # Create tazRel file in temporary directory.
        str_path_tazRel = createTazRel(
            in_str_dir_sumo_tmp, in_str_id_tazRel, in_str_sim_st,
            in_str_sim_end, in_df_perturbed_od
        )
        # Create odtrips.xml file.
        lv_suc_odtrips, str_path_odtrips = createOdTrips(
            in_str_dir_sumo_tmp, in_str_path_taz, str_path_tazRel
        )
        # Check if odtrips file has been successfully created.
        if lv_suc_odtrips != 0:
            print(f"        Error on odtrips generation! Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
            return None
        
        # Create rou.xml (Route) file.
        lv_suc_rou, str_path_rou = createRoute(
            in_str_dir_sumo_tmp, in_str_path_net, str_path_odtrips, in_str_path_vType
        )
        
        # Check if route file has been successfully created.
        if lv_suc_rou != 0:
            print(f"        Error on route generation! Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
            return None
        
        # Define link flow output file path.
        str_edgeInfo_id = "edgeInfo_spsa"
        str_edgeInfo_fName = "edgeInfo_spsa.xml"
        str_path_edgeInfo = os.path.join(os.path.abspath(in_str_dir_edgeInfo), str_edgeInfo_fName)
        
        # Define edgeInfo.additional.xml file.
        str_path_edgeInfoAdd = createAddEdgeInfo(in_str_dir_sumo_tmp, str_edgeInfo_id, str_path_edgeInfo)
        print(f"        All temporary files are ready for simulation. Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        
        # Run simulation.
        str_command_sim_tmp = "sumo -n {} -r {} -b 0 --step-length {} -v False --no-step-log True --no-warnings True --additional-files {}".format(
            in_str_path_net, str_path_rou, in_fl_sim_step, str_path_edgeInfoAdd
        )
        lv_suc_sim = os.system(str_command_sim_tmp)
        
        if lv_suc_sim != 0:
            print(f"        Error on SUMO simulation! Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
            return None
        
        print(f"        SUMO Simulation done! Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        
        # Read output edgeInfo file and convert it into dataframe format.
        df_out_sim = xml2df_edge_flow(str_path_edgeInfo)
        tup_dim_df_out_sim = df_out_sim.shape
        print(f"        Simulation output is available! Dimension: {tup_dim_df_out_sim[0]} by {tup_dim_df_out_sim[1]}.")
        print(f"        Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
                
        # Initialize temporary folders.
        delAllInDir(in_str_dir_sumo_tmp)
        delAllInDir(in_str_dir_edgeInfo)
        
        # Return output dataframe.
        return df_out_sim
        
    def load_gnn_model (self, in_str_path_gnn:str, in_str_path_adj_idx:str):
        # Load PyG GNN model on CPU.
        self.gnn_model = gnn_GATv2_CONV_LIN(
            in_dim_x= self.int_dim_od, in_dim_y= self.tup_dim_flow[1],
            in_dim_hid= self.int_dim_od, in_num_layers=3,
            in_lc_norm= False, in_lc_dropout= False
        ).to(torch.device('cpu'))
        # Load weighting factors.
        self.gnn_model.load_state_dict(torch.load(in_str_path_gnn))
        # Set to evaluation mode.
        self.gnn_model.eval()
        print("GNN Model has been loaded. Set to evaluation mode.")
        # Load adjcency matrix for further use.
        arr_edge_idx = np.load(in_str_path_adj_idx)
        self.ts_edge_idx = torch.tensor(arr_edge_idx, dtype= torch.long)
        print("Graph edge index has been loaded.")
    
    def sim_gnn_get_flow (self, in_int_iter_cur_opti:int, in_int_iter_cur_gradi:int, in_df_od:pd.DataFrame)-> pd.DataFrame:
        # Adjust input argument. 
        if in_int_iter_cur_gradi == 999:
            in_int_iter_cur_gradi = "Minimization"
        # Build the graph with input od dataframe.
        arr_x = in_df_od.values
        ts_x = torch.tensor(arr_x, dtype=torch.float)
        graph = Data(x= ts_x, edge_index= self.ts_edge_idx.t().contiguous())
        # Get prediction.
        with torch.no_grad():
            pred_y = self.gnn_model(graph)
            pred_y = pred_y.numpy()
        # Convert to dataframe and return it.
        idx_tmp = self.df_true_flow.index
        cols_tmp = self.df_true_flow.columns
        df_flow_pred = pd.DataFrame(pred_y, index= idx_tmp, columns= cols_tmp)
        # tup_dim_df_out_sim = df_flow_pred.shape
        # print(f"        Simulation output is available! Dimension: {tup_dim_df_out_sim[0]} by {tup_dim_df_out_sim[1]}.")
        # print(f"        Iteration_{in_int_iter_cur_opti}_{in_int_iter_cur_gradi}.")
        return df_flow_pred
    
    @staticmethod
    def n_Rmse (in_df_true:pd.DataFrame, in_df_pred:pd.DataFrame) -> float:
        # Convert dataframes to ndarray.
        arr_df_true = in_df_true.values
        arr_df_pred = in_df_pred.values
        # Calculate MSE/RMSE/Normalized RMSE values.
        mse_df_flow = mean_squared_error(y_true= arr_df_true, y_pred= arr_df_pred)
        rmse_df_flow = sqrt(mse_df_flow)
        n_rmse_df_flow = float(rmse_df_flow/arr_df_true.mean())
        # Return n_Rmse.
        return n_rmse_df_flow
    
    @staticmethod
    def rmse (in_df_true:pd.DataFrame, in_df_pred:pd.DataFrame) -> float:
        # Convert dataframes to ndarray.
        arr_df_true = in_df_true.values
        arr_df_pred = in_df_pred.values
        # Calculate MSE/RMSE/Normalized RMSE values.
        mse_df_flow = mean_squared_error(y_true= arr_df_true, y_pred= arr_df_pred)
        rmse_df_flow = sqrt(mse_df_flow)        
        # Return rmse.
        return rmse_df_flow

    @staticmethod
    def n_Rmse_iq (in_df_true:pd.DataFrame, in_df_pred:pd.DataFrame) -> float:
        # Convert dataframes to ndarray.
        arr_df_true = in_df_true.values
        arr_df_pred = in_df_pred.values
        # Calculate MSE/RMSE/Normalized RMSE values.
        # In this function, interquartile range is used for normalization.
        mse_df_flow = mean_squared_error(y_true= arr_df_true, y_pred= arr_df_pred)
        rmse_df_flow = sqrt(mse_df_flow)
        q1_arr_df_true = np.quantile(arr_df_true, 0.25)
        q3_arr_df_true = np.quantile(arr_df_true, 0.75)
        q_range = q3_arr_df_true - q1_arr_df_true
        n_rmse_df_flow_iq = float(rmse_df_flow/q_range)
        # Return n_Rmse.
        return n_rmse_df_flow_iq
