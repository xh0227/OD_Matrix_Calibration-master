# Import necessary modules.
import time
import numpy as np
import pandas as pd
from od_cal_cls.spsa import spsa
from od_cal_fct.user_utill import savePklFile
import plotly.graph_objects as go

# Create SPSA object.
spsa_sumo = spsa(
    in_fl_step_ini= 100, in_fl_param_a= 9, in_fl_param_alpha= 0.12,
    in_fl_perturb_ini= 2, in_fl_param_gamma= 0.08, 
    in_int_iter_gradi= 1, in_int_iter_opti= 300, 
    in_lv_seg_step= False, in_lv_seg_perturb= False, in_lv_min_bound= True,
)

now = time.localtime()
key = "{:02d}{:02d}_{:02d}_{:02d}".format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
cal = "{}_{}_{}__{}_{}".format(
    spsa_sumo.fl_step_ini, spsa_sumo.fl_param_a, spsa_sumo.fl_param_alpha,
    spsa_sumo.fl_perturb_ini, spsa_sumo.fl_param_gamma
)

# Set up plotting templete.
tempelete_01_white = dict(
    layout = go.Layout(
        # Layout properties
        title_font_size= 14,
        title_x= 0.1,
        font_size= 11,
        font_color= "#000000",
        font_family= "Times New Roman",
        margin_b = 65,
        margin_l = 60,
        margin_r = 30,
        margin_t = 50,
        plot_bgcolor= "#ffffff",
        # X axis properties
        xaxis_color= "#000000",
        xaxis_linecolor= "#000000",
        xaxis_ticks= "inside",        
        xaxis_tickfont_color= "#000000",
        xaxis_tickfont_family= "Times New Roman",
        xaxis_mirror= True,
        xaxis_showline= True,
        xaxis_showgrid= False,
        # Y axis properties
        yaxis_color= "#000000",
        yaxis_linecolor= "#000000",
        yaxis_ticks= "inside",
        yaxis_tickfont_color= "#000000",
        yaxis_tickfont_family= "Times New Roman",
        yaxis_mirror= True,
        yaxis_showline= True,
        yaxis_showgrid= False,
    )
)

# Visualize & Check step value.
def eval_step(x, step_ini, param_A, param_alpha):
    step = step_ini / ((x + param_A)**param_alpha)
    return step

iter_opti = list(range(1,201))
step = list(map(lambda x: eval_step(
                                    x, step_ini= spsa_sumo.fl_step_ini, 
                                    param_A= spsa_sumo.fl_param_a, 
                                    param_alpha= spsa_sumo.fl_param_alpha
                                    ), 
                iter_opti
                )
            )

fig_step = go.Figure()

fig_step.add_trace(
    go.Scatter(
        x= iter_opti,
        y= step,
        line_color = "#000000",
    )
)

fig_step.update_layout(
    title= "Step Changes",
    xaxis_title= "Number Iteration",
    yaxis_title= "Step Value [NrVeh/hr]",
    width= 500,
    height= 350,
    template= tempelete_01_white,
)

fig_step.update_xaxes(
    range= [0, 200]
)

fig_step.write_html(f"./figures/{key}__{cal}_step_eval.html")

# Visualize & Check perturbation value.
def eval_perturb(x, perturb_ini, param_gamma):
    perturb = perturb_ini / (x**param_gamma)
    return perturb

iter_opti = list(range(1,201))
perturb = list(map(lambda x: eval_perturb(
                                            x, perturb_ini= spsa_sumo.fl_perturb_ini, 
                                            param_gamma=spsa_sumo.fl_param_gamma
                                        ),
                    iter_opti
                    )
                )

fig_perturb = go.Figure()

fig_perturb.add_trace(
    go.Scatter(
        x= iter_opti,
        y= perturb,
        line_color = "#000000",
    )
)

fig_perturb.update_layout(
    title= "Perturbing Changes",
    xaxis_title= "Number Iteration",
    yaxis_title= "Perturbing Value [NrVeh/hr]",
    width= 500,
    height= 350,
    template= tempelete_01_white,
)

fig_perturb.update_xaxes(
    range= [0, 200]
)

fig_perturb.write_html(f"./figures/{key}__{cal}_perturb_eval.html")

# Get iterating numbers.
int_nr_iter_opti = spsa_sumo.int_iter_opti
int_nr_iter_gradi = spsa_sumo.int_iter_gradi

# Import true od matrix.
str_path_od_true = "./data_tabular/true_od_v02.csv"
spsa_sumo.read_od_true(str_path_od_true)

# Import biased od matrix.
str_path_od_base = "./data_tabular/biased_od_v02.csv"
spsa_sumo.read_od_base(str_path_od_base)

# Import true flow.
str_path_flow_true = "./data_tabular/edge_flow_true_v02.csv"
spsa_sumo.read_true_flow(str_path_flow_true)

# Load GNN model and trained weighting factor.
str_path_gnn = "gnn_model/0718_08_32__GNN_params.pth"
str_path_adj_idx = "data_tabular/node_adg_matrix/arr_adj_idx.npy"
spsa_sumo.load_gnn_model(str_path_gnn, str_path_adj_idx)

# Measure time.
start_time = time.time()
# Boost gradient?
lv_bst_gradi = False
fac_bst_gradi = 5
# Which method for simulation?
lv_sim_gnn = True
lv_sim_gnn_min = True
# Want to have historical perturbing data from SUMO ?
lv_perturb_hist = False
lv_min_hist = False

# LOOP_1, Optimizing Iteration.
for iter_opti in range(1, int_nr_iter_opti + 1):
    
    # Parameters update for optimisation loop.
    # Step size (a_k), Perturbation coefficient (c_k).
    spsa_sumo.param_update(iter_opti)
    
    # Define empty list to collect gradient samples.
    lst_df_gradi_tmp1 = list()
    
    # LOOP_2, Gradient Estimattion Iteration.
    for iter_gradi in range(1, int_nr_iter_gradi + 1):
        
        # Update bernoulli random dataframe.
        spsa_sumo.bernoulli_delta(iter_opti, iter_gradi)
        
        # Get OD matrix perturbed in plus direction.
        df_od_perturbed_plus = spsa_sumo.perturbation_plus(iter_opti, iter_gradi)        
        # Run simulation with perturbed OD matrix and get flow dataframe.
        if lv_sim_gnn:
            # Using GNN.
            df_flow_perturbed_plus = spsa_sumo.sim_gnn_get_flow(iter_opti, iter_gradi, df_od_perturbed_plus)
        else:
            # Using SUMO.
            df_flow_perturbed_plus = spsa.sim_sumo_get_flow(iter_opti, iter_gradi, df_od_perturbed_plus)
            if lv_perturb_hist:
                df_od_perturbed_plus.to_csv(f"./history_od/{key}__{cal}_od_perturb_p_{iter_opti}_{iter_gradi}.csv")
                df_flow_perturbed_plus.to_csv(f"./history_flow/{key}__{cal}_flow_perturb_p_{iter_opti}_{iter_gradi}.csv")
            
        # Get Normalized RMSE with true flow value.
        gof_plus = spsa.n_Rmse_iq(spsa_sumo.df_true_flow, df_flow_perturbed_plus)
        print(f"        gof_plus: {gof_plus}, Iteration_{iter_opti}_{iter_gradi}.")
        
        # Get OD matrix perturbed in minmus direction.
        df_od_perturbed_minus = spsa_sumo.perturbation_minus(iter_opti, iter_gradi)
        # Run simulation with perturbed OD matrix and get flow dataframe.
        if lv_sim_gnn:
            # Using GNN.
            df_flow_perturbed_minus = spsa_sumo.sim_gnn_get_flow(iter_opti, iter_gradi, df_od_perturbed_minus)
        else:
            # Using SUMO.
            df_flow_perturbed_minus = spsa.sim_sumo_get_flow(iter_opti, iter_gradi, df_od_perturbed_minus)
            if lv_perturb_hist:
                df_od_perturbed_minus.to_csv(f"./history_od/{key}__{cal}_od_perturb_m_{iter_opti}_{iter_gradi}.csv")
                df_flow_perturbed_minus.to_csv(f"./history_flow/{key}__{cal}_flow_perturb_m_{iter_opti}_{iter_gradi}.csv")
        # Get Normalized RMSE with true flow value.
        gof_minus = spsa.n_Rmse_iq(spsa_sumo.df_true_flow, df_flow_perturbed_minus)
        print(f"        gof_minus: {gof_minus}, Iteration_{iter_opti}_{iter_gradi}.")
        
        # Calculate sample gradient and store it in list.
        with np.errstate(divide='ignore', invalid='ignore'):
            arr_gradi_tmp2 = (gof_plus - gof_minus) / (2*spsa_sumo.fl_perturb*spsa_sumo.arr_bernoulli)
        np.fill_diagonal(arr_gradi_tmp2,0)
        lst_df_gradi_tmp1.append(arr_gradi_tmp2)
    
    # Calculate mean value of smaple gradients.
    arr_gradi_mean_tmp1 = sum(lst_df_gradi_tmp1) / len(lst_df_gradi_tmp1)
    if lv_bst_gradi:
        arr_gradi_mean_tmp1 = arr_gradi_mean_tmp1 * fac_bst_gradi
    spsa_sumo.lst_gradi.append(arr_gradi_mean_tmp1)
    print(f"    Gradient estimation is ready. Iteration_{iter_opti}")
    
    # Minimizing with segmented step size value.
    # OD matrix will be adjusted in opposite direction of loss funtion gradient.
    spsa_sumo.minimization_loss(iter_opti)
    if lv_sim_gnn_min:
        # Using GNN.
        df_flow_min = spsa_sumo.sim_gnn_get_flow(iter_opti, 999, spsa_sumo.df_od)
    else:
        # Using SUMO.
        df_flow_min = spsa.sim_sumo_get_flow(iter_opti, 999, spsa_sumo.df_od)
        if lv_min_hist:
            spsa_sumo.df_od.to_csv(f"./history_od/{key}__{cal}_od_min_{iter_opti}.csv")
            df_flow_min.to_csv(f"./history_flow/{key}__{cal}_flow_min_{iter_opti}.csv")
    nRmse_min = spsa.n_Rmse(spsa_sumo.df_true_flow, df_flow_min)
    spsa_sumo.lst_nRmse.append(nRmse_min)
    
    if iter_opti == 1:
        spsa_sumo.nRmse_best = nRmse_min
        spsa_sumo.df_flow_best = df_flow_min
        spsa_sumo.df_od_best = spsa_sumo.df_od
    else:
        if nRmse_min < spsa_sumo.nRmse_best:
            spsa_sumo.nRmse_best = nRmse_min
            spsa_sumo.df_flow_best = df_flow_min
            spsa_sumo.df_od_best = spsa_sumo.df_od
    
    # Make time stamp.
    time_epoch = time.time() - start_time
    spsa_sumo.lst_time_epoch.append(int(time_epoch))
    
    print(f"    Iteration_{iter_opti}/{int_nr_iter_opti}: nRMSE {nRmse_min}")
    print(f"    Best nRMSE so far: {spsa_sumo.nRmse_best}")
    
# Store object for future purpose.
savePklFile(f"./reporting/{key}__{cal}_obj_spsa.pkl",spsa_sumo)

# Make history data frame.
dic_hist_spsa = {
    "Iteration"     : range(1, int_nr_iter_opti + 1),
    "Time"          : spsa_sumo.lst_time_epoch,
    "Perturbation"  : spsa_sumo.lst_perturb,
    "Step"          : spsa_sumo.lst_step,
    "NRMSE"         : spsa_sumo.lst_nRmse,
}
df_hist_spsa = pd.DataFrame(dic_hist_spsa)
df_hist_spsa.to_csv(f"./reporting/{key}__{cal}_result_spsa.csv")
spsa_sumo.df_od_best.to_csv(f"./reporting/{key}__{cal}_od_best.csv")
spsa_sumo.df_flow_best.to_csv(f"./reporting/{key}__{cal}_yflow_best.csv")

# Print out MAPE
from torchmetrics import MeanAbsolutePercentageError as MAPE
import torch

ts_od_best = torch.from_numpy(spsa_sumo.df_od_best.astype(float).values)
ts_od_true = torch.from_numpy(spsa_sumo.df_od_true.astype(float).values)

clc_mape = MAPE()
print(" ")
print(clc_mape(ts_od_best, ts_od_true))
print(" ")

# Print out some results.
print(" ")
print("         OD_BEST - OD_BASE ")
print(spsa_sumo.df_od_best.astype(int) - spsa_sumo.df_od_base.astype(int))
print(" ")
print("         OD_BASE")
print(spsa_sumo.df_od_base.astype(int))
print(" ")
print("         OD_BEST")
print(spsa_sumo.df_od_best.astype(int))
print(" ")
print("         OD_TRUE")
print(spsa_sumo.df_od_true.astype(int))
print(" ")
print("         BEST POINT")
print(
    df_hist_spsa[
        df_hist_spsa.NRMSE == df_hist_spsa.NRMSE.min()
    ].head(5)
)

# Plottings
fig_iter = go.Figure()

fig_iter.add_trace(
    go.Scatter(
        x= df_hist_spsa["Iteration"],
        y= df_hist_spsa["NRMSE"],
        line_color = "#000000",
    )
)

fig_iter.update_layout(
    title= "SPSA",
    xaxis_title= "Number Iteration",
    yaxis_title= "NRMES with y_mean",
    width= 500,
    height= 350,
    template= tempelete_01_white,
)

fig_iter.update_xaxes(
    range= [0, df_hist_spsa["Iteration"].max() + 5]
)

fig_iter.update_yaxes(
    range= [0,df_hist_spsa["NRMSE"].max()]
)

fig_iter.write_html(f"./figures/{key}__{cal}_IterNRMSE.html")

arr_flow_true_flatten = spsa_sumo.df_true_flow.values.flatten()
arr_flow_fdsa_flatten = spsa_sumo.df_flow_best.values.flatten()

fig_counts = go.Figure()

fig_counts.add_trace(
    go.Scatter(
        x= arr_flow_true_flatten,
        y= arr_flow_fdsa_flatten,
        mode= "markers",
        marker_symbol= "circle-open",
        marker_color= "#000000",   
    )
)

fig_counts.add_traces(
    go.Scatter(
        x= [0,500],
        y= [0,500],
        mode= "lines",        
        marker_color= "#fc4040",        
    )
)

fig_counts.update_layout(
    title= "SPSA",
    xaxis_title= "True Counts",
    yaxis_title= "SPSA Counts",
    width= 500,
    height= 500,
    showlegend= False,
    template= tempelete_01_white,
)

fig_counts.update_xaxes(
    range= [0, 500]
)

fig_counts.update_yaxes(
    range= [0,500]
)

fig_counts.write_html(f"./figures/{key}__{cal}_counts.html")
