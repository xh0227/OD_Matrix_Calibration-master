"""
IMPORT MODULES
"""
import os
import torch
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from torch_geometric.data import Data

# DEFNITION OF EDGES
dic_edge = {
    "alpha" : ["alpha_e", "alpha_w"], "beta" : ["beta_e", "beta_w"],
    "1" : ["1_e", "1_w"], "2" : ["2_e", "2_w"], "3" : ["3_e", "3_w"],
    "4" : ["4_e", "4_w"], "5" : ["5_e", "5_w"], "6" : ["6_e", "6_w"],
    "a" : ["a_n", "a_s"], "b" : ["b_n", "b_s"], "c" : ["c_n", "c_s"],
    "d" : ["d_n", "d_s"], "e" : ["e_n", "e_s"], "f" : ["f_n", "f_s"],
}

# DEFNITION OF EDGE SEQUENCE
lst_edge_seq = [
    "alpha", "a", "b", "c", "d", "e", "f",
    "1", "2", "3", "4", "5", "6", "beta",
]

# Helper functions for FUNCTION.1
def searchEdge (in_id_edge: str, in_dic_edge: dict):
    for key, val in in_dic_edge.items():
        if in_id_edge in val:
            return key
        else:
            continue
    return 'NaN'

# FUNCTION.1 CONVERT EDGE FLOW DATA.
def read_edge_flow (in_str_dir_flow: str, in_str_dir_out: str):
    
    # List up edge flow data file in xml format.
    lst_edge_flow = os.listdir(in_str_dir_flow)
    lst_edge_flow_fil = [i for i in lst_edge_flow if "edgeInfo" in i]
    total_len = len(lst_edge_flow_fil)
    total_idx = 0
    
    # LOOP_1: Each edge flow data file (.xml)
    for edge_flow_xml in lst_edge_flow_fil:
        total_idx += 1
        file_path_tmp1 = in_str_dir_flow + "/" + edge_flow_xml
        tree_tmp1 = ET.parse(file_path_tmp1)
        root_tmp1 = tree_tmp1.getroot()
        nrEdges_tmp1 = len(root_tmp1[0].findall("edge"))
        lst_df_single_tmp1 = []
        
        # LOOP_2: Each edge info in the file.
        for edge_idx in range(nrEdges_tmp1):
            dic_tmp2= {}
            dic_tmp2["edge_id"] = root_tmp1[0][edge_idx].get("id", "NaN")
            edge_density_tmp2 = float(root_tmp1[0][edge_idx].get("density", 'NaN'))
            edge_speed_tmp2 = float(root_tmp1[0][edge_idx].get("speed", 'NaN'))
            edge_flow_tmp2 = edge_density_tmp2 * edge_speed_tmp2 * 3.6
            dic_tmp2["edge_flow"] = edge_flow_tmp2
            df_single_tmp2 = pd.DataFrame(
                [list(dic_tmp2.values())],
                columns= list(dic_tmp2.keys())                
            )
            lst_df_single_tmp1.append(df_single_tmp2)
        
        # Concatnate edge flows per each file.
        df_concat_rows_tmp1 = pd.concat(
            lst_df_single_tmp1,
            ignore_index= True, sort= False
        )
        # Add upper edge "id" column.
        df_concat_rows_tmp1["id"] = df_concat_rows_tmp1["edge_id"].apply(
            lambda x: searchEdge(x, dic_edge)
        )
        # Create final output dataframe.
        df_out_tmp1 = pd.DataFrame(np.full((len(lst_edge_seq),3),np.nan),
            columns= ["id", "flow_0", "flow_1"]
        )
        # LOOP_3: Insert proper infomation into output dataframe.
        # Predefined edge sequence will be applied here.        
        for idx_out_row in range(len(lst_edge_seq)):
            # edge id from sequence.
            id_tmp3 = lst_edge_seq[idx_out_row]
            # child edge ids from dictionary.
            edge_id_0_tmp3 = dic_edge[id_tmp3][0]
            edge_id_1_tmp3 = dic_edge[id_tmp3][1]
            # flow value from child edges.
            flow_0_tmp3 = df_concat_rows_tmp1[
                df_concat_rows_tmp1["edge_id"] == edge_id_0_tmp3
            ]["edge_flow"].values
            flow_1_tmp3 = df_concat_rows_tmp1[
                df_concat_rows_tmp1["edge_id"] == edge_id_1_tmp3
            ]["edge_flow"].values
            # Insert values into output df.
            df_out_tmp1.iat[idx_out_row, 0] = id_tmp3
            df_out_tmp1.iat[idx_out_row, 1] = float(flow_0_tmp3)
            df_out_tmp1.iat[idx_out_row, 2] = float(flow_1_tmp3)
        
        # Create output file name and store it.
        out_fName_tmp1 = edge_flow_xml.replace(".xml", ".csv")
        out_path_tmp1 = in_str_dir_out + "/" + out_fName_tmp1
        df_out_tmp1.to_csv(out_path_tmp1, index= False)
        
        # Notice to user.
        print("Processing...{}/{}".format(total_idx, total_len))
    
    # Notice to user.
    print("TASK DONE, CHECK OUTPUT DIR!!")
    
# Helper functions for FUNCTION.2
    # NONE
# FUNCTION.2 CONVERT ADJCENCY MATRIX.
def conv_adj_mat (in_str_file_adjMat: str, in_str_file_edge_idx: str):
    # Import raw adjcency matrix csv file.
    df_adj = pd.read_csv(in_str_file_adjMat, index_col= "idx")
    arr_adj = df_adj.values
    # Check if adjcency matrix is symmetric.
    if not np.array_equal(arr_adj, arr_adj.T):
        print("Adjcency matrix is not symmetric. Check again!")
        return
    else:
        # If symmetric, store non-zero indices pair. 
        lst_edge_index = []
        for idx_row in range(df_adj.shape[0]):
            for idx_col in range(df_adj.shape[1]):
                if df_adj.iat[idx_row, idx_col] == 0:
                    continue
                else:
                    lst_edge_index.append([int(idx_row), int(idx_col)])
    # Make edge_index array and store it.
    arr_adj_2cols = np.array(lst_edge_index)
    np.save(in_str_file_edge_idx, arr_adj_2cols)
    # Notice to user.
    print("TASK DONE, CHECK OUTPUT DIR!!")
    return

# Helper functions for FUNCTION.3
    # NONE
# FUNCTION.3 MAKE GRAPHS
def make_graphs (
    in_dir_od:str, in_dir_flow:str, in_file_edge_idx:str,
    in_dir_graphs:str, 
):
    # Import file list for od matrix.
    lst_df_od = os.listdir(in_dir_od)
    lst_df_od_fil = [i for i in lst_df_od if "od_sample" in i]
    
    # Import file list for flow matrix.
    lst_df_flow = os.listdir(in_dir_flow)
    lst_df_flow_fil = [i for i in lst_df_flow if "edgeInfo" in i]
    
    # Import edge_index array file.
    arr_edge_idx = np.load(in_file_edge_idx)
    tensor_edge_idx = torch.tensor(arr_edge_idx, dtype=torch.long)
    
    # LOOP_1: Each od & flow sample
    nrOdFlow = len(lst_df_od_fil)
    for idx in range(nrOdFlow):
        idx += 1
        str_idx_tmp1 = "{:04d}".format(idx)
        str_file_od_tmp1 = list(filter(lambda x: str_idx_tmp1 in x, lst_df_od_fil))[0]
        str_file_flow_tmp1 = list(filter(lambda x: str_idx_tmp1 in x, lst_df_flow_fil))[0]
        str_path_od_tmp1 = in_dir_od + "/" + str_file_od_tmp1
        str_path_flow_tmp1 = in_dir_flow + "/" + str_file_flow_tmp1
        df_od_tmp1 = pd.read_csv(str_path_od_tmp1, index_col= 0)
        df_flow_tmp1 = pd.read_csv(str_path_flow_tmp1, index_col=0)
        
        # LOOP_2: Each node (edge in the real map...)
        # for idx_node in range(df_od_tmp1.shape[0]):
        #     if idx_node == 0:
        #         arr_x_tmp2 = df_od_tmp1.iloc[idx_node, :].values
        #         arr_y_tmp2 = np.array(
        #             [df_flow_tmp1.iat[idx_node, 1], df_flow_tmp1.iat[idx_node, 2]]
        #         )
        #     else:
        #         arr_x_tmp2 = np.vstack((arr_x_tmp2, df_od_tmp1.iloc[idx_node, :].values))
        #         arr_y_cur_tmp2 = np.array(
        #             [df_flow_tmp1.iat[idx_node, 1], df_flow_tmp1.iat[idx_node, 2]]
        #         )
        #         arr_y_tmp2 = np.vstack((arr_y_tmp2, arr_y_cur_tmp2))
        
        # Define ndarrays.
        arr_x_tmp1 = df_od_tmp1.values
        arr_y_tmp1 = df_flow_tmp1.values
        # Define Tensors.
        tensor_x_tmp1 = torch.tensor(arr_x_tmp1, dtype=torch.int)
        tensor_y_tmp1 = torch.tensor(arr_y_tmp1, dtype=torch.float)
        # Define Graph and store it.
        graph_tmp1 = Data(
            x= tensor_x_tmp1, y= tensor_y_tmp1,
            edge_index= tensor_edge_idx.t().contiguous()
        )
        str_path_graph_tmp1 = in_dir_graphs + "/" + "graph_{:04d}.pt".format(idx)
        torch.save(graph_tmp1, str_path_graph_tmp1)
        # Notice to user about status.
        print("Process Done {}/{}...".format(idx, nrOdFlow+1))
    
    # Final notive to user.
    print("TASK DONE. CHECK OUTPUT DIR !!!")
    
# Helper functions for FUNCTION.4
    # NONE
# FUNCTION.4 Read edge flow xml file and return dataframe.
def xml2df_edge_flow (in_str_path_edgeInfo:str):   
    
    # Read edgeInfo file. 
    tree_tmp1 = ET.parse(in_str_path_edgeInfo)
    root_tmp1 = tree_tmp1.getroot()
    nrEdges_tmp1 = len(root_tmp1[0].findall("edge"))
    lst_df_single_tmp1 = []
    
    # LOOP_2: Each edge info in the file.
    for edge_idx in range(nrEdges_tmp1):
        dic_tmp2= {}
        dic_tmp2["edge_id"] = root_tmp1[0][edge_idx].get("id", "NaN")
        edge_density_tmp2 = float(root_tmp1[0][edge_idx].get("density", 'NaN'))
        edge_speed_tmp2 = float(root_tmp1[0][edge_idx].get("speed", 'NaN'))
        edge_flow_tmp2 = edge_density_tmp2 * edge_speed_tmp2 * 3.6
        dic_tmp2["edge_flow"] = edge_flow_tmp2
        df_single_tmp2 = pd.DataFrame(
            [list(dic_tmp2.values())],
            columns= list(dic_tmp2.keys())                
        )
        lst_df_single_tmp1.append(df_single_tmp2)
    
    # Concatnate edge flows per each file.
    df_concat_rows_tmp1 = pd.concat(
        lst_df_single_tmp1,
        ignore_index= True, sort= False
    )
    # Add upper edge "id" column.
    df_concat_rows_tmp1["id"] = df_concat_rows_tmp1["edge_id"].apply(
        lambda x: searchEdge(x, dic_edge)
    )
    # Create final output dataframe.
    df_out_tmp1 = pd.DataFrame(np.full((len(lst_edge_seq),3),np.nan),
        columns= ["id", "flow_0", "flow_1"]
    )
    # LOOP_3: Insert proper infomation into output dataframe.
    # Predefined edge sequence will be applied here.        
    for idx_out_row in range(len(lst_edge_seq)):
        # edge id from sequence.
        id_tmp3 = lst_edge_seq[idx_out_row]
        # child edge ids from dictionary.
        edge_id_0_tmp3 = dic_edge[id_tmp3][0]
        edge_id_1_tmp3 = dic_edge[id_tmp3][1]
        # flow value from chied edge.
        flow_0_tmp3 = df_concat_rows_tmp1[
            df_concat_rows_tmp1["edge_id"] == edge_id_0_tmp3
        ]["edge_flow"].values
        flow_1_tmp3 = df_concat_rows_tmp1[
            df_concat_rows_tmp1["edge_id"] == edge_id_1_tmp3
        ]["edge_flow"].values
        # Insert values into output df.
        df_out_tmp1.iat[idx_out_row, 0] = id_tmp3
        df_out_tmp1.iat[idx_out_row, 1] = float(flow_0_tmp3)
        df_out_tmp1.iat[idx_out_row, 2] = float(flow_1_tmp3)
    
    # Reset Index.    
    df_out_tmp1.set_index("id", inplace= True, drop= True)
        
    return df_out_tmp1

# Next Function.

if __name__ == "main":
    path_flow = "data_sumo/tmp_edgeInfo/edgeInfo_spsa.xml"
    xml2df_edge_flow(path_flow)