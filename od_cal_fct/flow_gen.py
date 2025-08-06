# Import necessary modules.
import os
import pandas as pd

# Function to create tazRel.xml file.
def createTazRel (
    in_str_path_sumo_tmp, in_str_id_tazRel,
    in_str_sim_st, in_str_sim_end, in_df_od: pd.DataFrame
):
    
    str_path_output_tmp = in_str_path_sumo_tmp + "/tmp_tazRel.tazRel.xml"
    
    with open(str_path_output_tmp, "w") as tazRel:
        
        print(
            "<data>"
            , file= tazRel
        )

        # Define interval information. 
        print(
            "    <interval id=\"{}\" begin=\"{}\" end=\"{}\">".format(in_str_id_tazRel, in_str_sim_st, in_str_sim_end)
            , file= tazRel
        )
        
        # Loop calculation for each origin and destination pair.
        # Row index stands for origin and Column index stands for destination.
        tup_dim_tmp = in_df_od.shape
        for i_row in range(tup_dim_tmp[0]):
            for i_col in range(tup_dim_tmp[1]):
                val_tmp = in_df_od.iat[i_row, i_col]      # Flow value.
                name_row_tmp = in_df_od.iloc[i_row,:].name    # Name of Origin.
                name_col_tmp = in_df_od.iloc[:,i_col].name    # Name of Destination.
                # If there is no flow given to certain OD pair. Trip definition will not be updated.
                if name_row_tmp != name_col_tmp:
                    print(
                        "        <tazRelation from=\"{}\" to=\"{}\" count=\"{}\"/>".format(name_row_tmp, name_col_tmp, val_tmp)
                        , file= tazRel
                    )
                else:
                    continue
        
        # Interval closing.
        print(
            "    </interval>"
            , file= tazRel
        )    
        
        print(
            "</data>"
            , file= tazRel
        )
    
    return str_path_output_tmp

# Function to create odtrips.xml file.
def createOdTrips(
    in_str_path_sumo_tmp, in_str_path_taz, in_str_path_tazRel
):
    
    str_path_output_tmp = in_str_path_sumo_tmp + "/tmp_OD_trips.odtrips.xml"
    
    str_command_tmp = "od2trips " + "-n {} -z {} -o {} --no-step-log True".format(
        in_str_path_taz, in_str_path_tazRel, str_path_output_tmp
    )
    
    lv_suc_tmp = os.system(str_command_tmp)
    
    return lv_suc_tmp, str_path_output_tmp

# Function to create rou.xml file.
def createRoute(
    in_str_path_sumo_tmp, in_str_path_net, 
    in_str_path_odtrips, in_str_path_vType 
):
    
    str_path_output_tmp = in_str_path_sumo_tmp + "/tmp_od_routes.rou.xml"
    
    str_command_tmp = "duarouter -n {} --route-files {} -o {} --no-step-log True --additional-files {}".format(
        in_str_path_net, in_str_path_odtrips, str_path_output_tmp, in_str_path_vType
    )
    
    lv_suc_tmp = os.system(str_command_tmp)
    
    return lv_suc_tmp, str_path_output_tmp

# Function to create tmp_edgeInfo.additional.xml file
def createAddEdgeInfo(
    in_str_path_sumo_tmp, in_str_edgeInfo_id, in_str_path_edgeInfo
):
    str_path_output_tmp = in_str_path_sumo_tmp + "/tmp_edgeInfo.additional.xml"
    
    with open(str_path_output_tmp, "w") as edgeInfoAdd:
        
        print(
            "<additional>"
            , file= edgeInfoAdd
        )
        
        print(
            "    <edgeData id=\"{}\" file=\"{}\"/>".format(in_str_edgeInfo_id, in_str_path_edgeInfo)
            , file= edgeInfoAdd
        )
        
        print(
            "</additional>"
            , file= edgeInfoAdd
        )
        
    return str_path_output_tmp