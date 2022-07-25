# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:20:46 2022

@author: DPedroja
"""
import os
import os.path
import pandas as pd    
import numpy as np
from datetime import datetime as dt

def get_paths(folder):
    paths_list = []
    for path in os.listdir(folder):
        full_path = os.path.join(folder, path)
        if os.path.isfile(full_path):
            paths_list.append(full_path)
    return paths_list

def Y_D(mdy):
    x = '{:%Y-%m}'.format(dt.strptime(mdy, "%m/%d/%Y"))
    return x
def Y_M_D(ymd):
    x = '{:%Y-%m}'.format(dt.strptime(ymd, "%Y-%m-%d"))
    return x

if __name__ == "__main__":
    path_1 = "SUPPLY/basins.csv"
    path_2 = "SUPPLY/Monthly_AcFt_RR_Independent_ZERO_PRECIP_Jun22_to_Oct31.csv"
    
def process_prms_flows(path):
    # read in basins.csv
    basins = pd.read_csv(path)
    # select headwater basins
    headwater_basins = basins[basins["MAINSTEM"] == "N"]
    headwater_basins = headwater_basins[["BASIN", "FLOWS_TO"]]
    headwater_basins.set_index("BASIN", inplace = True, drop = True)
    # select mainstem basins
    mainstem_basins = basins[basins["MAINSTEM"] == "Y"]
    mainstem_basins = mainstem_basins[["BASIN", "FLOWS_TO"]]
    mainstem_basins.set_index("BASIN", inplace = True, drop = True)
    # read in raw flows
    path_2 = "SUPPLY/Monthly_AcFt_RR_Independent_ZERO_PRECIP_Jun22_to_Oct31.csv"
    flows = pd.read_csv(path_2, dtype = {"Date" : str})
    # change date format
    flows["Date"] = flows["Date"].apply(Y_M_D)
    # add these dates to headwater_basins and mainstem_basins
    for date in flows["Date"].values:
        mainstem_basins[date] = 0
        headwater_basins[date] = 0
    flows.set_index("Date", inplace = True)
    # rename basins
    flows.columns = headwater_basins.index
    # transpose data
    flows = flows.transpose()
    flows.index.name = "BASIN"
    flows.columns = flows.columns.values
    flows.insert(0, "FLOWS_TO", headwater_basins["FLOWS_TO"].values)
    # merge & sort
    all_flows = pd.concat([flows, mainstem_basins], axis = 0)
    all_flows = all_flows.sort_index()
    all_flows.to_csv("SUPPLY/all_flows_processed.csv")
    
def make_date_strings(date_format, first, last):
    dates_list = pd.date_range(start=first, end=last)
    data_range = pd.DataFrame(dates_list, columns=["yyyy-mm-dd"])
    data_range["m/d/yyyy"] = data_range["yyyy-mm-dd"].apply(lambda x: "{:%#m/%#d/%Y}".format(x))
    data_range["yyyy-m-d"] = data_range["yyyy-mm-dd"].apply(lambda x: "{:%Y-%#m-%#d}".format(x))
    data_range["yyyy-mm"]  = data_range["yyyy-mm-dd"].apply(lambda x: "{:%Y-%m}".format(x))
    data_range["Dates"] = data_range[date_format]
    data_range["Dates"].to_csv("output/data_range.csv", header = True)
    return data_range
#
#
#def make_paths(UPPER_LOWER):
#
#    cwd = os.getcwd()
#    if UPPER_LOWER == "UPPER":
#        mod_1_path = os.path.join(cwd, "UPPER_RUSSIAN_RIVER/module_1")
#        mod_2_path = os.path.join(cwd, "UPPER_RUSSIAN_RIVER/module_2")
#        combined_output_path = os.path.join(cwd, "UPPER_RUSSIAN_RIVER/combined_output")
#        demand_path = os.path.join(cwd, "DEMAND")
#    elif UPPER_LOWER == "LOWER":
#        mod_1_path = os.path.join(cwd, "LOWER_RUSSIAN_RIVER/module_1")
#        mod_2_path = os.path.join(cwd, "LOWER_RUSSIAN_RIVER/module_2")
#        combined_output_path = os.path.join(cwd, "LOWER_RUSSIAN_RIVER/combined_output")
#        demand_path = os.path.join(cwd, "DEMAND")
#    return mod_1_path, mod_2_path, combined_output_path, demand_path
#
#
#def demand_processing(demand_path, file_name):
#    # create a list of a subset of columns
#    cols = ['APPL_ID', 'RIPARIAN', 'BASIN', 'MAINSTEM', 'UPPER_RUSSIAN', 'PRIORITY',
#            'PRIORITY_CLASS', 'JAN_MEAN_DIV', 'FEB_MEAN_DIV', 'MAR_MEAN_DIV', 'APR_MEAN_DIV', 'MAY_MEAN_DIV', 'JUN_MEAN_DIV', 'JUL_MEAN_DIV',
#            'AUG_MEAN_DIV', 'SEP_MEAN_DIV', 'OCT_MEAN_DIV', 'NOV_MEAN_DIV', 'DEC_MEAN_DIV', 'NULL_DEMAND', 'ZERO_DEMAND', 'MAY_SEPT_ZERO_DEMAND']
#    # create a list of the demand columns
#    demand_cols = ['JAN_MEAN_DIV', 'FEB_MEAN_DIV', 'MAR_MEAN_DIV', 'APR_MEAN_DIV', 'MAY_MEAN_DIV', 'JUN_MEAN_DIV', 'JUL_MEAN_DIV',
#            'AUG_MEAN_DIV', 'SEP_MEAN_DIV', 'OCT_MEAN_DIV', 'NOV_MEAN_DIV', 'DEC_MEAN_DIV']
#    # create a list of desired dates in desired format
#    dates = ["2022-01",	"2022-02", "2022-03", "2022-04", "2022-05",	"2022-06", "2022-07", "2022-08", "2022-09",	"2022-10", "2022-11", "2022-12"]
#    # create a dictionary with demand column keys and dates
#    cols_dates_dict = dict(zip(demand_cols, dates))
#    # NOW READ IN THE DEMAND DATASET, specify columns above, set index to APPL_ID
#    all_data = pd.read_csv(os.path.join(demand_path, file_name), usecols = cols, index_col = "APPL_ID")
#    # name the index
#    all_data.index.name = "USER"
#    for col in demand_cols:
#        all_data[col].replace(0.0, 0.0002022, inplace = True)
#        all_data[col].fillna(0.00002022, inplace = True)
#    # rename columns                      
#    all_data.rename(columns = cols_dates_dict, inplace = True)
#    return all_data, dates
#
#def demand_subset(upper_Y_N, mainstem_Y_N):
#    # subset of records
#    riparian_nonmainstem = all_data[(all_data["RIPARIAN"]=="Y") & (all_data["MAINSTEM"] == mainstem_Y_N) & 
#                                       (all_data["UPPER_RUSSIAN"]== upper_Y_N)]
#    appropriative_nonmainstem = all_data[(all_data["RIPARIAN"]=="N") & (all_data["MAINSTEM"] == mainstem_Y_N) & 
#                                       (all_data["UPPER_RUSSIAN"]== upper_Y_N)]
#    # write output to Upper/Lower Russian River module 1 input folder
#    appropriative_nonmainstem.to_csv("input/appropriative_demand.csv")
#    riparian_nonmainstem.to_csv("input/riparian_demand.csv")
#    
#    
#def calculations(dates_to_run, UPPER_LOWER):
#    for date in dates_to_run["Dates"].unique():
#        flows_proportional[date] = np.divide(flows[date], sum(flows[date])).values
#        sum_flow = sum(flows[date])
#        sum_allocations = sum(basin_app[date + "_ALLOCATIONS"] + basin_rip[date + "_ALLOCATIONS"])
#        net = sum_flow - sum_allocations
#        if UPPER_LOWER == "UPPER":
#            evap_loss_share = config_file.loc["EVAP_LOSS"][date] / 3
#        elif UPPER_LOWER == "LOWER":
#            evap_loss_share = config_file.loc["EVAP_LOSS"][date] / 2
#        def_surp = net - evap_loss_share
#        pvp = config_file.loc["PVP_FLOW"][date] - evap_loss_share
#        flows_proportional[date] = flows_proportional[date] * max(0, def_surp)
#        if def_surp > 0 and UPPER_LOWER == "UPPER":
#            pvp_flow_mod.loc["R_02_M", date] = pvp_flow_mod.loc["R_02_M"][date] + pvp
#            net_PVP = pvp
#        elif def_surp > 0 and UPPER_LOWER == "LOWER":
#            pvp_flow_mod.loc["R_14_M", date] = pvp_flow_mod.loc["R_14_M"][date] + pvp
#            net_PVP = pvp
#        elif def_surp < 0 and UPPER_LOWER == "UPPER":
#            pvp_flow_mod.loc["R_02_M", date] = pvp_flow_mod.loc["R_02_M"][date] + pvp + def_surp       
#            net_PVP = pvp + def_surp
#        elif def_surp < 0 and UPPER_LOWER == "LOWER":
#            pvp_flow_mod.loc["R_14_M", date] = pvp_flow_mod.loc["R_14_M"][date] + pvp + def_surp       
#            net_PVP = pvp + def_surp
#        # write flows (even if 0)
#        flows_proportional.to_csv("input/flows.csv")
#        # write PVP flows
#        pvp_flow_mod.to_csv("input/pvp_flow_mod.csv")
#        return net_PVP
#    



