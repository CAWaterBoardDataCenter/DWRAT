# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:26:43 2022

@author: DPedroja
"""
##################################################################################
############ Prepare Upper Russian River Allocation-Tool input .csv files ########
##################################################################################
import pandas as pd
import numpy as np
import os.path
from code_modules import *

########################  SELECT AMONG DATE FORMATS BELOW: ##########################    
# MONTHLY:  "yyyy-mm" (PREFERRED)
# DAILY:    "yyyy-mm-dd" (PREFERRED), "yyyy-m-d" , "m/d/yyyy" (ACCEPTED)

#########################  SPECIFY FIRST AND LAST DATES: ############################ 
if __name__ == "__main__":
    date_format = "yyyy-mm"
    first = "2022-08"
    last = "2022-08"
############################# SPECIFY OUTLET BASIN: #################################   
    # this is for the basin connectivity module below 
    outlet = "R_13_M"
    
###################### POPULATE AND READ IN CONFIG FILE: ############################
# read in config file for evap, pvp
config_file = pd.read_excel("UPPER_RUSSIAN_RIVER/config_file.xlsx", index_col = "INPUT_NAME")
#####################################################################################
########### NOW YOU CAN RUN THE CODE AS LONG AS THE OUTLET BASIN IS CORRECT: ########
#####################################################################################

# Make some paths for later
# specify either "UPPER" or "LOWER"
def make_paths(UPPER_LOWER):
    import os.path
    cwd = os.getcwd()
    if UPPER_LOWER == "UPPER":
        mod_1_path = os.path.join(cwd, "UPPER_RUSSIAN_RIVER/module_1")
        mod_2_path = os.path.join(cwd, "UPPER_RUSSIAN_RIVER/module_2")
        combined_output_path = os.path.join(cwd, "UPPER_RUSSIAN_RIVER/combined_output")
        demand_path = os.path.join(cwd, "DEMAND")
    elif UPPER_LOWER == "LOWER":
        mod_1_path = os.path.join(cwd, "LOWER_RUSSIAN_RIVER/module_1")
        mod_2_path = os.path.join(cwd, "LOWER_RUSSIAN_RIVER/module_2")
        combined_output_path = os.path.join(cwd, "LOWER_RUSSIAN_RIVER/combined_output")
        demand_path = os.path.join(cwd, "DEMAND")
    return mod_1_path, mod_2_path, combined_output_path, demand_path

mod_1_path, mod_2_path, combined_output_path, demand_path = make_paths("UPPER")
     
##################################################################################
################################# FLOWS ##########################################
##################################################################################
# Change working directory to module 1
os.chdir(mod_1_path)
# Verify the unmodified flows.csv is in the Upper Russian River folder, module 1, inputs
flows = pd.read_csv("input/flows.csv")

# Build basin_connectivity_matrix.csv with basin_connectivity_matrix.py code_module 
# SPECIFY OUTLET ABOVE !!!!!!
basin_connectivity_matrix = basin_connectivity_matrix.basin_connectivity_main(outlet)
# write this to the Upper Russian River folder, module 2, inputs since it will be the same for both modules
basin_connectivity_matrix.to_csv(os.path.join(mod_2_path, "input/basin_connectivity_matrix.csv"))

##################################################################################
################################ DEMAND ##########################################
##################################################################################

def demand_processing(demand_path, file_name):
    # create a list of a subset of columns
    cols = ['APPLICATION_NUMBER', "APPLICATION_PRIMARY_OWNER", 'RIPARIAN', 'BASIN', 'MAINSTEM_RR', 'UPPER_RUSSIAN', 
            'JAN_MEAN_DIV', 'FEB_MEAN_DIV', 'MAR_MEAN_DIV', 'APR_MEAN_DIV', 'MAY_MEAN_DIV', 'JUN_MEAN_DIV', 'JUL_MEAN_DIV',
            'AUG_MEAN_DIV', 'SEP_MEAN_DIV', 'OCT_MEAN_DIV', 'NOV_MEAN_DIV', 'DEC_MEAN_DIV', 'NULL_DEMAND', 'ZERO_DEMAND', 'MAY_SEPT_ZERO_DEMAND']
    # create a list of the demand columns
    demand_cols = ['JAN_MEAN_DIV', 'FEB_MEAN_DIV', 'MAR_MEAN_DIV', 'APR_MEAN_DIV', 'MAY_MEAN_DIV', 'JUN_MEAN_DIV', 'JUL_MEAN_DIV',
            'AUG_MEAN_DIV', 'SEP_MEAN_DIV', 'OCT_MEAN_DIV', 'NOV_MEAN_DIV', 'DEC_MEAN_DIV']
    # create a list of desired dates in desired format
    dates = ["2022-01",	"2022-02", "2022-03", "2022-04", "2022-05",	"2022-06", "2022-07", "2022-08", "2022-09",	"2022-10", "2022-11", "2022-12"]
    # create a dictionary with demand column keys and dates
    cols_dates_dict = dict(zip(demand_cols, dates))
    # NOW READ IN THE DEMAND DATASET, specify columns above, set index to APPL_ID
    all_data = pd.read_csv(os.path.join(demand_path, file_name), usecols = cols, index_col = "APPLICATION_NUMBER")
    # name the index
    all_data.index.name = "USER"
    for col in demand_cols:
        all_data[col].replace(0.0, 0.0002022, inplace = True)
        all_data[col].fillna(0.00002022, inplace = True)
    # rename columns                      
    all_data.rename(columns = cols_dates_dict, inplace = True)
    return all_data, dates

all_data, dates = demand_processing(demand_path, "RUSSIAN_RIVER_DATABASE_2022.csv")


###############################################################################################
########################### ASSIGN PRIORITY DATES #############################################
###############################################################################################


#### Module to assign PRIORITY numbers
#### Overrides values from a table

x = pd.read_clipboard()
x.set_index(index, inplace = True)

def PRIORITY(demand_path, index, field_name)

cwd = os.getcwd()
path = os.path.join(cwd, "DEMAND/RUSSIAN_RIVER_DATABASE_2022.csv")
#path_override_U_L = "UPPER"
index = "APPLICATION_NUMBER"
field = "ASSIGNED_PRIORITY_DATE_SUB"


priority_dates = pd.read_csv(path, usecols = [index, field], index_col = index)

priority_dates["PRIORITY"] = priority_dates.sample(frac = 1).rank(axis = 0, method = "first").reindex_like(priority_dates)
priority_dates.index.name = "APPLICATION_NUMBER"


#x["PRIORITY"] = x["ASSIGNED_PRIORITY_DATE_SUB"].sample(frac = 1).rank(axis = 0, method = "first").reindex_like(x)

if path_override_U_L == "UPPER":
    override_path = os.path.join(cwd, "DEMAND/OVERRIDE_PRIORITY_UPPER.csv")
if path_override_U_L == "LOWER":
    override_path = os.path.join(cwd, "DEMAND/OVERRIDE_PRIORITY_LOWER.csv")
    
override_priorities = pd.read_csv(override_path, index_col = "APPLICATION_NUMBER")
    
priority_dates["PRIORITY"].loc[override_priorities.index] = override_priorities["PRIORITY"].values















###############################################################################################
########################### DEMAND DATA MODIFICATIONS ##########################################
###############################################################################################

# Set PVID demand to 0 
all_data.loc[["A013557", "S014313"], dates] = 0
# all_data.loc["S014313"]

##################################################################################
################ UPPER/LOWER RUSSIAN RIVER INPUTS FOR MODULE 1 (NONMAINSTEM) #####
##################################################################################
# select "Y" or "N" for UPPER_RUSSIAN, and MAINSTEM_RR
def demand_subset(upper_Y_N, mainstem_Y_N):
    # subset of records
    riparian_nonmainstem = all_data[(all_data["RIPARIAN"]=="Y") & (all_data["MAINSTEM_RR"] == mainstem_Y_N) & 
                                       (all_data["UPPER_RUSSIAN"]== upper_Y_N)]
    appropriative_nonmainstem = all_data[(all_data["RIPARIAN"]=="N") & (all_data["MAINSTEM_RR"] == mainstem_Y_N) & 
                                       (all_data["UPPER_RUSSIAN"]== upper_Y_N)]
    # write output to Upper/Lower Russian River module 1 input folder
    appropriative_nonmainstem.to_csv("input/appropriative_demand.csv")
    riparian_nonmainstem.to_csv("input/riparian_demand.csv")

demand_subset("Y", "N")

# Build  diverter connectivity_matrix .csv's with riparian/appropriative_user_matrices.py code_module 
# run imported modules
appropriative_user_matrices.appropriative_user_matrices_main()
riparian_user_matrices.riparian_user_matrices_main()

##################################################################################
############################## RUN ALLOCATION TOOL ###############################
##################################################################################
# Using data range and date format set above
data_range = utils.make_date_strings(date_format, first, last)
# RUN ALLOCATION TOOL FOR MODULE 1, assign variables for 4 outputs
(basin_app, basin_rip, user_app, user_rip) = water_allocation_tool_2022.main(data_range)

#%%
##################################################################################
############################### EVAP LOSSES, PVP FLOWS ###########################
##################################################################################
# Change path to module 2
os.chdir(mod_2_path)

# make some empty dataframes
pvp_flow_mod = pd.DataFrame(data = 0, index = flows["BASIN"], columns = data_range["Dates"].unique())
flows_proportional = pd.DataFrame(data = 0, index = flows["BASIN"], columns = data_range["Dates"].unique())
flows_proportional.insert(0, "FLOWS_TO", flows["FLOWS_TO"].values)

# This loop calculates remaining natural flow per basin and applies evaporative losses,
# and thereby generates the flows.csv for module 2 as well as PVP flows for the pvp_flow_mod.csv
def calculations(data_range, UPPER_LOWER):
    for date in data_range["Dates"].unique():
        flows_proportional[date] = np.divide(flows[date], sum(flows[date])).values
        sum_flow = sum(flows[date])
        sum_allocations = sum(basin_app[date + "_ALLOCATIONS"] + basin_rip[date + "_ALLOCATIONS"])
        net = sum_flow - sum_allocations
        if UPPER_LOWER == "UPPER":
            evap_loss_share = config_file.loc["EVAP_LOSS"][date] / 3
        elif UPPER_LOWER == "LOWER":
            evap_loss_share = config_file.loc["EVAP_LOSS"][date] / 2
        def_surp = net - evap_loss_share
        pvp = max(0, config_file.loc["PVP_FLOW"][date] - evap_loss_share)
        flows_proportional[date] = flows_proportional[date] * max(0, def_surp)
        if def_surp > 0 and UPPER_LOWER == "UPPER":
            pvp_flow_mod.loc["R_02_M", date] = pvp_flow_mod.loc["R_02_M"][date] + pvp
            net_PVP = pvp
        elif def_surp > 0 and UPPER_LOWER == "LOWER":
            pvp_flow_mod.loc["R_17_M", date] = pvp_flow_mod.loc["R_17_M"][date] + pvp
            net_PVP = pvp
        elif def_surp < 0 and UPPER_LOWER == "UPPER":
            pvp_flow_mod.loc["R_02_M", date] = pvp_flow_mod.loc["R_02_M"][date] + pvp + def_surp       
            net_PVP = pvp + def_surp
        elif def_surp < 0 and UPPER_LOWER == "LOWER":
            pvp_flow_mod.loc["R_17_M", date] = pvp_flow_mod.loc["R_17_M"][date] + pvp + def_surp       
            net_PVP = pvp + def_surp
        # write flows (even if 0)
        flows_proportional.to_csv("input/flows.csv")
        # write PVP flows
        pvp_flow_mod.to_csv("input/pvp_flow_mod.csv")
        return net_PVP
net_PVP = calculations(data_range, "UPPER")

##################################################################################
################ UPPER/LOWER RUSSIAN RIVER INPUTS FOR MODULE 2 (MAINSTEM) ########
##################################################################################

demand_subset("Y", "Y")

# Build  diverter connectivity_matrix .csv's with riparian/appropriative_user_matrices.py code_module 
# re-run imported modules
appropriative_user_matrices.appropriative_user_matrices_main()
riparian_user_matrices.riparian_user_matrices_main()

# RUN PVP ALLOCATION TOOL VERSION FOR MODULE 2
# use code_module
(basin_app_2, basin_rip_2, user_app_2, user_rip_2) = water_allocation_tool_2021_PVP.PVP_main(data_range)

##################################################################################
######################### COMBINE MODULE 1 AND MODULE 2 OUTPUTS ##################
##################################################################################
# set working directory to output path
os.chdir(combined_output_path)
# date suffix
out_file_name = "_" + data_range["Dates"].unique()[0] + "_" + data_range["Dates"].unique()[-1]

# BASIN OUTPUT
# overwrite module 1 mainstem output with module 2 basin output
mainstem_basins = ["R_02_M", "R_03_M", "R_04_M", "R_05_M", "R_06_M", "R_09_M", "R_10_M", "R_12_M", "R_13_M"]
for basin in mainstem_basins:
    basin_app.loc[basin] = basin_app_2.loc[basin]
    basin_rip.loc[basin] = basin_rip_2.loc[basin]
# Write output to csv's
basin_app.to_csv("combined_basin_appropriative_output" + out_file_name + ".csv")
basin_rip.to_csv("combined_basin_riparian_output" + out_file_name + ".csv")

# USER OUTPUT
# just combine the lists
combined_user_riparian = pd.concat([user_rip, user_rip_2], axis = 0).sort_values(by = ["BASIN", "USER"])

# 2022-06_DEMAND with no zeros	
# 2022-06_ALLOCATIONS_AF based on this	
# 2022-06_PERCENT_ALLOCATION

combined_user_appropriative = pd.concat([user_app, user_app_2], axis = 0).sort_values(by = ["BASIN", "USER"])
# Write output to csv's
combined_user_riparian.to_csv("combined_user_riparian_output" + out_file_name + ".csv")
combined_user_appropriative.to_csv("combined_user_appropriative_output" + out_file_name + ".csv")


# SUMMARY
summary_df = pd.DataFrame()
month = data_range["Dates"].values[0]
summary_df["VALUE"] = "name"
summary_df["VALUE"]["MONTH"] = month
summary_df["VALUE"]["PVP_FLOW"] = config_file[month].loc["PVP_FLOW"]
summary_df["VALUE"]["net_PVP"] = net_PVP
#config_file[month].loc["PVP_FLOW"]
#net_PVP
summary_df["VALUE"]["EVAP_LOSSES"] = config_file[month].loc["EVAP_LOSS"].round(1)
summary_df["VALUE"]["Natural_af/m"] = basin_rip[month + "_FLOW"].sum()

riparian_nonMS_alloc = user_rip[month + "_ALLOCATIONS"].sum()
riparian_nonMS_demand = user_rip[month + "_DEMAND"].sum()
riparian_nonMS_shortage = riparian_nonMS_demand - riparian_nonMS_alloc

app_nonMS_alloc = user_app[month + "_ALLOCATIONS"].sum()
app_nonMS_demand = user_app[month + "_DEMAND"].sum()
app_nonMS_shortage = app_nonMS_demand - app_nonMS_alloc

riparian_MS_alloc = user_rip_2[month + "_ALLOCATIONS"].sum()
riparian_MS_demand = user_rip_2[month + "_DEMAND"].sum()
riparian_MS_shortage = riparian_MS_demand - riparian_MS_alloc

appropriative_MS_alloc = user_app_2[month + "_ALLOCATIONS"].sum()
appropriative_MS_demand = user_app_2[month + "_DEMAND"].sum()
app_MS_shortage = appropriative_MS_demand - appropriative_MS_alloc

flow_to_lower_RR = net_PVP - appropriative_MS_alloc
