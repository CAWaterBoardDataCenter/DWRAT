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
from basin_connectivity_matrix import basin_connectivity
from main_date_range import date_string
from water_allocation_tool_2021 import allocation_tool
from water_allocation_tool_2021_PVP import allocation_tool_PVP

##################################################################################
################################# FLOWS ##########################################
##################################################################################

# Read in the regular flows.csv. 
# This should be stored in the Upper Russian River folder, module 1, inputs
flows = pd.read_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/flows.csv")
# write flows.csv to main working directory input folder
flows.to_csv("input/flows.csv")
# Build basin_connectivity_matrix.csv with script
basin_connectivity_matrix = basin_connectivity("R_13_MSRR")
# write this to the Upper Russian River folder, module 1, inputs
basin_connectivity_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/basin_connectivity_matrix.csv")
basin_connectivity_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/basin_connectivity_matrix.csv")
basin_connectivity_matrix.to_csv("input/basin_connectivity_matrix.csv")
##################################################################################
################################ DEMAND ##########################################
##################################################################################
# Export a copy of the demand data set to the Upper Russian River folder
# create a list of a subset of columns
cols = ['APPL_ID', 'RIPARIAN', 'BASIN', 'MAINSTEM', 'UPPER_RUSSIAN', 'PRIORITY',
        'PRIORITY_CLASS', 'JAN_MEAN_DIV', 'FEB_MEAN_DIV', 'MAR_MEAN_DIV',
        'APR_MEAN_DIV', 'MAY_MEAN_DIV', 'JUN_MEAN_DIV', 'JUL_MEAN_DIV',
        'AUG_MEAN_DIV', 'SEP_MEAN_DIV', 'OCT_MEAN_DIV', 'NOV_MEAN_DIV',
        'DEC_MEAN_DIV', 'NULL_DEMAND', 'ZERO_DEMAND', 'MAY_SEPT_ZERO_DEMAND']
# create a list of the demand columns
demand_cols = ['JAN_MEAN_DIV', 'FEB_MEAN_DIV', 'MAR_MEAN_DIV',
        'APR_MEAN_DIV', 'MAY_MEAN_DIV', 'JUN_MEAN_DIV', 'JUL_MEAN_DIV',
        'AUG_MEAN_DIV', 'SEP_MEAN_DIV', 'OCT_MEAN_DIV', 'NOV_MEAN_DIV',
        'DEC_MEAN_DIV']
# create a list of desired dates in desired format
dates = ["2022-01",	"2022-02", "2022-03", "2022-04", "2022-05",	"2022-06", "2022-07", "2022-08", "2022-09",	"2022-10", "2022-11", "2022-12"]
# create a dictionary with demand column keys and dates
cols_dates_dict = dict(zip(demand_cols, dates))
# read in the demand dataset, specify columns above, set index to APPL_ID
all_data = pd.read_csv("UPPER_RUSSIAN_RIVER/RR_DATABASE_2022_NEW_and_PVID_priority.csv", usecols = cols, index_col = "APPL_ID")
# Rename the index to Allocation Tool field name
all_data.index.name = "USER"

################################ SET PVID DEMAND TO 0 ############################
all_data.loc[["A013557", "S014313"], demand_cols] = 0
# all_data.loc["A013557"]
# all_data.loc["S014313"]

################################ REPLACE 0s and NaNs ############################
# replace 0s with minimal value, and NaNs with another one
for col in demand_cols:
    all_data[col].replace(0.0, 0.0002022, inplace = True)
    all_data[col].fillna(0.00002022, inplace = True)
# rename columns                      
all_data.rename(columns = cols_dates_dict, inplace = True)

##################################################################################
################ UPPER RUSSIAN RIVER INPUTS FOR MODULE 1 (NONMAINSTEM) ###########
##################################################################################
# subset of records
riparian_nonmainstem_upper = all_data[(all_data["RIPARIAN"]=="Y") & (all_data["MAINSTEM"] == "N") & 
                                   (all_data["UPPER_RUSSIAN"]=="Y")]
appropriative_nonmainstem_upper = all_data[(all_data["RIPARIAN"]=="N") & (all_data["MAINSTEM"] == "N") & 
                                   (all_data["UPPER_RUSSIAN"]=="Y")]
# write output to Upper Russian River module 1 input folder
appropriative_nonmainstem_upper.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/appropriative_demand.csv")
riparian_nonmainstem_upper.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/riparian_demand.csv")

######################### CONSTRUCT USER CONNECTIVITY MATRICES###################
# requires demand and flows .csv's
# write output from above to working directory input folder
appropriative_nonmainstem_upper.to_csv("input/appropriative_demand.csv")
riparian_nonmainstem_upper.to_csv("input/riparian_demand.csv")
# run scripts
exec(open("./appropriative_user_matrices.py").read())
exec(open("./riparian_user_matrices.py").read())

# read in connectivity matrices output 
appropriative_user_matrix = pd.read_csv("input/appropriative_user_matrix.csv")
appropriative_connectivity_matrix = pd.read_csv("input/appropriative_user_connectivity_matrix.csv")
riparian_user_matrix = pd.read_csv("input/riparian_user_matrix.csv")
riparian_connectivity_matrix = pd.read_csv("input/riparian_user_connectivity_matrix.csv")

# write matrices to to module 1 input files
appropriative_user_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/appropriative_user_matrix.csv")
appropriative_connectivity_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/appropriative_user_connectivity_matrix.csv")
riparian_user_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/riparian_user_matrix.csv")
riparian_connectivity_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_1/input_m1/riparian_user_connectivity_matrix.csv") 

##################################################################################
############################## RUN ALLOCATION TOOL ###############################
##################################################################################

# SET DATA RANGE
# a data range can be specified with these two lines:
dates_to_run = date_string("2022-06", "2022-06")
# RUN ALLOCATION TOOL FOR MODULE 1
allocation_tool(dates_to_run)

# RETRIEVE OUTPUT
out_file_name = "_" + dates_to_run["Dates"].unique()[0] + "_" + dates_to_run["Dates"].unique()[-1]

basin_appropriative = pd.read_csv("output/basin_appropriative_output" + out_file_name + ".csv")
basin_riparian = pd.read_csv("output/basin_riparian_output" + out_file_name + ".csv")
user_appropriative = pd.read_csv("output/user_appropriative_output" + out_file_name + ".csv")
user_riparian= pd.read_csv("output/user_riparian_output" + out_file_name + ".csv")

basin_appropriative.to_csv("UPPER_RUSSIAN_RIVER/module_1/output_m1/basin_appropriative_output" + out_file_name + ".csv")
basin_riparian.to_csv("UPPER_RUSSIAN_RIVER/module_1/output_m1/basin_riparian_output" + out_file_name + ".csv")
user_appropriative.to_csv("UPPER_RUSSIAN_RIVER/module_1/output_m1/user_appropriative_output" + out_file_name + ".csv")
user_riparian.to_csv("UPPER_RUSSIAN_RIVER/module_1/output_m1/user_riparian_output" + out_file_name + ".csv")

##################################################################################
############################### EVAP LOSSES, PVP FLOWS ###########################
##################################################################################
# read in config file for evap, pvp
config_file = pd.read_excel("UPPER_RUSSIAN_RIVER/config_file.xlsx", index_col = "INPUT_NAME")
pvp_flow_mod = pd.DataFrame(data = 0, index = flows["BASIN"], columns = dates_to_run["Dates"].unique())
flows_proportional = pd.DataFrame(data = 0, index = flows["BASIN"], columns = dates_to_run["Dates"].unique())
flows_proportional.insert(0, "FLOWS_TO", flows["FLOWS_TO"].values)


#%%

for date in dates_to_run["Dates"].unique():
    flows_proportional[date] = np.divide(flows[date], sum(flows[date])).values
#    print(date + "_ALLOCATIONS")
#    print(sum(basin_appropriative[date + "_ALLOCATIONS"]))
#    print(sum(basin_riparian[date + "_ALLOCATIONS"]))
    sum_flow = sum(flows[date])
    sum_allocations = sum(basin_appropriative[date + "_ALLOCATIONS"] + basin_riparian[date + "_ALLOCATIONS"])
    net = sum_flow - sum_allocations
    evap_loss_share = config_file.loc["EVAP_LOSS"][date] / 3
    def_surp = net - evap_loss_share
    pvp = config_file.loc["PVP_FLOW"][date] - evap_loss_share
    flows_proportional[date] = flows_proportional[date] * max(0, def_surp)
    if def_surp > 0:
        print("surplus: " + str(def_surp))
        pvp_flow_mod.loc["R_02_MSRR", date] = pvp_flow_mod.loc["R_02_MSRR"][date] + pvp
    else:
        print("deficit: " + str(def_surp))
        pvp_flow_mod.loc["R_02_MSRR", date] = pvp_flow_mod.loc["R_02_MSRR"][date] + pvp + def_surp       

##################################################################################
################ UPPER RUSSIAN RIVER INPUTS FOR MODULE 2 (MAINSTEM) ##############
##################################################################################

flows_proportional.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/flows.csv")
flows_proportional.to_csv("input/flows.csv")
pvp_flow_mod.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/pvp_flow_mod.csv")
pvp_flow_mod.to_csv("input/pvp_flow_mod.csv")

appropriative_mainstem_upper = all_data[(all_data["RIPARIAN"]=="N") & (all_data["MAINSTEM"] == "Y") & 
                                   (all_data["UPPER_RUSSIAN"]=="Y")]
riparian_mainstem_upper = all_data[(all_data["RIPARIAN"]=="Y") & (all_data["MAINSTEM"] == "Y") & 
                                   (all_data["UPPER_RUSSIAN"]=="Y")]

# write output to Upper Russian River module 2 input folder
appropriative_mainstem_upper.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/appropriative_demand.csv")
riparian_mainstem_upper.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/riparian_demand.csv")

######################### CONSTRUCT USER CONNECTIVITY MATRICES###################
# requires demand and flows .csv's
# write output from above to working directory input folder
appropriative_mainstem_upper.to_csv("input/appropriative_demand.csv")
riparian_mainstem_upper.to_csv("input/riparian_demand.csv")

# run scripts
exec(open("./appropriative_user_matrices.py").read())
exec(open("./riparian_user_matrices.py").read())

# read in connectivity matrices output 
appropriative_user_matrix = pd.read_csv("input/appropriative_user_matrix.csv")
appropriative_connectivity_matrix = pd.read_csv("input/appropriative_user_connectivity_matrix.csv")
riparian_user_matrix = pd.read_csv("input/riparian_user_matrix.csv")
riparian_connectivity_matrix = pd.read_csv("input/riparian_user_connectivity_matrix.csv")

# write matrices to to module 2 input files
appropriative_user_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/appropriative_user_matrix.csv")
appropriative_connectivity_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/appropriative_user_connectivity_matrix.csv")
riparian_user_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/riparian_user_matrix.csv")
riparian_connectivity_matrix.to_csv("UPPER_RUSSIAN_RIVER/module_2/input_m2/riparian_user_connectivity_matrix.csv") 

#%%

# RUN ALLOCATION TOOL FOR MODULE 2
allocation_tool_PVP(dates_to_run)

# RETRIEVE OUTPUT
basin_appropriative_2 = pd.read_csv("output/basin_appropriative_output" + out_file_name + ".csv")
basin_riparian_2 = pd.read_csv("output/basin_riparian_output" + out_file_name + ".csv")
user_appropriative_2 = pd.read_csv("output/user_appropriative_output" + out_file_name + ".csv")
user_riparian_2 = pd.read_csv("output/user_riparian_output" + out_file_name + ".csv")

basin_appropriative_2.to_csv("UPPER_RUSSIAN_RIVER/module_2/output_m2/basin_appropriative_output" + out_file_name + ".csv")
basin_riparian_2.to_csv("UPPER_RUSSIAN_RIVER/module_2/output_m2/basin_riparian_output" + out_file_name + ".csv")
user_appropriative_2.to_csv("UPPER_RUSSIAN_RIVER/module_2/output_m2/user_appropriative_output" + out_file_name + ".csv")
user_riparian_2.to_csv("UPPER_RUSSIAN_RIVER/module_2/output_m2/user_riparian_output" + out_file_name + ".csv")

##################################################################################
######################### COMBINE MODULE 1 AND MODULE 2 OUTPUTS ##################
##################################################################################
# BASIN OUTPUT
mainstem_basins = ["R_02_MSRR", "R_03_MSRR", "R_04_MSRR", "R_05_MSRR", "R_06_MSRR", "R_09_MSRR", "R_10_MSRR", "R_12_MSRR", "R_13_MSRR"]
for output in [basin_appropriative, basin_riparian, basin_appropriative_2, basin_riparian_2]:
    output.set_index("BASIN", inplace = True)
for basin in mainstem_basins:
    basin_appropriative.loc[basin] = basin_appropriative_2.loc[basin]
    basin_riparian.loc[basin] = basin_riparian_2.loc[basin]
basin_appropriative.to_csv("UPPER_RUSSIAN_RIVER/combined_output/combined_basin_appropriative_output" + out_file_name + ".csv")
basin_riparian.to_csv("UPPER_RUSSIAN_RIVER/combined_output/combined_basin_riparian_output" + out_file_name + ".csv")
# USER OUTPUT
combined_user_riparian = pd.concat([user_riparian, user_riparian_2], axis = 0)
combined_user_appropriative = pd.concat([user_appropriative, user_appropriative_2], axis = 0)
combined_user_riparian = combined_user_riparian.sort_values(by = ["BASIN", "USER"])
combined_user_appropriative = combined_user_appropriative.sort_values(by = ["BASIN", "USER"])
combined_user_riparian.to_csv("UPPER_RUSSIAN_RIVER/combined_output/combined_user_riparian_output" + out_file_name + ".csv")
combined_user_appropriative.to_csv("UPPER_RUSSIAN_RIVER/combined_output/combined_user_appropriative_output" + out_file_name + ".csv")













