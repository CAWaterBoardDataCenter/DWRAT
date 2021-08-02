# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:29:08 2020

@author: dpedroja
"""
################################# REQURIES NUMPY VERSION 18.1.1 #################################### 

import pulp as pulp
import numpy as np
np.__version__
import pandas as pd
import datetime
from main_date_range import date_string

# FIRST SELECT YOUR TIME SERIES

# daily? monthly?
# format

# yyyy-mm-dd or yyyy-mm are the recommended daily / monthly date formats. But mm/dd/yyyy and yyyy-m-d are also available.

# if desired, select a data range with these two lines:
        
# dates_to_run = date_string("1996-06", "1996-06")
dates_to_run = date_string("2021-08", "2021-08")
dates_to_run.to_csv("input/data_range.csv")

# or just run existing file for evaluation by commenting above out
data_range = pd.read_csv("input/data_range.csv")

#### for yyyy-mm-dd run this line
# data_range["Dates"] = data_range["yyyy-mm-dd"]

# #### for yyyy-m-d run this line
# data_range["Dates"] = data_range["yyyy-m-d"]

# #### for m/d/yyyy run this line
# data_range["Dates"] = data_range["m/d/yyyy"]

# # #### for monthly run this line
data_range["Dates"] = data_range["yyyy-mm"]
data_range["Dates"].to_csv("output/data_range.csv", header = True)

# check your data range here:
# data_range["Dates"].unique()

# NOW YOU ARE READY

# RAW DATA
# flow
start = datetime.datetime.now().time()
flow_table_df = pd.read_csv('input/flows.csv', index_col = "BASIN")
# flow_table_df.sort_index(axis = "index")
flow_table_df = flow_table_df.sort_index(axis = "index")

# demand
rip_demand_df = pd.read_csv('input/riparian_demand.csv')
rip_demand_df.set_index("USER", inplace = True)
rip_demand_df.sort_index(axis = "index", inplace = True)
rip_users = rip_demand_df.index.values

app_demand_df = pd.read_csv('input/appropriative_demand.csv')
app_demand_df.set_index("USER", inplace = True)
app_demand_df.sort_index(axis = "index", inplace = True)
app_users = app_demand_df.index.values

app_users_count = np.size(app_users)
    
# basically just user location
riparian_basin_user_matrix_df = pd.read_csv("input/riparian_user_matrix.csv", index_col="BASIN")
riparian_basin_user_matrix_df.sort_index(axis = "index", inplace = True)
riparian_basin_user_matrix_df.sort_index(axis = "columns", inplace = True)
riparian_basin_user_matrix = riparian_basin_user_matrix_df.to_numpy()

appropriative_basin_user_matrix_df = pd.read_csv("input/appropriative_user_matrix.csv", index_col="BASIN")
appropriative_basin_user_matrix_df.sort_index(axis = "index", inplace = True)
appropriative_basin_user_matrix_df.sort_index(axis = "columns", inplace = True)
appropriative_basin_user_matrix = appropriative_basin_user_matrix_df.to_numpy()

# user connectivity
riparian_user_connectivity_matrix_df = pd.read_csv('input/riparian_user_connectivity_matrix.csv', index_col="BASIN")
riparian_user_connectivity_matrix_df.sort_index(axis = "index", inplace = True)
riparian_user_connectivity_matrix_df.sort_index(axis = "columns", inplace = True)
riparian_user_connectivity_matrix = riparian_user_connectivity_matrix_df.to_numpy()
riparian_user_connectivity_matrix_T = riparian_user_connectivity_matrix.T

appropriative_user_connectivity_matrix_df = pd.read_csv('input/appropriative_user_connectivity_matrix.csv', index_col="BASIN")
appropriative_user_connectivity_matrix_df.sort_index(axis = "index", inplace = True)
appropriative_user_connectivity_matrix_df.sort_index(axis = "columns", inplace = True)
appropriative_user_connectivity_matrix = appropriative_user_connectivity_matrix_df.to_numpy()

# basin connectivity

downstream_connectivity_df = pd.read_csv("input/basin_connectivity_matrix.csv", index_col = "BASIN")
# downstream_connectivity_df = pd.read_csv("input/basin_connectivity_matrix.csv")
# downstream_connectivity_df.set_index(downstream_connectivity_df["BASIN"].astype(str), drop = True, inplace = True)


# downstream_connectivity_df = downstream_connectivity_df.iloc[:, 1:]

# downstream_connectivity_df.sort_index(axis = "index", inplace = True)
# downstream_connectivity_df.sort_index(axis = "columns", inplace = True)
downstream_connectivity_matrix = downstream_connectivity_df.to_numpy()
upstream_connectivity_matrix = downstream_connectivity_matrix.T
# need another upstream connectivity matrix for modification in appropriative available flow calcs
upstream_connectivity_matrix_mod = downstream_connectivity_matrix.T

# basins = list(downstream_connectivity_df.index.astype(str))
basins = list(downstream_connectivity_df.index)
# type(downstream_connectivity_df.index[5])


############################################################################################
# Riparian output files
output_cols = data_range["Dates"].unique().tolist()

rip_basin_proportions_output = pd.DataFrame(columns=[output_cols], index=basins)
# rip_basin_proportions_output.index.name = "BASIN"
# rip_basin_available_flow = pd.DataFrame(columns=[output_cols], index=basins)

# Appropriative output files
app_user_allocations_output = pd.DataFrame(columns=[output_cols], index= app_users)
# app_user_allocations_output.index.name = "USER"

############################################################################################

for c, day in enumerate(data_range["Dates"].unique()):
    # print(c, day)
    riparian_demand_data = rip_demand_df[day].to_numpy()
    net_flow = flow_table_df[day].to_numpy()
    
    # AVAILABLE FLOW: 
    # available flow: initialize and populate a 1 x k list for available basin flow
    # available flow in basin k is sum of all upstream basin inflows less environmental requirement
    # Matrix operations:
    # upstream connectivity matrix * net flow
    # (k x k) * (k x 1) = k x 1 list of available net basin flows
         #      [1, 0, 0, 0, 0, 0, 0, 0]		[5.6]			[ 5.6]
         #   	[0, 1, 0, 0, 0, 0, 0, 0]		[5.6]			[ 5.6]
         #   	[1, 1, 1, 0, 0, 0, 0, 0]		[5.6]			[16.8]
         #   	[0, 0, 0, 1, 0, 0, 0, 0]	∙	[5.6]		=	[ 5.6]	
         #   	[1, 1, 1, 1, 1, 0, 0, 0]		[5.6]			[28.0]
         #   	[1, 1, 1, 1, 1, 1, 0, 0]		[5.6]			[33.6]
         #   	[0, 0, 0, 0, 0, 0, 1, 0]		[5.6]			[ 5.6]
         #   	[1, 1, 1, 1, 1, 1, 1, 1]		[5.6]			[44.8]
    available_flow_data = np.matmul(upstream_connectivity_matrix, net_flow.T) 
        # DOWNSTREAM PENALTY
    # number of users upstream of i divided by total users
    # Matrix/vector operations
    # row sum of the k x i user connectivity matrix / count of  i = k x 1 list of downstream penalties
    #           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]				
    #          	[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]				
    #         	[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]				
    #         	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]	row sum 	=	[ 2,  1,  4,  1,  7,  9,  1, 11]	/	11						
    #        	[0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]				
    #        	[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]				
    #        	[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]				
    #        	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #
    # 	        =	[0.18181818, 0.09090909, 0.36363636, 0.09090909, 0.63636364, 0.81818182, 0.09090909, 1.]
    downstream_penalty_list = np.divide(np.sum(riparian_user_connectivity_matrix, 1), np.count_nonzero(rip_users))
    
    # UPSTREAM BASIN DEMAND
    # basin-wide demand is the sum of user demand upstream of each basin
    # Matrix/vector operations:
    # 1 x i list of user demand ∙ i x k user connectivity matrix  = 1 x k basin demand matrix
    basin_rip_demand_data_T = np.matmul(riparian_demand_data, riparian_user_connectivity_matrix_T)
   
    # ALPHA
    # minimum of the ratios of downstream penalties to basin demands, element by element division, division by zero should return 0
    alpha = min(np.divide(downstream_penalty_list, basin_rip_demand_data_T, out = np.full_like(downstream_penalty_list, 999999999), where=basin_rip_demand_data_T!=0))

    # DICTIONARIES FOR CONSTRAINTS
    available_flow = {basins[k] : available_flow_data[k] for k, basin in enumerate(basins)}
    downstream_penalty = {basins[k] : downstream_penalty_list[k] for k, basin in enumerate(basins)}
    
    # DEFINE PROBLEM
    Riparian_LP = pulp.LpProblem("RiparianAllocation", pulp.LpMinimize)
    
    # DEFINE DECISION VARIABLES
    basin_proportions = pulp.LpVariable.dicts("Proportions", basins, 0, 1, cat="Continuous")
    # basin_proportions = pulp.LpVariable.dicts("Proportions", basins, 0, 1, cat="Integer")
    # convert dictionary of decision variables to an array
    basin_proportions_list = pd.Series(basin_proportions).to_numpy()
    
    # USER ALLOCATION
    # user allocation i is their basin's allocation * user i's demand
    # need a 1 x k array of basin proportions ∙ k x i basin user matrix * demand (element-wise) 
    # Matrix/vector operations:
    user_allocation_list = np.multiply((np.matmul(basin_proportions_list.T, riparian_basin_user_matrix)), riparian_demand_data)
    
    # dictionary
    user_allocation = {rip_users[i] : user_allocation_list[i] for i, user in enumerate(rip_users)}
    
    # UPSTREAM ALLOCATION: 
    # Sum of user allocations upstream of user i
    # Matrix/vector operations:
    # need k x i upstream user matrix ∙ i by 1 user allocation matrix = k x 1 upstream basin allocation
    #							                    [UserAllocation_A1]	       
    #   	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A2]	    [18*Proportions_A + 0]
    #       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]		[UserAllocation_A3]		[8*Proportions_B + 0]
    #       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A4]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 0]
    #       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]	∙	[UserAllocation_A5]	=	[3*Proportions_D + 0]
    #       [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]		[UserAllocation_A6]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 3*Proportions_D + 13*Proportions_E + 0]
    #       [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]		[UserAllocation_A7]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 3*Proportions_D + 13*Proportions_E + 14*Proportions_F + 0]
    #       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]		[UserAllocation_A8]		[9*Proportions_G + 0]
    #       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]		[UserAllocation_A9]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 3*Proportions_D + 13*Proportions_E + 14*Proportions_F + 9*Proportions_G + 8*Proportions_H + 0]
    #                   							[UserAllocation_A10]
    #                   							[UserAllocation_A11]
    upstream_allocation_list = np.matmul(riparian_user_connectivity_matrix, user_allocation_list)
    # dictionary
    upstream_allocation = {basins[k] : upstream_allocation_list[k] for k, basin in enumerate(basins)}
    
    # OBJECTIVE FUNCTION
    Riparian_LP += alpha * pulp.lpSum([basin_proportions[k]*downstream_penalty[k] for k in basins]) - pulp.lpSum([user_allocation[i] for i in rip_users])
    
    # CONSTRAINTS
    # mass balance
    for k in basins:
        Riparian_LP += pulp.lpSum([upstream_allocation[k]]) <= available_flow[k]
    
    # upstream cannot exceed downstream
    # need k by i downstream proportions matrix

    for k in basins:
        downstream_basins = list(downstream_connectivity_df.index[downstream_connectivity_df[k]==1])
        for j in downstream_basins:
            Riparian_LP += basin_proportions[j] <= basin_proportions[k]   
    
            
    # SOLVE USING PULP SOLVER
    Riparian_LP.solve()
    print("Status: ", pulp.LpStatus[Riparian_LP.status])
    for v in Riparian_LP.variables():
          print(v.name, "=", v.varValue)
    print("Objective = ", pulp.value(Riparian_LP.objective))
    
    # basin demand dictionary
    basin_demand = {basins[k] : (np.matmul(riparian_basin_user_matrix, riparian_demand_data)[k]) for k,basin in enumerate(basins)}
    
    # this loop is necessary to turn LP output into values
    basin_allocation = []
    for k, basin in enumerate(basins):
        basin_allocation.append(basin_proportions[basin].value() * basin_demand[basin])
    print("Basin Total Allocations", basin_allocation)
    # build output table        
    for k in basins:
        rip_basin_proportions_output.loc[k, [day]] = basin_proportions[k].varValue

#################################       APPROPRIATIVE_LP   ##############################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


    
        
    dates = data_range["Dates"]
    rip_demand_matrix = np.array(rip_demand_df[dates])
    
    rip_basin_proportions_output.to_csv("input/rip_basin_proportions_0.csv")
    
    # cum_demand_AUG = 146.4923911
    # cum_flow_AUG = 109.67466
    # proportions_AUG = cum_flow_AUG / cum_demand_AUG
    
    # cum_demand_SEP = 129.3441404
    # cum_flow_SEP = 78.58817
    # proportions_SEP = cum_flow_SEP / cum_demand_SEP
    
    rip_basin_proportions_output = pd.read_csv("input/rip_basin_proportions_AUG.csv", index_col = "BASIN")
    
    basin_proportion_matrix = np.array(rip_basin_proportions_output[dates])
    rip_user_allocations_output = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix))*rip_demand_matrix),columns=[dates], index=rip_users)
    rip_user_allocations_output.index.name = "USER"    
    rip_user_allocations_matrix = np.array(rip_user_allocations_output[dates])
    rip_basin_allocations = np.matmul(riparian_basin_user_matrix, rip_user_allocations_matrix)
    # rip_basin_allocations_matrix = pd.DataFrame(rip_basin_allocations, columns=[dates], index = basins) 
    rip_basin_allocations_output = pd.DataFrame(rip_basin_allocations, columns=[dates], index=basins)
    rip_basin_allocations_output.index.name = "BASIN"
      
  
    # static matrices
    new_df = pd.DataFrame(available_flow_data, index = basins)
    new_df.columns= ["available_flow_data"]
    new_df["net_flow"] = pd.DataFrame(net_flow, index = basins)     
    new_df["basin_allocations"] = rip_basin_allocations_output
    
    # x = sum(new_df["basin_allocations"][0:8])
    # x/0.74
    ### upstream riparian allocations
    new_df["rip_upstream_allocations"] =  np.matmul(riparian_user_connectivity_matrix, rip_user_allocations_matrix)
    # downstream riparian allocations
    rip_downstream_allocations = np.matmul(downstream_connectivity_matrix, rip_basin_allocations_output)
    new_df["rip_downstream_allocations"] = np.matmul(downstream_connectivity_matrix, rip_basin_allocations_output)
    
    # INITIAL VALUES
    
    ### remainder
    initial_remainder =  (rip_downstream_allocations.iloc[0].values - available_flow_data[0])
    new_df["remainder"] = initial_remainder[0]
    
    ### running total 
    initial_total = available_flow_data[0]
    new_df["running_total"] = initial_total
    
    ## net_appropriative_available
    if available_flow_data[0] <= rip_downstream_allocations.iloc[0].values:
        initial_naa = 0
    else:
        initial_naa = available_flow_data[0] - new_df["basin_allocations"][0]
    new_df["net_appropriative_available"] = initial_naa
    
    ### flow allocation
    if new_df["remainder"][0] >= 0:
        initial_flow_allocation = new_df["net_flow"][0]
    else:
        initial_flow_allocation = new_df["basin_allocations"][0]
    new_df["flow_allocation"] = initial_flow_allocation
    
    
    # need another upstream connectivity matrix for modification in appropriative available flow calcs
    downstream_connectivity_mod_df = pd.DataFrame(downstream_connectivity_df)
    downstream_connectivity_matrix_mod = downstream_connectivity_mod_df.to_numpy()
    upstream_connectivity_matrix_mod = downstream_connectivity_matrix_mod.T
    
    # need upstream_connectivity_matrix without the diagonal 1s
    np.fill_diagonal(upstream_connectivity_matrix_mod, 0)
    # now need to add back ones for headwaters basins
    headwaters_NEW_ROUTE = ['L_14', 'L_15', 'L_16', 'L_17', 'L_18', 'L_19', 'L_20', 'L_21', 'L_23', 'L_27', 'L_28']
    hw = []
    for basin in headwaters_NEW_ROUTE:
        hw.append(basins.index(basin))        
    for k in hw:
        upstream_connectivity_matrix_mod[k][k] = 1
    # need a matrix of directly upstream basins
    up_df = pd.read_csv("input/next_upstream.csv", index_col = "BASIN")
    up_matrix = up_df.to_numpy()
    
    next_downstream = pd.read_csv("input/next_downstream.csv", index_col = "BASIN")
    
    basin_order_LSR =[k for k, basin in enumerate(basins[1:])]
    

    for k in basin_order_LSR:
        # sum of upstream allocations
        a = np.matmul(upstream_connectivity_matrix_mod, new_df["flow_allocation"])
        # STEP 1 running total of cumulative available flow less cumulative upstream flow allocations 
        #        NOT including basin k   
        # if k is a headwaters basin this equals available flow
        if upstream_connectivity_matrix_mod[k+1][k+1] == 1:
            new_df["running_total"][k+1] = new_df["available_flow_data"][k+1]
        # otherwise subtract upstream flow allocations
        else:
            new_df["running_total"][k+1] = new_df["available_flow_data"][k+1] - a[k+1]
        # STEP 2 remainder, i.e. the deficit or surplus of downstream demand
        if upstream_connectivity_matrix_mod[k+1][k+1] == 1:
            new_df["remainder"][k+1] = new_df["rip_downstream_allocations"][k+1] - new_df["running_total"][k+1]
        else:
            new_df["remainder"][k+1] = np.matmul(up_matrix[k+1], new_df["remainder"]) - new_df["running_total"][k+1]
        # STEP 3 Flow allocation
        if new_df["remainder"][k+1] > 0:
            if new_df["remainder"][next_downstream.index.get_loc(next_downstream[next_downstream.iloc[k+1]==1].index[0])] >0:
                new_df["flow_allocation"][k+1] = new_df["net_flow"][k+1]
            else: 
                new_df["flow_allocation"][k+1] = min(new_df["net_flow"][k+1], new_df["basin_allocations"][k+1])
        else:
            new_df["flow_allocation"][k+1] = min(new_df["net_flow"][k+1], new_df["basin_allocations"][k+1])
        # STEP 4 net appropriative flow
        if new_df["remainder"][k+1] > 0:
            if new_df["remainder"][next_downstream.index.get_loc(next_downstream[next_downstream.iloc[k+1]==1].index[0])] >0:
                new_df["net_appropriative_available"][k+1] = 0
            else: 
                # new_df["net_appropriative_available"][k+1] = min((new_df["available_flow_data"][k+1]-new_df["rip_upstream_allocations"][k+1]), new_df["basin_allocations"][k+1])
                new_df["net_appropriative_available"][k+1] = new_df["available_flow_data"][k+1]-new_df["rip_upstream_allocations"][k+1]
        else:
            new_df["net_appropriative_available"][k+1] = new_df["available_flow_data"][k+1]-new_df["rip_upstream_allocations"][k+1]
        
        # run twice
        for k in basin_order_LSR:
        # sum of upstream allocations
            a = np.matmul(upstream_connectivity_matrix_mod, new_df["flow_allocation"])
            # STEP 1 running total of cumulative available flow less cumulative upstream flow allocations 
            #        NOT including basin k   
            # if k is a headwaters basin this equals available flow
            if upstream_connectivity_matrix_mod[k+1][k+1] == 1:
                new_df["running_total"][k+1] = new_df["available_flow_data"][k+1]
            # otherwise subtract upstream flow allocations
            else:
                new_df["running_total"][k+1] = new_df["available_flow_data"][k+1] - a[k+1]
            # STEP 2 remainder, i.e. the deficit or surplus of downstream demand
            if upstream_connectivity_matrix_mod[k+1][k+1] == 1:
                new_df["remainder"][k+1] = new_df["rip_downstream_allocations"][k+1] - new_df["running_total"][k+1]
            else:
                new_df["remainder"][k+1] = np.matmul(up_matrix[k+1], new_df["remainder"]) - new_df["running_total"][k+1]
            # STEP 3 Flow allocation
            if new_df["remainder"][k+1] > 0:
                if new_df["remainder"][next_downstream.index.get_loc(next_downstream[next_downstream.iloc[k+1]==1].index[0])] >0:
                    new_df["flow_allocation"][k+1] = new_df["net_flow"][k+1]
                else: 
                    new_df["flow_allocation"][k+1] = min(new_df["net_flow"][k+1], new_df["basin_allocations"][k+1])
            else:
                new_df["flow_allocation"][k+1] = min(new_df["net_flow"][k+1], new_df["basin_allocations"][k+1])
            
            # STEP 4 net appropriative flow
            if new_df["remainder"][k+1] > 0:
                if new_df["remainder"][next_downstream.index.get_loc(next_downstream[next_downstream.iloc[k+1]==1].index[0])] >0:
                    new_df["net_appropriative_available"][k+1] = 0
                else: 
                    # new_df["net_appropriative_available"][k+1] = min((new_df["available_flow_data"][k+1]-new_df["rip_upstream_allocations"][k+1]), new_df["basin_allocations"][k+1])
                    new_df["net_appropriative_available"][k+1] = new_df["available_flow_data"][k+1]-new_df["rip_upstream_allocations"][k+1]
            else:
                new_df["net_appropriative_available"][k+1] = new_df["available_flow_data"][k+1]-new_df["rip_upstream_allocations"][k+1]
            
                new_df.to_csv("output/new_df_working.csv")
  
  
    
#########################################################################################
#########################################################################################
##################################################################################################################################################################################
#########################################################################################
##################################################################################################################################################################################
#########################################################################################

    
    unallocated_flow = np.array(net_flow - basin_allocation)

    
    # if np.sum(net_app_available) > 0:
#     if np.sum(unallocated_flow) > 0:
#         # app_available_flow = np.matmul(upstream_connectivity_matrix, unallocated_flow)
#         # app_available_flow = net_app_available
#         # rip_basin_available_flow[[day]] = net_app_available
#         # rip_basin_available_flow[[day]] = available_flow
# #       # print("Excess flow available for Appropriative allocation:" , net_app_available.round(3))
#     else:
#         app_available_flow = 0
#         print("No flow is available for appropriative allocations")
#         rip_basin_available_flow[[day]] = app_available_flow
#         for i in app_users:
#             app_user_allocations_output.loc[i, [day]] = 0
#         print(c+1, "of", len(data_range["Dates"].unique()), "complete. Processing day:", day)
#         continue
    
    appropriative_demand_data = app_demand_df[day].to_numpy()
    priority = app_demand_df["PRIORITY"].to_numpy()
    shortage_penalty_data = np.array([(app_users_count - priority[i]) for i, user in enumerate(app_users)])    
        
    # DICTIONARIES
    app_demand = {app_users[i] : appropriative_demand_data[i] for i, user in enumerate(app_users)}
    shortage_penalty = {app_users[i] : shortage_penalty_data[i] for i, user in enumerate(app_users)}
    # app_available_flow = {basins[k] : available_flow_data[k] for k, basin in enumerate(basins)}
    app_available_flow = {basins[k] : new_df["net_appropriative_available"][k] for k, basin in enumerate(basins)}

    
    # DEFINE PROBLEM
    Appropriative_LP = pulp.LpProblem("AppropriativeProblem", pulp.LpMinimize)
    
    # DEFINE DECISION VARIABLES
    user_allocation = pulp.LpVariable.dicts("UserAllocation", app_users, 0)
    # convert dictionary of decision variables to an array
    user_allocation_list = pd.Series(user_allocation).to_numpy()
    
    # OBJECTIVE FUNCTION
    Appropriative_LP += pulp.lpSum(  (shortage_penalty[user])*(app_demand[user]-user_allocation[user]) for user in app_users)
    
    # UPSTREAM APPROPRIATIVE BASIN ALLOCATION
    # sum of appropriative allocations upstream of basin k
    # Matrix/vector operations
    # k by i upstream matrix ∙ i by 1 user_allocation for a k by 1 result constrained to available flow.
    #                   							[UserAllocation_A1 ]		
    #   	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A2 ]		[UserAllocation_A11 + UserAllocation_A4]	     
    #       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]		[UserAllocation_A3 ]		[UserAllocation_A3]
    #       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A4 ]		[UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4]
    #       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]	∙	[UserAllocation_A5 ]	=	[UserAllocation_A7]
    #       [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]		[UserAllocation_A6 ]		[UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4 + UserAllocation_A6 + UserAllocation_A7 + UserAllocation_A8]
    #       [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]		[UserAllocation_A7 ]		[UserAllocation_A1 + UserAllocation_A10 + UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4 + UserAllocation_A6 + UserAllocation_A7 + UserAllocation_A8]
    #       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]		[UserAllocation_A8 ]		[UserAllocation_A9]
    #       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]		[UserAllocation_A9 ]		[UserAllocation_A1 + UserAllocation_A10 + UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4 + UserAllocation_A5 + UserAllocation_A6 + UserAllocation_A7 + UserAllocation_A8 + UserAllocation_A9]
    #                   							[UserAllocation_A10]
    #                   							[UserAllocation_A11]
    upstream_basin_allocation = np.matmul(appropriative_user_connectivity_matrix, user_allocation_list)
    # dictionary
    upstream_dict = {basins[k] : upstream_basin_allocation[k] for k, basin in enumerate(basins)}
    
    # CONSTRAINTS:
    # 1.  allocation is <= available flow;
    for basin in basins:
        # Appropriative_LP += pulp.lpSum(upstream_dict[basin]) <= available_flow[basin]
        Appropriative_LP += pulp.lpSum(upstream_dict[basin]) <= app_available_flow[basin]
        
    # 2.  allocation is <= to reported demand
    for user in app_users:
        Appropriative_LP += pulp.lpSum(user_allocation[user]) <= (app_demand[user])
    # 3. ADDED: sum of user allocation <= total unallocated flow
    # Appropriative_LP += pulp.lpSum(user_allocation[i] for i in app_users) <= unallocated_flow
    # ????????????????????????????
    Appropriative_LP += pulp.lpSum(user_allocation[i] for i in app_users) <= app_available_flow
    
    
    
    # SOLVE USING PULP SOLVER
    Appropriative_LP.solve()
    print("status:", pulp.LpStatus[Appropriative_LP.status])
    # for v in Appropriative_LP.variables():
    #     print(v.name, "=", v.varValue)
    print("Objective = ", pulp.value(Appropriative_LP.objective))
    
    # this loop is necessary to turn LP output into values
    user_allocations = []
    for i, user in enumerate(app_users):
        user_allocations.append(user_allocation[user].value())

    app_basin_allocations = np.matmul(appropriative_basin_user_matrix, user_allocations)
    print("Basin Appropriative Allocations:") 
    print(app_basin_allocations)
    
    # build output table1
    for i in app_users:
        app_user_allocations_output.loc[i, [day]] = user_allocation[i].varValue
    print(c+1, "of", len(data_range["Dates"].unique()), "complete. Processing day:", day)
    
finish = datetime.datetime.now().time()
print("Hi. I'm done. Time at completion was:", finish, ". Starting time was:", start)


# # basin output
# # new_df.columns
# basin_output_df = new_df[['available_flow_data', 'net_flow', "flow_allocation"]]
# basin_output_df["riparian_proportions"] = rip_basin_proportions_output
# basin_output_df["rip_basin_allocations"] = new_df["basin_allocations"] 
# basin_output_df["rip_basin_demand"] = basin_demand.values()
# basin_output_df["rip_basin_shortage"] = basin_output_df["rip_basin_demand"] - basin_output_df["rip_basin_allocations"] 
# basin_output_df["rip_basin_shortage_%"] = basin_output_df["rip_basin_shortage"] / basin_output_df["rip_basin_demand"]
# basin_output_df["appropriative_output"] = "*"
# basin_output_df["appropriative_available_flow"] = new_df["net_appropriative_available"]
# basin_output_df["app_basin_allocations"] = app_basin_allocations
# app_demand_matrix = app_demand_df[dates].to_numpy()
# basin_output_df["app_basin_demand"] = np.matmul(appropriative_basin_user_matrix , app_demand_matrix)
# basin_output_df["app_basin_shortage"] = basin_output_df["app_basin_demand"] - basin_output_df["app_basin_allocations"] 
# basin_output_df["app_basin_shortage_%"] = basin_output_df["app_basin_shortage"] / basin_output_df["app_basin_demand"]

# basin_output_df.to_csv("output/basin_output.csv")

# # # user output
# # ## Appropriative
# user_output_df = app_demand_df[["BASIN", "PRIORITY"]]
# user_output_df["app_allocations"] = app_user_allocations_output[dates]
# user_output_df["app_demand"] = app_demand_df[dates]
# user_output_df["app_shortage"] = user_output_df["app_demand"] - user_output_df["app_allocations"]
# user_output_df["app_shortage_%"] = user_output_df["app_shortage"] / user_output_df["app_demand"]
# user_output_df.to_csv("output/appropriative_user_output_df.csv")

# ## Riparian
# del (rip_user_allocations_output)
# rip_demand_matrix = np.array(rip_demand_df[dates])           
# basin_proportion_matrix = np.array(rip_basin_proportions_output[dates])
# rip_user_allocations_output = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix))*rip_demand_matrix), index=rip_demand_df.index)
# rip_user_allocations_output.columns = dates.values
# rip_user_allocations_output.index

# rip_user_output_df = rip_demand_df[["BASIN", "PRIORITY"]]
# rip_user_output_df["rip_allocations"] = rip_user_allocations_output[dates]
# rip_user_output_df["rip_demand"] = rip_demand_df[dates]
# rip_user_output_df["rip_shortage"] = rip_user_output_df["rip_demand"] - rip_user_output_df["rip_allocations"]
# rip_user_output_df["rip_shortage_%"] = rip_user_output_df["rip_shortage"] / rip_user_output_df["rip_demand"]
# rip_user_output_df.to_csv("output/riparian_user_output_df.csv")


# AUGUST basin output for Keenan
# new_df.columns
basin_output_df = new_df[['available_flow_data', "flow_allocation", "net_flow"]]
basin_output_df["aug_riparian_proportions"] = rip_basin_proportions_output
basin_output_df["aug_rip_basin_allocations"] = new_df["basin_allocations"] 
basin_output_df["aug_rip_basin_demand"] = basin_demand.values()
basin_output_df["aug_rip_basin_shortage"] = basin_output_df["aug_rip_basin_demand"] - basin_output_df["aug_rip_basin_allocations"] 
basin_output_df["aug_rip_basin_shortage_%"] = basin_output_df["aug_rip_basin_shortage"] / basin_output_df["aug_rip_basin_demand"]
# basin_output_df["appropriative_output"] = "*"
# basin_output_df["appropriative_available_flow"] = new_df["net_appropriative_available"]
basin_output_df["aug_app_basin_allocations"] = app_basin_allocations
app_demand_matrix = app_demand_df[dates].to_numpy()
basin_output_df["aug_app_basin_demand"] = np.matmul(appropriative_basin_user_matrix , app_demand_matrix)
basin_output_df["aug_app_basin_shortage"] = basin_output_df["aug_app_basin_demand"] - basin_output_df["aug_app_basin_allocations"] 
basin_output_df["aug_app_basin_shortage_%"] = basin_output_df["aug_app_basin_shortage"] / basin_output_df["aug_app_basin_demand"]

basin_output_df.to_csv("output/basin_output_AUGUST_aug.csv")

# # user output
# ## Appropriative
user_output_df = app_demand_df[["BASIN", "RIPARIAN", "PRIORITY"]]
user_output_df["2021-08_ALLOCATION"] = app_user_allocations_output[dates]
user_output_df["2021-08_DEMAND"] = app_demand_df[dates]
user_output_df["2021-08_SHORTAGE"] = user_output_df["2021-08_DEMAND"] - user_output_df["2021-08_ALLOCATION"]
user_output_df["2108-08_%_SHORTAGE"] = user_output_df["2021-08_SHORTAGE"] / user_output_df["2021-08_DEMAND"]


## Riparian
del (rip_user_allocations_output)
rip_demand_matrix = np.array(rip_demand_df[dates])           
basin_proportion_matrix = np.array(rip_basin_proportions_output[dates])
rip_user_allocations_output = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix))*rip_demand_matrix), index=rip_demand_df.index)
rip_user_allocations_output.columns = dates.values
rip_user_allocations_output.index


rip_user_output_df = rip_demand_df[["BASIN", "RIPARIAN", "PRIORITY"]]
rip_user_output_df["2021-08_ALLOCATION"] = rip_user_allocations_output[dates]
rip_user_output_df["2021-08_DEMAND"] = rip_demand_df[dates]
rip_user_output_df["2021-08_SHORTAGE"] = rip_user_output_df["2021-08_DEMAND"] - rip_user_output_df["2021-08_ALLOCATION"]
rip_user_output_df["2108-08_%_SHORTAGE"] = rip_user_output_df["2021-08_SHORTAGE"] / rip_user_output_df["2021-08_DEMAND"]


combined_user_output_df = pd.concat([user_output_df,rip_user_output_df], axis = 0)

combined_user_output_df.to_csv("output/combined_user_output_df_AUGUST.csv")





