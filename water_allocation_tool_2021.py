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


# FIRST SELECT YOUR TIME SERIES DATE FORMAT

# yyyy-mm-dd or yyyy-mm are the recommended daily / monthly date formats. But mm/dd/yyyy and yyyy-m-d are also available.

# a data range can be specified with these two lines:
dates_to_run = date_string("2021-05", "2021-09")
dates_to_run.to_csv("input/data_range.csv")

# or just run existing file for evaluation by commenting above out
data_range = pd.read_csv("input/data_range.csv")

#### for yyyy-mm-dd run this line
# data_range["Dates"] = data_range["yyyy-mm-dd"]

# #### for yyyy-m-d run this line
# data_range["Dates"] = data_range["yyyy-m-d"]

# #### for m/d/yyyy run this line
# data_range["Dates"] = data_range["m/d/yyyy"]

# # #### for monthly yyyy-mm run this line
data_range["Dates"] = data_range["yyyy-mm"]
data_range["Dates"].to_csv("output/data_range.csv", header = True)

# check your data range here:
# data_range["Dates"].unique()

# NOW YOU ARE READY

# READ IN RAW DATA
# flow
start = datetime.datetime.now().time()
flow_table_df = pd.read_csv('input/flows.csv', index_col = "BASIN")
# flow_table_df.sort_index(axis = "index")
flow_table_df = flow_table_df.sort_index(axis = "index")

# riparian demand
rip_demand_df = pd.read_csv('input/riparian_demand.csv')
rip_demand_df.set_index("USER", inplace = True)
rip_demand_df.sort_index(axis = "index", inplace = True)
rip_users = rip_demand_df.index.values

# appropriative demand
app_demand_df = pd.read_csv('input/appropriative_demand.csv')
app_demand_df.set_index("USER", inplace = True)
app_demand_df.sort_index(axis = "index", inplace = True)
app_users = app_demand_df.index.values

app_users_count = np.size(app_users)
    
# basically just user BASIN in matrix form
riparian_basin_user_matrix_df = pd.read_csv("input/riparian_user_matrix.csv", index_col="BASIN")
riparian_basin_user_matrix_df.sort_index(axis = "index", inplace = True)
riparian_basin_user_matrix_df.sort_index(axis = "columns", inplace = True)
riparian_basin_user_matrix = riparian_basin_user_matrix_df.to_numpy()

appropriative_basin_user_matrix_df = pd.read_csv("input/appropriative_user_matrix.csv", index_col="BASIN")
appropriative_basin_user_matrix_df.sort_index(axis = "index", inplace = True)
appropriative_basin_user_matrix_df.sort_index(axis = "columns", inplace = True)
appropriative_basin_user_matrix = appropriative_basin_user_matrix_df.to_numpy()

# user connectivity matrices
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
downstream_connectivity_matrix = downstream_connectivity_df.to_numpy()
upstream_connectivity_matrix = downstream_connectivity_matrix.T

# need another copy of upstream connectivity matrix for modification in "Calculation of Net Available Flow Subroutine" below
upstream_connectivity_matrix_mod = downstream_connectivity_matrix.T

# list of basins
basins = list(downstream_connectivity_df.index)

########################### CREATE EMPTY OUTPUT DATAFRAMES #################################
# create columns using data_range
output_cols = data_range["Dates"].unique().tolist()
# Riparian output files
rip_basin_proportions_output = pd.DataFrame(columns=[output_cols], index=basins)
# Appropriative output files
app_user_allocations_output = pd.DataFrame(columns=[output_cols], index= app_users)
###########################################################################################


################################    START OF RIPARIAN LP     ##############################
###########################################################################################
###########################################################################################


for c, day in enumerate(data_range["Dates"].unique()):
    # print(c, day)
    riparian_demand_data = rip_demand_df[day].to_numpy()
    net_flow = flow_table_df[day].to_numpy()
    
    # AVAILABLE FLOW: 
    # k is used for basins, i is used for users 
    # available flow: initialize and populate a 1 x k list for available basin flow
    # available flow in basin k is sum of all upstream basin inflows less environmental requirement
    # matrix / vector operations:
    # upstream connectivity matrix * net flow
    # (k x k) * (k x 1) = k x 1 list of available net basin flows
    available_flow_data = np.matmul(upstream_connectivity_matrix, net_flow.T) 

    # DOWNSTREAM PENALTY
    # number of basins upstream of k divided by total basins
    # matrix / vector operations:
    # row sum of the k x k downstream connectivity matrix / count of  k x 1 list of basins
    
    # OLD
    # downstream_penalty_list = (np.divide(np.sum(riparian_user_connectivity_matrix, 1), np.count_nonzero(rip_users)))
    # NEW
    downstream_penalty_list = (np.divide(np.sum(upstream_connectivity_matrix, 1), np.count_nonzero(basins)))
    
    # BASIN RIPARIAN DEMAND + dictionary 
    basin_demand = {basins[k] : (np.matmul(riparian_basin_user_matrix, riparian_demand_data)[k]) for k, basin in enumerate(basins)}
    # Dataframe same as above
    basin_demand_df = pd.DataFrame.from_dict(basin_demand, orient = "index")
    basin_demand_df.columns = ["riparian_demand"]
    # matrix / vector operations:
    # !! incomplete
        
    # UPSTREAM BASIN DEMAND
    # basin-wide demand is the sum of user demand upstream of each basin
    # matrix / vector operations:
    # 1 x i list of user demand ∙ i x k user connectivity matrix  = 1 x k basin demand matrix
    basin_rip_demand_data_T = np.matmul(riparian_demand_data, riparian_user_connectivity_matrix_T)
    
    # ALPHA - Not currently used
    # minimum of the ratios of downstream penalties to basin demands, element by element division, division by zero should return 0
    # alpha = min(np.divide(downstream_penalty_list, basin_rip_demand_data_T, out = np.full_like(downstream_penalty_list, 999999999), where=basin_rip_demand_data_T!=0))
    
    # DICTIONARIES FOR CONSTRAINTS
    available_flow = {basins[k] : available_flow_data[k] for k, basin in enumerate(basins)}
    downstream_penalty = {basins[k] : downstream_penalty_list[k] for k, basin in enumerate(basins)}
    
    # DEFINE PROBLEM
    Riparian_LP = pulp.LpProblem("RiparianAllocation", pulp.LpMaximize)
    
    # DEFINE DECISION VARIABLES
    basin_proportions = pulp.LpVariable.dicts("Proportions", basins, 0, 1, cat="Continuous")

    # convert dictionary of decision variables to an array
    basin_proportions_list = pd.Series(basin_proportions).to_numpy()
    
    # USER ALLOCATION
    # user allocation i is their basin's allocation * user i's demand
    # need a 1 x k array of basin proportions ∙ k x i basin user matrix * demand (element-wise) 
    # matrix / vector operations:
    user_allocation_list = np.multiply((np.matmul(basin_proportions_list.T, riparian_basin_user_matrix)), riparian_demand_data)
    
    # user allocation dictionary
    user_allocation = {rip_users[i] : user_allocation_list[i] for i, user in enumerate(rip_users)}
    
    # UPSTREAM ALLOCATION: 
    # Sum of user allocations upstream of user i
    # matrix / vector operations:
    # need k x i upstream user matrix ∙ i by 1 user allocation matrix = k x 1 upstream basin allocation
    upstream_allocation_list = np.matmul(riparian_user_connectivity_matrix, user_allocation_list)
    
    # upstream allocation dictionary
    upstream_allocation = {basins[k] : upstream_allocation_list[k] for k, basin in enumerate(basins)}
    
    # OBJECTIVE FUNCTION
    # OLD
    # Riparian_LP += alpha * pulp.lpSum([basin_proportions[k]*downstream_penalty[k] for k in basins]) - pulp.lpSum([user_allocation[i] for i in rip_users])
    # NEW
    Riparian_LP += pulp.lpSum([user_allocation[i] for i in rip_users]) - pulp.lpSum([basin_proportions[k]*downstream_penalty[k]*basin_demand[k] for k in basins])  
    
    # CONSTRAINTS
    # mass balance
    for k in basins:
        Riparian_LP += pulp.lpSum([upstream_allocation[k]]) <= available_flow[k]
    
    # upstream basin's proportion cannot exceed any downstream basins
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
    
    # this loop turns LP variable output into a list
    basin_allocation = []
    for k, basin in enumerate(basins):
        basin_allocation.append(basin_proportions[basin].value() * basin_demand[basin])
    print("Basin Total Allocations", basin_allocation)
    # populate output table        
    for k in basins:
        rip_basin_proportions_output.loc[k, [day]] = basin_proportions[k].varValue


################################    END OF RIPARIAN LP     ##############################
#########################################################################################
#########################################################################################

#%%
#########################  CALCULATION OF NET AVAILABLE FLOW #####################
    # collect some riparian output
    dates = data_range["Dates"]
    rip_demand_matrix = np.array(rip_demand_df[dates])
    basin_proportion_matrix = np.array(rip_basin_proportions_output[dates])
    
    # riparian user allocations (basin proportion * user demand)
    rip_user_allocations_output = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix))*rip_demand_matrix),columns=[dates], index=rip_users)
    rip_user_allocations_output.index.name = "USER"    
    rip_user_allocations_matrix = np.array(rip_user_allocations_output[dates])
        
    # aggregate basin riparian allocations
    rip_basin_allocations = np.matmul(riparian_basin_user_matrix, rip_user_allocations_matrix)
    rip_basin_allocations_output = pd.DataFrame(rip_basin_allocations, columns=[dates], index=basins)
    rip_basin_allocations_output.index.name = "BASIN"
    
    # accumulated flow not accounting for riparian allocations
    available_flow_df = pd.DataFrame(available_flow_data, index = basins)
    available_flow_df.columns= ["cumulative_flow"]
    # individual basin inflow 
    available_flow_df["basin_flow"] = pd.DataFrame(net_flow, index = basins)     
    # aggregate basin riparian allocations from above
    available_flow_df["basin_allocations"] = rip_basin_allocations_output
    ### upstream riparian allocations 
    available_flow_df["rip_upstream_allocations"] =  np.matmul(riparian_user_connectivity_matrix, rip_user_allocations_matrix)
        
    available_flow_df["riparian_demand"] = basin_demand_df["riparian_demand"]
    # NEED TO REMOVE BASINS WITH RIPARIAN SHORTAGE FOR CALCULATIONS BELOW
    # calculate riparian shortage Y/N (0 = shortage, 1 = not)
    available_flow_df["short_rip_basins"] = basin_proportion_matrix
    for k, proportion in enumerate(basin_proportion_matrix):
        if proportion < 1 and available_flow_df["riparian_demand"][k]> 0:
            available_flow_df["short_rip_basins"][k] = 0
        else:
            available_flow_df["short_rip_basins"][k] = 1
       
    # net available flow:       
    # (cumulative_flow without in basin flow where there is riparian shortage)
    # minus
    # (rip_upstream_allocations - allocations where there is riparian shortage)
    # replace allocated proportions with rip shortage Y/N
    basin_proportion_matrix_new = np.array(pd.DataFrame(available_flow_df["short_rip_basins"]) )
    
    # recalculate available flow matrix without shorted basin flows 
    available_flow_df["new_cumulative_flow"] = np.matmul(upstream_connectivity_matrix, np.array(available_flow_df["basin_flow"]*available_flow_df["short_rip_basins"])) 
    # recalculate user allocations without shorted basin users:
    rip_user_allocations_output_new = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix_new))*rip_demand_matrix),columns=[dates], index=rip_users)
    rip_user_allocations_matrix_new = np.array(rip_user_allocations_output_new)
    available_flow_df["new_upstream_allocations"] =  np.matmul(riparian_user_connectivity_matrix, rip_user_allocations_matrix_new)
    # availabe flow for appropriatives is new cumulative flows less new upstream allocations   
    available_flow_df["available_app_flow"] = available_flow_df["new_cumulative_flow"] - available_flow_df["new_upstream_allocations"]
    
    # output if desired   
    available_flow_df.to_csv("output/available_appropriative_flow.csv")
#%%

#################################       APPROPRIATIVE_LP   ##############################
#########################################################################################
#########################################################################################
#########################################################################################
    
    appropriative_demand_data = app_demand_df[day].to_numpy()
    priority = app_demand_df["PRIORITY"].to_numpy()
    # shortage pentalty (inverse of priority)      
    shortage_penalty_data = (np.array([(1000*(1/(priority[i]))) for i, user in enumerate(app_users)])    )
 
    # DICTIONARIES
    app_demand = {app_users[i] : appropriative_demand_data[i] for i, user in enumerate(app_users)}
    shortage_penalty = {app_users[i] : shortage_penalty_data[i] for i, user in enumerate(app_users)}
    app_available_flow = {basins[k] : available_flow_df["available_app_flow"][k] for k, basin in enumerate(basins)}

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

    upstream_basin_allocation = np.matmul(appropriative_user_connectivity_matrix, user_allocation_list)
    # dictionary
    upstream_dict = {basins[k] : upstream_basin_allocation[k] for k, basin in enumerate(basins)}
    
    # CONSTRAINTS:
    # 1.  allocation is <= available flow;
    for basin in basins:
        Appropriative_LP += pulp.lpSum(upstream_dict[basin]) <= app_available_flow[basin]
        
    # 2.  allocation is <= to reported demand
    for user in app_users:
        Appropriative_LP += pulp.lpSum(user_allocation[user]) <= (app_demand[user])

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

#%%
################################    END OF APPROPRIATIVE LP  ########################
#####################################################################################
#####################################################################################


#################################  WRITING OUTPUT TO .CSV FILES #########################
#########################################################################################
#########################################################################################
#########################################################################################

# basin output
basin_output_df = available_flow_df[['cumulative_flow', 'basin_flow']]
basin_output_df["riparian_proportions"] = rip_basin_proportions_output
basin_output_df["rip_basin_allocations"] = available_flow_df["basin_allocations"] 
basin_output_df["rip_basin_demand"] = available_flow_df["riparian_demand"]
basin_output_df["rip_basin_shortage"] = basin_output_df["rip_basin_demand"] - basin_output_df["rip_basin_allocations"] 
basin_output_df["rip_basin_shortage_%"] = basin_output_df["rip_basin_shortage"] / basin_output_df["rip_basin_demand"]
basin_output_df["appropriative_output"] = "*"
basin_output_df["appropriative_available_flow"] = available_flow_df["available_app_flow"]
basin_output_df["app_basin_allocations"] = app_basin_allocations
app_demand_matrix = app_demand_df[dates].to_numpy()
basin_output_df["app_basin_demand"] = np.matmul(appropriative_basin_user_matrix , app_demand_matrix)
basin_output_df["app_basin_shortage"] = basin_output_df["app_basin_demand"] - basin_output_df["app_basin_allocations"] 
basin_output_df["app_basin_shortage_%"] = basin_output_df["app_basin_shortage"] / basin_output_df["app_basin_demand"]
basin_output_df = basin_output_df.add_suffix("_" + day)
basin_output_df.to_csv("output/basin_output.csv")

# # user output
# ## Appropriative

def five_sig_figs(x):
    output =  f"{x:.5f}"
    return output

user_output_df = app_demand_df[["BASIN", "PRIORITY"]]
user_output_df["app_allocations"] = app_user_allocations_output[dates]
user_output_df["app_demand"] = app_demand_df[dates]
user_output_df["app_shortage"] = (user_output_df["app_demand"] - user_output_df["app_allocations"])
user_output_df["app_shortage_%"] = (user_output_df["app_shortage"] / user_output_df["app_demand"])*100
# change values less than 0.00099 to 0 and round to 5 significant figures
user_output_df["app_shortage"][user_output_df["app_shortage"] < 0.00099] = 0
user_output_df["app_shortage_%"][user_output_df["app_shortage_%"] < 0.00099] = 0
user_output_df["app_shortage"] = user_output_df["app_shortage"].apply(five_sig_figs)
user_output_df["app_shortage_%"] = user_output_df["app_shortage_%"].apply(five_sig_figs)
user_output_df = user_output_df.add_suffix("_" + day)
user_output_df.to_csv("output/appropriative_user_output.csv")

## Riparian
del (rip_user_allocations_output)
rip_demand_matrix = np.array(rip_demand_df[dates])           
basin_proportion_matrix = np.array(rip_basin_proportions_output[dates])
rip_user_allocations_output = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix))*rip_demand_matrix), index=rip_demand_df.index)
rip_user_allocations_output.columns = dates.values
rip_user_allocations_output.index

rip_user_output_df = rip_demand_df[["BASIN", "PRIORITY"]]
rip_user_output_df["rip_allocations"] = rip_user_allocations_output[dates]
rip_user_output_df["rip_demand"] = rip_demand_df[dates]
rip_user_output_df["rip_shortage"] = rip_user_output_df["rip_demand"] - rip_user_output_df["rip_allocations"]
rip_user_output_df["rip_shortage_%"] = rip_user_output_df["rip_shortage"] / rip_user_output_df["rip_demand"]
# change values less than 0.00099 to 0 and round to 5 significant figures
rip_user_output_df["rip_shortage"][rip_user_output_df["rip_shortage"] < 0.00099] = 0
rip_user_output_df["rip_shortage_%"][rip_user_output_df["rip_shortage_%"] < 0.00099] = 0
rip_user_output_df["rip_shortage"] = rip_user_output_df["rip_shortage"].apply(five_sig_figs)
rip_user_output_df["rip_shortage_%"] = rip_user_output_df["rip_shortage_%"].apply(five_sig_figs)
rip_user_output_df = rip_user_output_df.add_suffix("_" + day)
rip_user_output_df.to_csv("output/riparian_user_output.csv")

#################################  WRITING OUTPUT FOR PORTAL TOOL########################
#########################################################################################
#########################################################################################
#########################################################################################

# basin output for portal tool and map
# new_df.columns
basin_output_df = available_flow_df[["cumulative_flow", "new_cumulative_flow", "basin_flow"]]
basin_output_df["sep_riparian_proportions"] = rip_basin_proportions_output
basin_output_df["sep_rip_basin_allocations"] = available_flow_df["basin_allocations"] 
basin_output_df["sep_rip_basin_demand"] = available_flow_df["riparian_demand"]
basin_output_df["sep_rip_basin_shortage"] = basin_output_df["sep_rip_basin_demand"] - basin_output_df["sep_rip_basin_allocations"] 
basin_output_df["sep_rip_basin_shortage_%"] = basin_output_df["sep_rip_basin_shortage"] / basin_output_df["sep_rip_basin_demand"]

basin_output_df["sep_app_basin_allocations"] = app_basin_allocations
app_demand_matrix = app_demand_df[dates].to_numpy()
basin_output_df["sep_app_basin_demand"] = np.matmul(appropriative_basin_user_matrix , app_demand_matrix)
basin_output_df["sep_app_basin_shortage"] = basin_output_df["sep_app_basin_demand"] - basin_output_df["sep_app_basin_allocations"] 
basin_output_df["sep_app_basin_shortage_%"] = basin_output_df["sep_app_basin_shortage"] / basin_output_df["sep_app_basin_demand"]

basin_output_df.columns = ['available_flow_data', 'new_cumulative_flow', "net_flow",
       'sep_riparian_proportions', 'sep_rip_basin_allocations',
       'sep_rip_basin_demand', 'sep_rip_basin_shortage',
       'sep_rip_basin_shortage_%', 'sep_app_basin_allocations',
       'sep_app_basin_demand', 'sep_app_basin_shortage',
       'sep_app_basin_shortage_%']

basin_output_df.to_csv("output/basin_output_PORTAL.csv")

# # user output
# ## Appropriative
user_output_df = app_demand_df[["BASIN", "RIPARIAN", "PRIORITY"]]
user_output_df["2021-09_ALLOCATION"] = app_user_allocations_output[dates]
user_output_df["2021-09_DEMAND"] = app_demand_df[dates]
user_output_df["2021-09_SHORTAGE"] = user_output_df["2021-09_DEMAND"] - user_output_df["2021-09_ALLOCATION"]
user_output_df["2108-09_%_SHORTAGE"] = user_output_df["2021-09_SHORTAGE"] / user_output_df["2021-09_DEMAND"]


## Riparian
del (rip_user_allocations_output)
rip_demand_matrix = np.array(rip_demand_df[dates])           
basin_proportion_matrix = np.array(rip_basin_proportions_output[dates])
rip_user_allocations_output = pd.DataFrame(((np.matmul(riparian_basin_user_matrix.transpose(), basin_proportion_matrix))*rip_demand_matrix), index=rip_demand_df.index)
rip_user_allocations_output.columns = dates.values
rip_user_allocations_output.index


rip_user_output_df = rip_demand_df[["BASIN", "RIPARIAN", "PRIORITY"]]
rip_user_output_df["2021-09_ALLOCATION"] = rip_user_allocations_output[dates]
rip_user_output_df["2021-09_DEMAND"] = rip_demand_df[dates]
rip_user_output_df["2021-09_SHORTAGE"] = rip_user_output_df["2021-09_DEMAND"] - rip_user_output_df["2021-09_ALLOCATION"]
rip_user_output_df["2108-09_%_SHORTAGE"] = rip_user_output_df["2021-09_SHORTAGE"] / rip_user_output_df["2021-09_DEMAND"]


combined_user_output_df = pd.concat([user_output_df,rip_user_output_df], axis = 0)

combined_user_output_df.to_csv("output/combined_user_output_PORTAL.csv")





