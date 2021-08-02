# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:47:55 2020

@author: dp
"""
# Riparian User Matrix (location)
import pandas as pd
import numpy

flow_table_df = pd.read_csv('input/flows.csv', index_col= "BASIN")
flow_table_df.sort_index(axis = "index", inplace = True)
basins = flow_table_df.index.values

rip_user_df = pd.read_csv('input/riparian_demand.csv', index_col = "USER")
rip_user_df.sort_index(axis = "index", inplace = True)
rip_user =  rip_user_df.index.values
user_location = rip_user_df["BASIN"].to_numpy()

basin_use = {rip_user[i] : user_location[i] for i, user in enumerate(rip_user)}  
index_dictionary = {basins[k] : [k] for k, basin in enumerate(basins)}

user_matrix = numpy.zeros([numpy.size(rip_user), numpy.size(basins)], dtype = int)

for i, user in enumerate(rip_user):
    user_matrix[i][index_dictionary[basin_use[user]]] = 1

user_matrix = user_matrix.T

riparian_user_matrix = pd.DataFrame(user_matrix, index = basins)
riparian_user_matrix.index.name = "BASIN"
riparian_user_matrix.columns = rip_user

riparian_user_matrix.to_csv("input/riparian_user_matrix.csv", index = True)

##########################################################################################################################

# User connectivity matrix (1 if user is upstream of basin)
basin_connectivity_matrix_df = pd.read_csv("input/basin_connectivity_matrix.csv", index_col = "BASIN")
basin_connectivity_matrix_df.sort_index(axis = "index", inplace = True)
basin_connectivity_matrix_df.sort_index(axis = "columns", inplace = True)
basin_connectivity_matrix = basin_connectivity_matrix_df.to_numpy()

user_connectivity = numpy.matmul(basin_connectivity_matrix.T, user_matrix)

user_connectivity = pd.DataFrame(user_connectivity, index = basins, columns = rip_user)
user_connectivity.index.name = "BASIN"

user_connectivity.to_csv("input/riparian_user_connectivity_matrix.csv", index = True)


