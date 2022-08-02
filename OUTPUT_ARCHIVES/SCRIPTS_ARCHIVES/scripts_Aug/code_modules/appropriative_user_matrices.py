# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:46:14 2020

@author: dp
"""

# Appropriative User Matrix (location)

def appropriative_user_matrices_main():
    import pandas as pd
    import numpy
    
    flow_table_df = pd.read_csv('input/flows.csv', index_col= "BASIN")
    flow_table_df.sort_index(axis = "index", inplace = True)
    basins = flow_table_df.index.values
    
    app_user_df = pd.read_csv('input/appropriative_demand.csv', index_col = "USER")
    app_user_df.sort_index(axis = "index", inplace = True)
    app_user =  app_user_df.index.values
    user_location = app_user_df["BASIN"].to_numpy()
    
    basin_use = {app_user[i] : user_location[i] for i, user in enumerate(app_user)}  
    index_dictionary = {basins[k] : [k] for k, basin in enumerate(basins)}
        
    user_matrix = numpy.zeros([numpy.size(app_user), numpy.size(basins)], dtype = int)
    
    for i, user in enumerate(app_user):
        user_matrix[i][index_dictionary[basin_use[user]]] = 1
    
    user_matrix = user_matrix.T
    
    appropriative_user_matrix = pd.DataFrame(user_matrix, index = basins)
    appropriative_user_matrix.index.name = "BASIN"
    appropriative_user_matrix.columns = app_user
    
    appropriative_user_matrix.to_csv("input/appropriative_user_matrix.csv", index = True)
    
    ##########################################################################################################################
    
    # User connectivity matrix (1 if user is upstream of basin)
    
    basin_connectivity_matrix_df = pd.read_csv("input/basin_connectivity_matrix.csv", index_col = "BASIN")
    basin_connectivity_matrix_df.sort_index(axis = "index", inplace = True)
    basin_connectivity_matrix_df.sort_index(axis = "columns", inplace = True)
    basin_connectivity_matrix = basin_connectivity_matrix_df.to_numpy()
    
    user_connectivity = numpy.matmul(basin_connectivity_matrix.T, user_matrix)
    user_connectivity = pd.DataFrame(user_connectivity, index = basins, columns = app_user)
    user_connectivity.index.name = "BASIN"
    
    user_connectivity.to_csv("input/appropriative_user_connectivity_matrix.csv", index = True)

if __name__ == "__main__":
    appropriative_user_matrices_main()


