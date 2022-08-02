# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:48:44 2020

@author: dpedroja
"""

# Script that constructs the basin connectivity matrix

if __name__ == "__main__":
#!!! SPECIFY THE OUTLET BASIN HERE !!!!!
    outlet = "R_13_MSRR"
# IF SCRIPT WON'T FINISH, IT'S BECAUSE THE OUTLET ABOVE IS INVALID !!!!!

def basin_connectivity_main(outlet):
    import pandas as pd
    import numpy

    flow_table_df = pd.read_csv('input/flows.csv', index_col= "BASIN")
    flow_table_df.sort_index(axis = "index", inplace = True)
    basins = flow_table_df.index.values
    flows_to = flow_table_df["FLOWS_TO"].to_numpy()
    
    # DICTIONARIES
    flows_to_dictionary = {basins[k] : flows_to[k] for k, basin in enumerate(basins)}
    index_dictionary = {basins[k] : [k] for k, basin in enumerate(basins)}
    
    # Initialize empty basin x basin identity matrix
    connectivity_matrix = numpy.identity(numpy.size(basins), dtype = int)
    
    for k, basin in enumerate(basins):
        while basin != outlet:
            connectivity_matrix[k][index_dictionary[flows_to_dictionary[basin]]] = 1
            basin = flows_to_dictionary[basin]
    
    cm_df = pd.DataFrame(connectivity_matrix, index = basins, columns = basins)
    cm_df.index.name = "BASIN"
    cm_df.to_csv("input/basin_connectivity_matrix.csv", index = True)
    return cm_df

if __name__ == '__main__':
    basin_connectivity_main()
