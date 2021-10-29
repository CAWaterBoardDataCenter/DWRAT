# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:48:44 2020

@author: dpedroja
"""

# Script that constructs the basin connectivity matrix

import pandas as pd
import numpy

######################################### YOU NEED TO SPECIFY THE OUTLET HERE !!!!!!!!!!!!!!!!!!!!

outlet = "L_21_MSRR"

######################################### YOU NEED TO SPECIFY THE OUTLET ABOVE !!!!!!!!!!!!!!!!!!!!

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




# Script that constructs the next_downstream matrix
# initialize an empty k x k matrix
empty_matrix = numpy.zeros((numpy.size(basins), numpy.size(basins)), dtype = int)
# set value to 1 at the index value of the FLOWS_TO basin
for k, basin in enumerate(flow_table_df.index):
    empty_matrix[k][ index_dictionary[flow_table_df["FLOWS_TO"][k]]] = 1
# create datafram and write output
next_down_df = pd.DataFrame(empty_matrix, index = basins, columns = basins)
next_down_df.index.name = "BASIN"
next_down_df.to_csv("input/next_downstream.csv", index = True)


# Script that constructs the next_upstream matrix
# initialize an empty k x k matrix
empty_matrix_1 = numpy.zeros((numpy.size(basins), numpy.size(basins)), dtype = int)
# set value to 1 at the index value of the FLOWS_TO basin
# need a column of basins
flow_table_df["BASIN"] = flow_table_df.index
for k, basin in enumerate(flow_table_df["FLOWS_TO"]):
    for i in flow_table_df["BASIN"][flow_table_df["FLOWS_TO"]==basin]:
        empty_matrix_1[  index_dictionary[basin], index_dictionary[flow_table_df["BASIN"][flow_table_df["FLOWS_TO"]==basin][i]]] = 1
    
# create datafram and write output
next_up_df = pd.DataFrame(empty_matrix_1, index = basins, columns = basins)
next_up_df.index.name = "BASIN"
next_up_df.to_csv("input/next_upstream.csv", index = True)