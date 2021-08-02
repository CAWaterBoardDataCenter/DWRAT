# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:33:49 2020

@author: dp
"""
import pandas as pd
import numpy as np
import os

# Daily Data:
# define a function date_string and provide a start and end date in format ("mm/dd/yyy" , "mm/dd/yyy" )
def convert_date_standard_string(datelike_object):
    """
    Return string version of date in format mm/dd/yyyy
    Parameters
    -----------
    datelike_object
        A value of type date, datetime, or Timestamp.
        (e.g., Python datetime.datetime, datetime.date,
        Pandas Timestamp)
    """
    return "{:%#m/%#d/%Y}".format(datelike_object)


def convert_date_standard_string_2(datelike_object):
    """
    Return string version of date in format yyyy-mm-dd
    Parameters
    -----------
    datelike_object
        A value of type date, datetime, or Timestamp.
        (e.g., Python datetime.datetime, datetime.date,
        Pandas Timestamp)
    """
    return "{:%Y-%#m-%#d}".format(datelike_object)

def convert_date_standard_string_monthly(datelike_object):
    """
    Return string version of date in format yyyy-mm
    Parameters
    -----------
    datelike_object
        A value of type date, datetime, or Timestamp.
        (e.g., Python datetime.datetime, datetime.date,
        Pandas Timestamp)
    """
    return "{:%Y-%m}".format(datelike_object)

    
def date_string(first, last):
    dates_list = pd.date_range(start=first, end=last)
    dates_df = pd.DataFrame(dates_list, columns=["yyyy-mm-dd"])
    dates_df["m/d/yyyy"] = dates_df["yyyy-mm-dd"].apply(convert_date_standard_string)
    dates_df["yyyy-m-d"] = dates_df["yyyy-mm-dd"].apply(convert_date_standard_string_2)
    dates_df["yyyy-mm"]  = dates_df["yyyy-mm-dd"].apply(convert_date_standard_string_monthly)
    return dates_df




# Leave this commented out after using the script
# assign the results of the function to a variable
    
output = date_string("10/01/2015", "9/30/2016")
output


# output.to_csv("input/data_range.csv")



