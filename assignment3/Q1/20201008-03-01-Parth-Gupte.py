import os
import pandas as pd
import numpy as np

def read_conditions(file_name = "asst-3-Q1.txt"):
    with open(os.path.dirname(__file__)+"/"+file_name,'r') as f:
        lines = f.readlines()
        cond_lst = []
        m , n = [int(x) for x in lines[0].split()]
        for line in lines[1:]:
            str_lst = line.split()
            cond = [str_lst[0].lower(),float(str_lst[1]),float(str_lst[2])]
            cond_lst.append(cond)
    
    return cond_lst

def read_data(file_name = "asst-3-Q1.xlsx"):
    f = open(os.path.dirname(__file__)+"/"+file_name,'r')
    data = pd.read_excel(os.path.dirname(__file__)+"/"+file_name,header=None)
    data.dropna(inplace=True)
    return data

def fix_row(row,cond):
    #checking range
    l, u = cond[1:]
    for x in row:
        if x<l or x>u:
            break
        elif cond[0] == 'int':
            if int(x) != x:
                break
        elif cond[0] == 'float':
            if float(x) != x:
                break
    else:
        if cond[0] == 'int':
            return [int(x) for x in row]
        elif cond[0] == 'float':
            return [float(x) for x in row]
        
    return [np.NaN]*len(row)
    
def fix_data(data:pd.DataFrame):
    '''
    Input a dataframe and a cleaned dataframe is returned
    '''
    for i in range(len(data)):
        row = list(data.loc[i])
        new_row = fix_row(row,cond_lst[i])
        data.loc[i] = new_row
    data.dropna(inplace= True)
    return data


cond_lst = read_conditions()
data = read_data()
fix_data(data)
print("20201008", len(data))