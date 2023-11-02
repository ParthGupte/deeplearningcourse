import numpy as np

clean_rows_lst = [0]*15

def read_data(file_name = "asst-4-data.txt"):
    with open(file_name,'r') as f:
        lines = f.readlines()
        return lines

def convert_row_to_list(row:str):
    flagged = False
    str_lst = row.split(",")
    new_lst = []
    for i in range(len(str_lst)):
        word = str_lst[i]
        word = word.strip()
        if word == '?':
            clean_rows_lst[i] -= 1
            flagged = True            
        if i in [0,2,4,10,11,12]:
            try:
                word = int(word)
            except:
                clean_rows_lst[i] -= 1
                flagged = True
        clean_rows_lst[i] += 1
        new_lst.append(word)
    if flagged:
        new_lst = []
    return new_lst

def count(row:list,counting_dict:dict,discrete_vars):
    for i in discrete_vars:
        word = row[i]
        if i not in counting_dict.keys():
            counting_dict[i] = [word]
        elif word not in counting_dict[i]:
            counting_dict[i].append(word)
            # print(counting_dict)

def clean_count(lines:list,discrete_vars = [1,3,5,6,7,8,9,13,14]):#clean and count
    new_lines = []
    counting_dict = {}
    # print(counting_dict)
    for row in lines:
        new_row = convert_row_to_list(row)
        if new_row == []:
            continue
        count(new_row,counting_dict,discrete_vars)
        new_lines.append(new_row)
    # print(counting_dict)
    return new_lines, counting_dict

def onehot(val:int,total:int):
    L = []
    for i in range(total):
        if i == val:
            L.append(1)
        else:
            L.append(0)
    return L

def convert_row(line:list,counting_dict:dict):#cleaned line to array
    new_line = []
    for i in range(len(line)):
        word = line[i]
        if i in counting_dict.keys():
            val = counting_dict[i].index(word)
            total = len(counting_dict[i])
            L = onehot(val,total)
            new_line.extend(L)
        else:
            new_line.append(word)     
    # print(new_line)       
    arr = np.array(new_line)
    return arr

            
def convert_cleaned_to_arr(lines:list,counting_dict:dict): #cleaned lines to array
    lst = []
    for row in lines:
        arr = convert_row(row,counting_dict)
        lst.append(arr)
    arr_data = np.array(lst)
    return arr_data


def convert(file_name = "asst-4-data.txt"):
    str_lines = read_data(file_name)
    clean_lines, counting_dict = clean_count(str_lines)
    arr_data = convert_cleaned_to_arr(clean_lines,counting_dict)
    return arr_data


arr_data = convert()
print(20201008,len(read_data()),clean_rows_lst[0],clean_rows_lst[1],clean_rows_lst[2],clean_rows_lst[3],clean_rows_lst[4],clean_rows_lst[5],clean_rows_lst[6],clean_rows_lst[7],clean_rows_lst[8],clean_rows_lst[9],clean_rows_lst[10],clean_rows_lst[11],clean_rows_lst[12],clean_rows_lst[13],clean_rows_lst[14])
