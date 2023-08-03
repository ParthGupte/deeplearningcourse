import os
import warnings

def csv_to_strings(file_name = "asst-2-Q2.txt"):
    with open(os.path.dirname(__file__)+"/"+file_name,'r') as f:
        lines = f.readlines()
        t =  int(lines[0])
        strs_lst = []
        for line in lines[1:]:
            line_clean = line.strip('\n')
            strs_lst.append(line_clean)
    return strs_lst

def most_freq_pair(line:str):
    count_D = {}
    for i in range(len(line)-1):
        pair = line[i:i+2]
        if pair in count_D.keys():
            count_D[pair] += 1
        else: 
            count_D[pair] = 1
    L = list(count_D.items())
    L.sort(key= lambda x: x[1])
    if L[-1][1] == L[-2][1]:
        warnings.warn("There is more than one most frequent pair")
    return L[-1][0]

strs_lst = csv_to_strings()
for line in strs_lst:
    print(20201008,most_freq_pair(line))