import os

def csv_to_vectors(file_name = "asst-2-Q1.txt"):
    with open(os.path.dirname(__file__)+"/"+file_name,'r') as f:
        lines = f.readlines()
        t =  int(lines[0])
        vec_lst = []
        for line in lines[1:]:
            str_Lst = line.split() 
            L = [float(x) for x in str_Lst]
            if len(L)%2 != 0:
                raise Exception("The line must have even no of values")
            n = len(L)
            x = L[:int(n/2)]
            y = L[int(n/2):]
            vec_lst.append((x,y))
    return vec_lst


    

def MSE(v1:list ,v2:list):
    S = 0
    if len(v1) != len(v2):
        raise Exception("Length of vectors must be equal")
    for x,y in zip(v1,v2):
        S += (x-y)**2
    
    return round(S/len(v1),4)

vec_lst = csv_to_vectors()
for v1,v2 in vec_lst:
    mse = MSE(v1,v2)
    print("20201008",mse)