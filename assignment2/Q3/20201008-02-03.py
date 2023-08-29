import os
import math as mt

def csv_to_input(file_name = "asst-2-Q3.txt"):
    with open(os.path.dirname(__file__)+"/"+file_name,'r') as f:
        lines = f.readlines()
        t =  int(lines[0])
        inputs_lst = []
        for line in lines[1:]:
            str_Lst = line.split() 
            L = [float(x) for x in str_Lst]
            scaler = L[-1]
            vec = L[:-1]
            inputs_lst.append((vec,scaler))
    return inputs_lst

def MSE(v1:list ,v2:list):
    S = 0
    if len(v1) != len(v2):
        raise Exception("Length of vectors must be equal")
    for x,y in zip(v1,v2):
        S += (x-y)**2
    
    return round(S/len(v1),4)

def wi(p,q):
    return p + 0.05*q

def dot(v:list,w:list):
    S = 0
    for x,y in zip(v,w):
        S += x*y
    return S

def to_base_b(i,b,n):
    base_b = []
    while i >= b:
        r = i % b
        i = i//b
        base_b.append(r)
    base_b.append(i)
    base_b.extend([0]*(n-len(base_b)))
    return base_b[::-1]

def i_to_w(i,wi_lst,n):
    b = len(wi_lst)
    b_L = to_base_b(i,b,n)
    # print(b_L)
    w_pq = [wi_lst[x] for x in b_L]
    # print(w_pq)
    w = [wi(p,q) for p,q in w_pq]
    return w, w_pq

def w_closest(vecs,scalers):
    n = len(vec)
    wi_lst = [(p,q) for p in range(3) for q in range(10)]
    b = len(wi_lst)
    w_dict = {}
    for i in range(b**n):
        w = i_to_w(i,wi_lst,n)[0]
        # print(vecs,'\n \n \n',scalers)
        w_scalers = [dot(v,w) for v in vecs]
        w_dict[i] = MSE(scalers,w_scalers)
    L = list(w_dict.items())
    L.sort(key = lambda x:x[1])
    i = L[0][0]
    w, w_pq = i_to_w(i,wi_lst,n)
    return w, w_pq, w_dict[i]
    

inputs_lst = csv_to_input()
vecs = []
scalers = []
for line in inputs_lst:
    vec, scaler = line
    vecs.append(vec)
    scalers.append(scaler)
w, w_pq, w_mse = w_closest(vecs,scalers)
print(20201008,w_mse)

# print(to_base_b(5,4))

# w_scaler = [dot(v,w) for v in vecs]

# print(w_scaler,scalers)
# print(MSE(w_scaler,scalers))