import numpy as np
import pandas as pd
import os

def read_data(file_name = "asst-3-Q2.xlsx"):
    f = open(os.path.dirname(__file__)+"/"+file_name,'r')
    data = pd.read_excel(os.path.dirname(__file__)+"/"+file_name,header=None)
    data.dropna(inplace=True)
    return data.to_numpy()

def decrypt_martix(X:np.ndarray,l:int):
    XTX = np.matmul(X.transpose(),X)
    eigvals, eigvecs = np.linalg.eig(XTX)
    idx_L = list(range(len(eigvals)))
    l_large_eig_idxs = sorted(idx_L,key = lambda x: idx_L[x])[:l]
    D = (eigvecs.transpose()[l_large_eig_idxs]).transpose()
    return D
    
def frobenius(M:np.ndarray):
    S = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            S += (M[i,j])**2
    S = (S**(1/2))/(M.shape[0]*M.shape[1])
    return S

def encrypt(X:np.ndarray,D:np.ndarray):
    E = D.transpose()
    # print(E.shape,X.shape)
    return np.matmul(E,X.transpose()).transpose()

def decrypt(B:np.ndarray,D:np.ndarray):
    return np.matmul(D,B.transpose()).transpose()


X = read_data()
D = decrypt_martix(X,2)
B = encrypt(X,D)
X_new = decrypt(B,D)

print(2021008,frobenius(X-X_new)/(X.shape[0]*X.shape[1]),frobenius(B))