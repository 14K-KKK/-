import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib as plt

All_1_Vec = np.ones(17,dtype=np.int)
df = pd.read_csv('data.csv',names = ['NUM','Density','SugRate','isGood'],index_col = ['NUM'])
x = np.array([list(df['Density']),list(df['SugRate']),All_1_Vec])
y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
print(All_1_Vec)
print(x)
print(y)
beta = np.array([[0],[0],[1]])
old_l = 0
n = 0

while 1:
    beta_T_x = np.dot(beta.T[0],x)
    cur_l = 0
    for i in range(17):
        cur_l = cur_l + (-y[i]*beta_T_x[i]+np.log(1+np.exp(beta_T_x[i])))

    if np.abs(cur_l-old_l) <= 0.0000001:
        break
    elif n > 1000:
        break

    n = n + 1
    old_l = cur_l
    db = 0
    d2b = 0
    for i in range(17):
        db = db - np.dot(np.array([x[:,i]]).T,(y[i]-(np.exp(beta_T_x[i])/(1+np.exp(beta_T_x[i])))))
        d2b = d2b + np.dot(np.array([x[:,i]]).T,np.array([x[:,i]])*(np.exp(beta_T_x[i])/(1+np.exp(beta_T_x[i])))*(1-(np.exp(beta_T_x[i])/(1+np.exp(beta_T_x[i])))))
    beta = beta - np.dot(linalg.inv(d2b),db)
    print('第',n,'次迭代','模型参数：', beta)

print('模型参数：',beta)
print('迭代次数：',n)


