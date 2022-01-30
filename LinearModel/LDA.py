import numpy as np
import pandas as pd
from numpy import linalg
from matplotlib import pyplot as plt

df = pd.read_csv('data.csv',names = ['NUM','Density','SugRate','isGood'])
#数据处理
data_good = df.loc[df['isGood']==1]
data_bad = df.loc[df['isGood']==0]
data1 = np.array(data_good[['Density','SugRate']])
data0 = np.array(data_bad[['Density','SugRate']])
#求均值向量
u1 = np.array(data_good.mean())
u11 = np.array([np.delete(u1,[0,3])]).T
u0 = np.array(data_bad.mean())
u00 = np.array([np.delete(u0,[0,3])]).T
#求类间方差矩阵Sb
Sb = np.dot(u00-u11,(u00-u11).T)
#求类内方差矩阵Sw
sigma0 = np.matrix([[0,0],[0,0]])
sigma1 = np.matrix([[0,0],[0,0]])
for i in range(9):
    sigma0 = sigma0 + np.dot(np.array([data0[i]]).T-u00, (np.array([data0[i]]).T-u00).T)
for i in range(8):
    sigma1 = sigma1 + np.dot(np.array([data1[i]]).T-u11, (np.array([data1[i]]).T-u11).T)
Sw = np.matrix(sigma0 + sigma1)
print('sigma0=',sigma0)
print('sigma1=',sigma1)
print('Sb=',Sb)
print('Sw=',Sw)
#U,S,V = linalg.svd(Sw)
Sw_inv = linalg.inv(Sw)
#求Omega
Omega = Sw_inv*(u00-u11)
print('Omega=',Omega)
#绘图
for i in range(17):
    if df['isGood'][i] > 0:
        plt.plot(df['Density'][i], df['SugRate'][i], '+g')
    else:
        plt.plot(df['Density'][i], df['SugRate'][i], 'or')

px = [0,1]
py = [-Omega[0,0]*px[0]/Omega[1,0],-Omega[0,0]*px[1]/Omega[1,0]]
plt.plot(px,py)

plt.show()
