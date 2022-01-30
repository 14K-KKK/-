import pandas as pd
import numpy as np

#定义sigmoid激活函数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#q=隐层神经元数，d=输入x的属性维数，l=输出y的维数，m=训练集样例数，yita=学习率，n=迭代次数
q = 10
d = 19
l = 1
m = 11
yita = 0.5
n = 0

#西瓜数据集3.0
dataset = pd.DataFrame(
    data=[
    # 1
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
    # 2
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
    # 3
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
    # 4
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
    # 5
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
    # 6
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
    # 7
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
    # 8
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],

    # ----------------------------------------------------
    # 9
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
    # 10
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
    # 11
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
    # 12
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
    # 13
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
    # 14
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
    # 15
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
    # 16
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
    # 17
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
                            ],
    columns = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率', '好坏'],
    index=range(17))

#数据集处理与参数赋初值
dataset['好坏'].loc[dataset['好坏'] == '好瓜'] = 1
dataset['好坏'].loc[dataset['好坏'] == '坏瓜'] = 0
data = pd.get_dummies(data = dataset, columns = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感'])
trainset = data.iloc[[0,1,3,4,6,8,9,10,11,12,13],:]
testset = data.iloc[[2,5,7,14,15,16],:]
#用于训练的输入输出
y = np.array(trainset['好坏']).reshape([m,l])
x = np.array(trainset.drop(labels='好坏', axis=1)).reshape([m,d])
error = np.array(np.zeros([1,m])).reshape([1,m])

w = np.random.rand(q,l)
v = np.random.rand(d,q)
threshold_y = np.random.rand(1,l)
threshold_b = np.random.rand(1,q)

while True:
    for i in range(m):
        #取m个输入样例x中的第i个
        xk = np.array(x[i]).reshape([1,d])
        #计算隐层神经元的输入alpha，输出b
        alpha = np.array(np.dot(x[i],v)).reshape([1,q])
        b = sigmoid(np.array([alpha-threshold_b],dtype=np.float64).reshape([1,q]))
        #计算输出层神经元的输入beta，输出y_out
        beta = np.array(np.dot(b,w)).reshape([1,l])
        y_out = sigmoid(np.array([beta-threshold_y],dtype=np.float64).reshape([1,l]))
        #计算误差error
        error[0,i] = 0.5*(np.array(y[i]).reshape(1,l)-y_out)*(np.array(y[i]).reshape(1,l)-y_out)
        #计算g，以及用于后续计算的g_extend
        g = np.array(y_out*(1-y_out)*(np.array(y[i]).reshape(1,l)-y_out)).reshape([1,l])
        g_extend = np.array(np.tile(g,(q,1))).reshape([q,l])
        #计算权向量和阈值的变化量
        delta_w = np.array(yita*np.tile(b.T,(1,l))*g_extend).reshape([q,l])
        delta_threshold_y = np.array(-yita*g).reshape([1,l])
        e = np.array(b*(1-b)*np.dot(w,g.T).T).reshape([1,q])
        delta_v = np.array(yita*np.tile(e,(d,1))*np.tile(xk.T,(1,q))).reshape(d,q)
        delta_threshold_b = np.array(-yita*e).reshape([1,q])
        #迭代赋值
        w = w + delta_w
        v = v + delta_v
        threshold_b = threshold_b + delta_threshold_b
        threshold_y = threshold_y + delta_threshold_y
        # print('delta_threshold_y=', delta_threshold_y)
        # print('threshold_y=', threshold_y)
        # print('beta=',beta)
        # print('y_out=', y_out)
        # print('y=',y[i])
    #计算数据集上的总误差
    total_error = np.dot(error,np.ones([m,1]))
    print('total_error=', total_error)
    n = n + 1
    if n > 5000:
        break
    elif total_error < 0.001:
        break

#验证
test_y = np.array(testset['好坏']).reshape([17-m,l])
test_x = np.array(testset.drop(labels='好坏', axis=1)).reshape([17-m,d])

correct_number = 0
for i in range(17-m):
    test_xk = np.array(test_x[i]).reshape([1,d])
    test_alpha = np.array(np.dot(test_x[i],v)).reshape([1,q])
    test_b = sigmoid(np.array([test_alpha-threshold_b],dtype=np.float64).reshape([1,q]))
    test_beta = np.array(np.dot(test_b,w)).reshape([1,l])
    test_y_out = sigmoid(np.array([test_beta-threshold_y],dtype=np.float64).reshape([1,l]))
    print('test_y=', test_y[i], 'test_y_out=', test_y_out)
    if ((test_y_out>0.5 and test_y[i]==1)or(test_y_out<0.5 and test_y[i]==0)):
        correct_number = correct_number + 1
#计算验证集正确率
correct_rate = correct_number/(17-m)

print('迭代次数=',n-1)
print('w=',w)
print('v=',v)
print('threshold_y=',threshold_y)
print('threshold_b=',threshold_b)
print('验证集正确率=',correct_rate)







