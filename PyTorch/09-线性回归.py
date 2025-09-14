#导入相关模块
import nodel
import torch
from scipy.ndimage import label
from torch.utils.data import TensorDataset #构造数据集对象
from torch.utils.data import DataLoader  #数据加载器
#from torch import nn #nn函数中有平方损失函数和假设函数
from torch.nn import Linear,MSELoss
#from torch import optim  #optim模块中有优化器函数
from torch.optim  import SGD
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

#构建数据集
def create_dataset():
    x,y,coef=make_regression(n_samples=100, n_features=1,noise = 10,coef=True, bias=1.5,random_state=0)
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x,y,coef
x,y,coef = create_dataset()
# plt.scatter(x,y)
# x1 = torch.linspace(x.min(),x.max(),1000)  #设置直线x的值
# y1 = torch.tensor([v*coef+1.5 for v in x1])  #设置要生成直线的y的值于x对应
# plt.plot(x1,y1)
# plt.grid()
# plt.show()




dataset = TensorDataset(x,y)                ##构造数据集对象
dataloader = DataLoader(dataset=dataset,batch_size=16,shuffle=True)      #数据加载器
model = Linear(in_features=1,out_features=1)                 #构建模型,指定特征向量和目标值的维度都为1
#损失和优化器
loss = MSELoss()    #损失
optimizer = SGD(params=model.parameters(),lr=0.01)

epochs = 100  #训练一百次
#损失变化
loss_epoch = []   #用列表来记录每次训练的损失
total_loss = 0.0   #用来记录总的损失值
train_sample = 0.0   #用来记录训练的样本个数
for _ in range(epochs):       #遍历每个轮次
    for train_x,train_y in dataloader:       #在每个轮次中迭代每个batch,线获取每个batch的特征数据和目标数据
      y_pred = model(train_x.type(torch.float32))            #把特征数据送入目标数据,(类型要转换为torch.float32,在模型中传输必须用这个),获取预测结果
      loss_values = loss(y_pred,train_y.reshape(-1,1).type(torch.float32))            #送入预测值y_pred和真实值train_y,reshape(-1,1)对于所有的样本每个样本是一个维度的目标值,获取最终loss值
      #拿到loss值后进行反向传播更新参数
      total_loss+=loss_values.item()         #记录总损失
      train_sample +=len(train_y)       #记录样本个数
      optimizer.zero_grad()      #梯度清零
      loss_values.backward()       #自动计算梯度
      optimizer.step()          # 用计算出的梯度更新参数
    loss_epoch.append(total_loss/train_sample)   #获取当前轮次的损失

plt.plot(range(epochs),loss_epoch)     #展示损失函数
plt.show()
plt.scatter(x,y)
x1 = torch.linspace(x.min(),x.max(),1000)  #设置直线x的值,展示拟合结果
y0 = torch.tensor([v * model.weight + model.bias for v in x1])           #拟合结果
y1 = torch.tensor([v*coef+1.5 for v in x1])  #设置要生成直线的y的值于x对应
plt.plot(x1,y0,label='pred')
plt.plot(x1,y1,label='real')
plt.legend()
plt.grid()
plt.show()