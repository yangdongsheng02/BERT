import torch
import torch.nn as nn #构建模型需要继承自nn.Model,导入神经网络模块，nn 是常用别名
from scipy.odr import Model
from torchsummary import summary             #summary获取网络的结构


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1 = nn.Linear(3,3)
        self.l2 = nn.Linear(3,2 )
        self.out = nn.Linear(2, 2)

    def forward(self, x):               #x为输入特征即输入数据,,forward方法定义前向传播,这里将输入数据送入网络结构
       x = torch.sigmoid(self.l1(x))
       x = torch.relu(self.l2(x))
       x = self.out(x)
       out = torch.softmax(x, dim=-1)
       return out
model = Model()
x = torch.randn(5,3)
out = model(x)
print(out.shape)

summary(model,input_size=(3,),batch_size=5)