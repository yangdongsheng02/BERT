# 导入相关模块
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch.nn import Linear, MSELoss
from torch.optim import SGD
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# 构建数据集
def create_dataset():
    x, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True, bias=1.5, random_state=0)
    x = torch.tensor(x).float()  # 转换为 float
    y = torch.tensor(y).float()  # 转换为 float
    return x, y, coef


x, y, coef = create_dataset()

dataset = TensorDataset(x, y)  # 构造数据集对象
dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)  # 数据加载器

model = Linear(in_features=1, out_features=1)  # 构建模型
loss_fn = MSELoss()  # 损失函数
optimizer = SGD(params=model.parameters(), lr=0.001)

epochs = 100  # 训练一百次
loss_epoch = []  # 用列表来记录每次训练的损失

for epoch in range(epochs):  # 遍历每个轮次
    total_loss = 0.0  # 每个epoch开始时重置总损失
    train_sample = 0.0  # 每个epoch开始时重置样本计数

    for train_x, train_y in dataloader:  # 在每个轮次中迭代每个batch
        # 确保数据类型正确
        train_x = train_x.float()
        train_y = train_y.float().reshape(-1, 1)  # 调整形状并转换类型

        # 前向传播
        y_pred = model(train_x)
        loss_values = loss_fn(y_pred, train_y)

        # 反向传播
        optimizer.zero_grad()  # 梯度清零
        loss_values.backward()  # 自动计算梯度
        optimizer.step()  # 用计算出的梯度更新参数

        # 记录损失
        total_loss += loss_values.item()  # 使用 .item() 获取标量值
        train_sample += len(train_y)

    # 记录每个epoch的平均损失
    avg_loss = total_loss / train_sample
    loss_epoch.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), loss_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# 绘制拟合结果
plt.subplot(1, 2, 2)
x1 = torch.linspace(x.min(), x.max(), 1000)  # 设置直线x的值

# 获取模型参数
weight = model.weight.data.item()  # 获取权重值
bias = model.bias.data.item()  # 获取偏置值

y0 = [v * weight + bias for v in x1]  # 拟合结果
y1 = [v * coef + 1.5 for v in x1]  # 真实直线

plt.scatter(x, y, alpha=0.5, label='data points')
plt.plot(x1, y0, 'r-', label='predicted', linewidth=2)
plt.plot(x1, y1, 'g--', label='real', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"真实参数: w={coef:.4f}, b=1.5")
print(f"预测参数: w={weight:.4f}, b={bias:.4f}")