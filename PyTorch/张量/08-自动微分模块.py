import torch

x = torch.ones(2,5)
y = torch.zeros(2,3)
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
z = torch.matmul(x,w) + b
loss = torch.nn.MSELoss()
loss = loss(z,y)
loss.backward()
print(w.grad)
print(b.grad)