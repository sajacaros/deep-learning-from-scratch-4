"""ch07/dezero4.py(2층 신경망)의 파이토치 버전.

DeZero               ->  PyTorch
  dezero.Model           torch.nn.Module
  L.Linear(out_size)     nn.Linear(in_size, out_size)  ← 입력 크기를 직접 적어준다
  model.cleargrads()     optimizer.zero_grad()
  optimizer.update()     optimizer.step()
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터셋 생성
np.random.seed(0)
torch.manual_seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

lr = 0.2
iters = 10000


class TwoLayerNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(1, 10, 1)
optimizer = optim.SGD(model.parameters(), lr=lr)  # 최적화할 매개변수를 등록

for i in range(iters):
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(loss.item())

# 그래프로 시각화([그림 7-12]와 같음)
plt.scatter(x.numpy(), y.numpy(), s=10)
plt.xlabel('x')
plt.ylabel('y')
t = torch.arange(0, 1, .01, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    y_pred = model(t)
plt.plot(t.numpy(), y_pred.numpy(), color='r')
plt.show()
