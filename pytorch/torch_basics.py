"""ch07/dezero1.py, ch07/dezero2.py 의 파이토치 버전.

DeZero의 Variable/cleargrad/backward 는 파이토치에서
Tensor(requires_grad=True) / grad 초기화 / backward 로 그대로 대응된다.
"""
import numpy as np
import torch


# =============================================================================
# 7.1절 - 텐서 연산 (ch07/dezero1.py)
# =============================================================================
# 벡터의 내적
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = torch.matmul(a, b)  # a @ b 로도 쓸 수 있다
print(c)

# 행렬의 곱
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.matmul(a, b)
print(c)

# 넘파이 배열과의 상호 변환
x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
x = torch.from_numpy(x_np)        # numpy -> torch (메모리를 공유한다)
print(x.numpy())                  # torch -> numpy


# =============================================================================
# 7.1절 - 자동 미분으로 최적화하기 (ch07/dezero2.py)
#
# DeZero               ->  PyTorch
#   Variable(...)          torch.tensor(..., requires_grad=True)
#   x.cleargrad()          x.grad = None
#   y.backward()           y.backward()
#   x.grad.data            x.grad
# =============================================================================
def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2


x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

iters = 10000  # 반복 횟수
lr = 0.001     # 학습률

for i in range(iters):
    y = rosenbrock(x0, x1)

    # 이전 반복에서 더해진 미분 초기화
    # (파이토치는 기울기를 '누적'하므로 매번 지워야 한다)
    x0.grad = None
    x1.grad = None

    # 미분(역전파)
    y.backward()

    # 변수 갱신
    # 갱신 자체는 미분 대상이 아니므로 no_grad 안에서 수행한다
    with torch.no_grad():
        x0 -= lr * x0.grad
        x1 -= lr * x1.grad

print('\n로젠브록 함수의 최솟값 위치 (정답은 (1, 1))')
print('x0 =', x0.item(), ', x1 =', x1.item())


# 참고: 위의 수동 갱신은 옵티마이저로 대체할 수 있다.
x = torch.tensor([0.0, 2.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.001)
for i in range(10000):
    loss = rosenbrock(x[0], x[1])
    optimizer.zero_grad()  # x.grad = None 과 같은 역할
    loss.backward()
    optimizer.step()       # x -= lr * x.grad 와 같은 역할
print('옵티마이저 사용 결과:', x.detach().numpy())
