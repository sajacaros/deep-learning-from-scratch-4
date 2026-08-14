"""ch07/s03_dezero3.py(선형 회귀)의 파이토치 버전."""
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == '__main__':
    # 토이 데이터셋
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 매개변수 정의
    W = torch.zeros((1, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)


    # 예측 함수
    def predict(x):
        return torch.matmul(x, W) + b  # 행렬의 곱으로 여러 데이터 일괄 계산


    # 평균 제곱 오차(식 7.2) 계산 함수
    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return torch.sum(diff ** 2) / len(diff)
        # 또는 torch.nn.functional.mse_loss(x0, x1)


    # 경사 하강법으로 매개변수 갱신
    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.grad = None
        b.grad = None
        loss.backward()

        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad

        if i % 10 == 0:  # 10회 반복마다 출력
            print(loss.item())

    print('====')
    print('W =', W.detach().numpy())
    print('b =', b.detach().numpy())

    # [그림 7-9] 학습 후 모델
    plt.scatter(x.numpy(), y.numpy(), s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = torch.arange(0, 1, .01, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        y_pred = predict(t)
    plt.plot(t.numpy(), y_pred.numpy(), color='r')
    plt.show()
