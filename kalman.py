import numpy as np
from scipy.linalg import inv

# 定义状态变量和状态转移矩阵
x = np.zeros((9, 1))  # [x, y, z, vx, vy, vz, ax, ay, az]
A = np.array([[1, 0, 0, 1, 0, 0, 0.5, 0, 0],
              [0, 1, 0, 0, 1, 0, 0, 0.5, 0],
              [0, 0, 1, 0, 0, 1, 0, 0, 0.5],
              [0, 0, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# 定义观测矩阵和观测噪声
H = np.eye(3)
R = np.eye(3) * 0.1

# 定义过程噪声和初始协方差矩阵
Q = np.eye(9) * 0.01
P = np.eye(9) * 1000

# 初始化状态估计
z = np.array([x[0], x[1], x[2]]).reshape(3, 1)  # 假设第一个观测值为初始观测值
x[:3] = z.flatten()
x[3:] = 0

# 使用卡尔曼滤波平滑每个时刻的观测值
def kalman_filter(observation):
    global x, P
    # 预测状态估计和协方差矩阵
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # 更新状态估计和协方差矩阵
    y = observation - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), inv(S))
    x = x + np.dot(K, y)
    P = np.dot((np.eye(9) - np.dot(K, H)), P)

    # 返回平滑后的观测值
    return x[:3].flatten()

# 测试代码
observed_data = np.random.rand(100, 3)  # 模拟100个时刻的观测值
smoothed_data = np.zeros_like(observed_data)
for i in range(len(observed_data)):
    smoothed_data[i] = kalman_filter(observed_data[i].reshape(3, 1))

# 打印结果
print("原始观测值：\n", observed_data[:10])
print("平滑后的观测值：\n", smoothed_data[:10])