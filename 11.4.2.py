
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#XOR 问题的输入和输出
# 输入: [0,0], [0,1], [1,0], [1,1]
# 输出: 0, 1, 1, 0
X = np.array([[0, 0],[0, 1],[1,0],[1, 1]])
y = np.array([[0],[1],[1],[0]])
print("XOR 数据集:")
print("输入 X:")
print(X)
print("输出 y:")
print(y)
print(f"数据形状:X{X.shape},y{y.shape}")

# 设置网络结构
input_size=2 #输入特征数
hidden_size=4 #隐藏层神经元数
output_size=1 #输出层神经元数
#初始化权重和偏置
# 使用较小的随机值初始化,避免梯度爆炸
np.random.seed(42)#设置随机种子,确保结果可重现
#输入层到隐藏层的权重和偏置
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
#隐藏层到输出层的权重和偏置
W2 = np.random.randn(hidden_size, output_size) * 0.1

b2 = np.zeros((1, output_size))
print("初始化后的网络参数:")
print(f"W1 形状:{W1.shape}, 值:\n{W1}")
print(f"b1 形状:{b1.shape}, 值:\n{b1}")
print(f"W2 形状:{W2.shape}, 值:\n{W2}")
print(f"b2 形状:{b2.shape}, 值:\n{b2}")

# Sigmoid 激活函数
def sigmoid(x):
    """
    将输入映射到(0,1)区间
    适合二分类问题的输出层
    """

    #限制 x 的范围,避免数值溢出
    x = np.clip(x, -250, 250)
    return 1 / (1 + np.exp(-x))
# Sigmoid 函数的导数
def sigmoid_derivative(x):
    """
        用于反向传播计算梯度
    注意:这里 x 应该是 sigmoid 函数的输出,而不是原始输入
    """

    return x * (1 - x)
#测试激活函数
test_input = np.array([-2, -1, 0, 1, 2])
print("Sigmoid 函数测试:")
print(f"输入:{test_input}")
print(f"输出:{sigmoid(test_input)}")
print(f"导数:{sigmoid_derivative(sigmoid(test_input))}")

def forward_propagation(X, W1, b1, W2, b2):
    """
    执行前向传播
    参数:
    X:输入数据
    W1, b1: 隐藏层权重和偏置
    W2, b2: 输出层权重和偏置
    返回:
    z1, a1: 隐藏层的加权输入和激活输出
    z2, a2:输出层的加权输入和激活输出
    """
    # 隐藏层计算
    z1 =np.dot(X, W1)+b1 # 加权输入
    a1 = sigmoid(z1)

    # 激活输出

    #输出层计算
    z2=np.dot(a1, W2)+b2 #加权输入
    a2 = sigmoid(z2)

    #激活输出(最终预测)
    return z1, a1, z2, a2

#测试前向传播
print("测试前向传播:")
z1, a1, z2, a2 = forward_propagation(X, W1, b1, W2, b2)
print(f"输入 X:\n{X}")
print(f"隐藏层加权输入 z1:\n{z1}")
print(f"隐藏层激活输出 a1:\n{a1}")
print(f"输出层加权输入 z2:\n{z2}")
print(f"最终输出 a2:\n{a2}")
print(f"预测结果(四舍五入):\n{np.round(a2)}")


def compute_loss(y_pred, y_true):
    """计算均方误差损失"""
    #MSE = 1/n * ∑(y_pred -y_true)^2
    MSE = mean_squared_error(y_pred,y_true)
    return np.mean((y_pred - y_true) ** 2)
#测试损失计算
current_loss = compute_loss(a2, y)
print(f"当前预测:\n{a2}")
print(f"真实标签:\n{y}")
print(f"初始损失:{current_loss:6f}")


def backward_propagation(X, y, z1, a1, z2, a2, W1, W2, learning_rate=0.1):
    m = X.shape[0] # 样本数量
    # 1. 计算输出层的误差和梯度
    #损失函数对输出的导数:dL/da2=2*(a2-y)/m
    d_loss_da2 = 2* (a2 - y) / m
    # Sigmoid 函数的导数:da2/dz2=a2*(1-a2)
    da2_dz2= sigmoid_derivative(a2)
    #链式法则:dL/dz2=dL/da2 * da2/dz2
    delta2 = d_loss_da2 *da2_dz2
    # 2.计算隐藏层的误差和梯度
    #误差从输出层传播到隐藏层:dL/da1 =delta2·W2^T
    d_loss_da1=np.dot(delta2, W2.T)
    # Sigmoid 函数的导数:da1/dz1=a1*(1-a1)
    da1_dz1 = sigmoid_derivative(a1)
    #链式法则:dL/dz1=dL/da1 *da1/dz1
    delta1 = d_loss_da1 * da1_dz1
    #3. 计算权重和偏置的梯度
    #输出层权重梯度:dL/dW2=a1^T·delta2
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    #隐藏层权重梯度:dL/dW1=X^T·delta1
    dW1= np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)
    #4.更新权重和偏置(梯度下降)
    W1_updated = W1 - learning_rate * dW1
    b1_updated = b1 - learning_rate * db1
    W2_updated = W2 - learning_rate * dW2
    b2_updated = b2 - learning_rate * db2
    return W1_updated, b1_updated, W2_updated, b2_updated, dW1, dW2
# 测试一次反向传播
print("测试一次反向传播:")
W1_new, b1_new, W2_new, b2_new, dW1, dW2 = backward_propagation(X, y, z1, a1, z2, a2, W1,W2)
print(f"W1 梯度形状:{dW1.shape}")
print(f"W2 梯度形状:{dW2.shape}")
print(f"权重已更新,准备开始训练 …. ")


#训练参数
epochs = 10000
learning_rate = 1.0
print_interval = 1000
#存储训练过程中的损失
losses = []
#重新初始化权重(为了清晰的训练演示)
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))
print("开始训练 MLP 网络…")
print(f"训练参数:{epochs}轮,学习率:{learning_rate}")
print("-" * 50)
for epoch in range(epochs):
#前向传播
    z1, a1, z2, a2 = forward_propagation(X, W1, b1, W2, b2)
#计算损失
    loss = compute_loss(a2, y)
    losses.append(loss)
#反向传播并更新权重
    W1, b1, W2, b2, _, _= backward_propagation(X, y, z1, a1, z2, a2, W1, W2, learning_rate)
#定期打印训练进度
    if epoch % print_interval == 0:
        print(f"轮次 {epoch:4d}, 损失:{loss:.6f}")
print("-" * 50)
print(f"训练完成!最终损失:{loss:.6f}")





