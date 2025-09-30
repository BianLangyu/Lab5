import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # 仅用于加载数据

# ---------------------------------
# 1. 数据加载与预处理
# ---------------------------------
def load_and_preprocess_data():
    """加载MNIST数据并进行预处理"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 扁平化: (60000, 28, 28) -> (60000, 784)
    # 并转置为 (784, 60000) 以方便矩阵运算
    x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    x_test_flatten = x_test.reshape(x_test.shape[0], -1).T

    # 归一化
    x_train_normalized = x_train_flatten / 255.0
    x_test_normalized = x_test_flatten / 255.0

    # 独热编码 (One-Hot Encoding)
    y_train_onehot = one_hot(y_train, 10)
    y_test_onehot = one_hot(y_test, 10)

    return x_train_normalized, y_train_onehot, x_test_normalized, y_test_onehot, y_test

def one_hot(Y, num_classes):
    """将标签转换为独热编码"""
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# ---------------------------------
# 2. 神经网络核心函数
# ---------------------------------
def initialize_parameters(input_size, hidden_size, output_size):
    """初始化网络参数"""
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def relu(Z):
    """ReLU 激活函数"""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """ReLU 的导数"""
    return Z > 0

def softmax(Z):
    """Softmax 激活函数 (为防止数值溢出进行优化)"""
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, params):
    """前向传播"""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_loss(A2, Y):
    """计算交叉熵损失"""
    m = Y.shape[1] # 样本数量
    # 添加一个极小值 1e-8 防止 log(0)
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / m
    return loss

def backward_propagation(params, cache, X, Y):
    """反向传播，计算梯度"""
    m = X.shape[1]
    W1, W2 = params["W1"], params["W2"]
    A1, A2 = cache["A1"], cache["A2"]
    Z1 = cache["Z1"]
    
    # 输出层梯度
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # 隐藏层梯度
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_parameters(params, grads, learning_rate):
    """使用梯度下降更新参数"""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params

def get_predictions(A2):
    """从输出概率中获取预测类别"""
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y_raw):
    """计算准确率"""
    return np.sum(predictions == Y_raw) / Y_raw.size

# ---------------------------------
# 3. 训练与评估
# ---------------------------------
def model(X_train, Y_train, X_test, Y_test_raw, hidden_size=128, epochs=500, learning_rate=0.1):
    """构建、训练和评估整个模型"""
    input_size = X_train.shape[0]
    output_size = Y_train.shape[0]
    
    params = initialize_parameters(input_size, hidden_size, output_size)
    
    history = {'loss': [], 'accuracy': []}

    for i in range(epochs):
        # 前向传播
        A2, cache = forward_propagation(X_train, params)
        
        # 计算损失
        loss = compute_loss(A2, Y_train)
        
        # 反向传播
        grads = backward_propagation(params, cache, X_train, Y_train)
        
        # 更新参数
        params = update_parameters(params, grads, learning_rate)
        
        if (i + 1) % 50 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, np.argmax(Y_train, axis=0))
            print(f"Epoch {i+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

    # 在测试集上评估
    A2_test, _ = forward_propagation(X_test, params)
    test_predictions = get_predictions(A2_test)
    test_accuracy = get_accuracy(test_predictions, Y_test_raw)
    print(f"\nMLP Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return history

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, Y_test_raw = load_and_preprocess_data()
    
    # 打印数据维度以确认
    print("X_train shape:", X_train.shape) # (784, 60000)
    print("Y_train shape:", Y_train.shape) # (10, 60000)
    print("X_test shape:", X_test.shape)   # (784, 10000)
    
    mlp_history = model(X_train, Y_train, X_test, Y_test_raw)

    # 绘制损失曲线
    plt.figure()
    plt.plot(mlp_history['loss'])
    plt.title('MLP Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch (x50)')
    plt.show()