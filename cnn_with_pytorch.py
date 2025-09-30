import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------------
# 1. 超参数与设备配置
# ------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# ---------------------------------
# 2. 数据加载与预处理
# ---------------------------------
# 定义转换：转为Tensor，并进行标准化
# MNIST数据集的均值和标准差
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载和加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------
# 3. 构建CNN模型
# ---------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1: 输入通道=1 (灰度图), 输出通道=16, 卷积核=5x5, padding=2 保持尺寸不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 28x28 -> 14x14
        )
        # 卷积层2: 输入通道=16, 输出通道=32, 卷积核=5x5, padding=2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2) # 14x14 -> 7x7
        )
        # 全连接层: 32个7x7的特征图 -> 10个输出类别
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 扁平化操作
        x = x.view(x.size(0), -1) # (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

# ---------------------------------
# 4. 训练与评估
# ---------------------------------
model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model():
    model.train() # 设置为训练模式
    history = {'loss': []}
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 1. 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 2. 反向传播与优化
            optimizer.zero_grad() # 梯度清零
            loss.backward()       # 计算梯度
            optimizer.step()      # 更新权重
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
    return history

def test_model():
    model.eval() # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 在评估阶段, 不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'CNN Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    print("--- Training CNN Model ---")
    cnn_history = train_model()
    print("\n--- Evaluating CNN Model ---")
    test_model()

    # 绘制损失曲线
    plt.figure()
    plt.plot(cnn_history['loss'])
    plt.title('CNN Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()