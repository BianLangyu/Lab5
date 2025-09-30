# 深度学习入门：MLP 与 CNN 实现手写数字识别

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

这是一个面向机器学习初学者的教学项目，旨在通过实践对比两种经典的神经网络模型——**多层感知机 (MLP)** 和 **卷积神经网络 (CNN)**，在 MNIST 手写数字识别任务上的表现。

项目的核心特色在于：
1.  **从零实现 MLP**: 使用纯 `NumPy` 手动实现一个完整的神经网络，帮助深入理解反向传播等底层原理。
2.  **框架实现 CNN**: 使用现代深度学习框架 `PyTorch` 构建一个高效的 CNN 模型，学习工业界标准的工作流程。

通过完成此项目，你将能深刻体会到不同网络结构在处理图像数据时的巨大差异和各自的优势。

## 🚀 项目结构

```
.
├── mlp_from_scratch.py    # Part 1: 使用 NumPy 从零实现 MLP
├── cnn_with_pytorch.py    # Part 2: 使用 PyTorch 实现 CNN
├── requirements.txt       # 项目所需的所有 Python 依赖库
└── README.md              # 本说明文档
```
*注：`data/` 目录将在你首次运行代码时自动创建，用于存放下载的 MNIST 数据集。*

## 🔧 环境配置 (使用 Conda)

为了保证代码顺利运行并简化依赖管理，强烈建议使用 **Anaconda** 或 **Miniconda**。

**前提条件**: 请确保你的系统已安装 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

**1. 克隆仓库**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. 创建并激活 Conda 环境**

*   **创建环境**
    打开终端（Anaconda Prompt for Windows），运行以下命令来创建一个名为 `ml-lab` 且使用 Python 3.9 的新环境：
    ```bash
    conda create -n ml-lab python=3.9 -y
    ```
*   **激活环境**
    创建成功后，使用以下命令激活该环境：
    ```bash
    conda activate ml-lab
    ```
    激活成功后，你的命令行提示符前会显示 `(ml-lab)`。

**3. 安装依赖**

本项目的所有依赖库都记录在 `requirements.txt` 文件中。在已激活的 `ml-lab` 环境中，使用 pip 一键安装：
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
torch版本需要与自己计算机的cuda版本匹配，具体下载链接：https://pytorch.org/get-started/previous-versions/

至此，你的 Conda 实验环境已准备就绪！

## ⚡️ 如何运行

请确保你已激活 `(ml-lab)` Conda 环境。

### 1. 运行 MLP (纯 NumPy 实现)

执行以下命令来训练并评估从零开始实现的 MLP 模型：
```bash
python mlp_from_scratch.py
```
**注意**：此脚本使用全批量梯度下降，训练过程可能需要几分钟时间。请耐心等待。

### 2. 运行 CNN (PyTorch 实现)

执行以下命令来训练并评估使用 PyTorch 构建的 CNN 模型：
```bash
python cnn_with_pytorch.py
```
**注意**：如果你有支持 CUDA 的 NVIDIA 显卡，此脚本会自动使用 GPU 进行加速，训练速度会快很多。

## 📈 预期结果

运行脚本后，你将在终端看到模型的训练过程，并在训练结束后看到弹出的损失曲线图。

### 终端输出

**MLP 脚本的输出应类似：**
```
X_train shape: (784, 60000)
Y_train shape: (10, 60000)
X_test shape: (784, 10000)
Epoch 50/500 - Loss: 0.5321 - Accuracy: 0.8567
Epoch 100/500 - Loss: 0.3854 - Accuracy: 0.8932
...
Epoch 500/500 - Loss: 0.1589 - Accuracy: 0.9551

MLP Test Accuracy: 95.31%
```

**CNN 脚本的输出应类似：**
```
Using cuda device
--- Training CNN Model ---
Epoch 1/10, Loss: 0.2345
Epoch 2/10, Loss: 0.0678
...
Epoch 10/10, Loss: 0.0211

--- Evaluating CNN Model ---
CNN Test Accuracy: 99.05%
```

### 可视化输出

脚本运行结束后，会自动绘制并显示模型的损失（Loss）随训练轮数（Epoch）变化的曲线图，帮助你直观地评估模型的收敛情况。

![Loss Curve Placeholder](https://via.placeholder.com/600x400.png/f0f0f0/333333?text=Loss+Curve+Will+Be+Displayed+Here)
*(示例图：这里会显示你生成的实际损失曲线)*

### 性能基准

一个成功实现的模型应该能达到以下近似的准确率：
*   **MLP (NumPy)**: 测试集准确率 > **90%** (通常可达 95% 左右)
*   **CNN (PyTorch)**: 测试集准确率 > **98%** (通常可达 99% 以上)

如果你的结果与此相差甚远，请仔细检查你的代码实现，特别是 MLP 的反向传播部分。

---
祝实验顺利！