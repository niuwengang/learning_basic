import torch                                      # PyTorch 核心库
import torch.nn as nn                             # 神经网络模块（层、损失函数等）
import torch.optim as optim                       # 优化器（Adam、SGD 等）
from torchvision import datasets, transforms      # 常用视觉数据集和图像变换
from torch.utils.data import DataLoader           # 批量加载数据的工具
import matplotlib.pyplot as plt                   # 绘图库，用于可视化结果


class VAEEncoder(nn.Module):
    pass

class VAEDecoder(nn.Module):
    pass


class VAE(nn.Module):
    pass


def train_vae(epochs=10, latent_dim=32, batch_size=128, lr=1e-3):
    # 检测是否有 GPU；有则用 CUDA 加速，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST 数据集
    transform = transforms.ToTensor()            # 将 PIL 图片转为 [0,1] 的 Tensor
    # train=True 加载训练集；download=True 若本地没有则自动下载
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    # shuffle=True：每个 epoch 打乱数据顺序，防止模型记住顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(latent_dim=latent_dim).to(device)  # 模型移到指定设备
    optimizer = optim.Adam(model.parameters(), lr=lr)      # Adam 自适应优化器
    criterion = nn.BCELoss(reduction='mean')                           # 二元交叉熵损失，适合像素值 [0,1]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0                          # 累计该 epoch 的总损失
        for x, _ in train_loader:                # _ 是标签，AE 不需要标签（无监督）
            x = x.view(x.size(0), -1).to(device) # 展平：(B,1,28,28) → (B,784)，并移到设备

            x_hat = model(x)                     # 前向传播：x → z → x̂
            loss = criterion(x_hat, x)           # 计算重建误差 BCE(x̂, x)

            optimizer.zero_grad()                # 清空上一步的梯度，防止累加
            loss.backward()                      # 反向传播：计算各参数的梯度
            optimizer.step()                     # 用梯度更新模型参数

            # loss.item() 是该 batch 的平均损失，乘 batch 大小还原为总损失
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_dataset)            # 除以样本总数得平均损失
        print(f"Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}")

    return model                                 # 返回训练好的模型



if __name__ == '__main__':
