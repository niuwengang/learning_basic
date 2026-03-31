import torch                                      # PyTorch 核心库
import torch.nn as nn                             # 神经网络模块（层、损失函数等）
import torch.optim as optim                       # 优化器（Adam、SGD 等）
from torchvision import datasets, transforms      # 常用视觉数据集和图像变换
from torch.utils.data import DataLoader           # 批量加载数据的工具
import matplotlib.pyplot as plt                   # 绘图库，用于可视化结果

# ─── 模型定义 ───────────────────────────────────────────────
class Encoder(nn.Module):
    # input_dim=784：MNIST 图片展平后的维度（28×28=784）
    # hidden_dim=256：隐藏层神经元数，介于输入和隐变量之间起过渡作用
    # latent_dim=32：隐变量 z 的维度，即"瓶颈"大小
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()                        # 调用父类 nn.Module 的初始化
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # 全连接：784 → 256，提取中间特征
            nn.ReLU(),                            # 激活函数，引入非线性，负值置零
            nn.Linear(hidden_dim, latent_dim),    # 全连接：256 → 32，压缩到隐空间
        )

    def forward(self, x):
        return self.net(x)                        # 顺序经过上面三层，返回隐变量 z


class Decoder(nn.Module):
    # latent_dim=32：接收来自 Encoder 的 z
    # hidden_dim=256：先扩展到中间维度
    # output_dim=784：最终还原到与输入相同的维度
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),    # 全连接：32 → 256，开始扩展
            nn.ReLU(),                            # 非线性激活
            nn.Linear(hidden_dim, output_dim),    # 全连接：256 → 784，还原维度
            nn.Sigmoid(),   # 输出范围 [0,1]，与 BCE 搭配；MNIST 像素值也在 [0,1]
        )

    def forward(self, z):
        return self.net(z)                        # z 经过解码网络，输出重建图像 x̂


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)   # 编码器子模块
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)   # 解码器子模块

    def forward(self, x):
        z = self.encoder(x)       # x → z：压缩到隐空间（确定性映射）
        x_hat = self.decoder(z)   # z → x̂：从隐空间还原
        return x_hat              # 返回重建结果，供计算重建误差


# ─── 训练 ───────────────────────────────────────────────────
def train_ae(epochs=10, latent_dim=32, batch_size=128, lr=1e-3):
    # 检测是否有 GPU；有则用 CUDA 加速，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST 数据集
    transform = transforms.ToTensor()            # 将 PIL 图片转为 [0,1] 的 Tensor
    # train=True 加载训练集；download=True 若本地没有则自动下载
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    # shuffle=True：每个 epoch 打乱数据顺序，防止模型记住顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(latent_dim=latent_dim).to(device)  # 模型移到指定设备
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


# ─── 可视化重建结果 ──────────────────────────────────────────
def visualize_ae(model, n=8):
    device = next(model.parameters()).device     # 获取模型所在设备（CPU/GPU）
    transform = transforms.ToTensor()
    # train=False：加载测试集，评估模型在未见过数据上的效果
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    # 取一个 batch（n 张）作为演示样本
    x_samples, _ = next(iter(DataLoader(test_dataset, batch_size=n))) #(n, 1, 28, 28)
    x_flat = x_samples.view(n, -1).to(device)   # 展平并移到设备

    with torch.no_grad():                        # 关闭梯度计算，节省内存，加快速度
        x_hat = model(x_flat).cpu().view(n, 28, 28)  # 前向推理，结果移回 CPU 并还原为图片形状

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))  # 2 行（原始/重建）× n 列
    for i in range(n):
        axes[0, i].imshow(x_samples[i].squeeze(), cmap="gray")  # 第 0 行：原始图片
        axes[0, i].axis("off")                   # 隐藏坐标轴
        axes[1, i].imshow(x_hat[i], cmap="gray") # 第 1 行：重建图片
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("原始", fontsize=10)   # 最左列标注行名
    axes[1, 0].set_ylabel("重建", fontsize=10)
    plt.tight_layout()                           # 自动调整子图间距，避免重叠
    plt.savefig("ae_reconstruction.png", dpi=120)  # 保存图片
    plt.show()                                   # 弹出显示窗口


if __name__ == "__main__":
    ae_model = train_ae(epochs=10, latent_dim=32)  # 训练 10 轮，隐变量维度 32
    visualize_ae(ae_model)                         # 可视化重建结果