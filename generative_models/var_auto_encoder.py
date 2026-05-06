import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms  # datasets: 内置数据集(MNIST等); transforms: 数据预处理/变换工具
from torch.utils.data import DataLoader       # 批量加载数据的工具
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

# ============================================================
# VAE（变分自编码器）原理简介
# ============================================================
# 普通自编码器（AE）把输入压缩成一个固定的隐向量 z，
# 但这个 z 是离散的点，无法在隐空间中随机采样生成新图像。
#
# VAE 的改进：让编码器不输出一个点，而是输出一个分布 q(z|x)，
# 通常假设为高斯分布 N(μ, σ²)。
# 训练目标（ELBO）= 重建质量 - KL散度惩罚
#   - 重建损失：让解码器能从采样的 z 还原出原始图像
#   - KL 散度：让隐分布尽量接近标准正态 N(0,I)，使隐空间连续可插值
#
# 前向流程：
#   x  →  Encoder  →  (μ, log σ²)
#                          ↓  重参数化采样  z = μ + σ·ε,  ε~N(0,I)
#   z  →  Decoder  →  x̂
# ============================================================


class VAEEncoder(nn.Module):
    """
    编码器：将输入图像 x 映射为隐变量分布的参数 (μ, log σ²)。
    输出两个向量而非一个，是 VAE 与普通 AE 的核心区别。
    """
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        """
        input_dim  : 输入维度，MNIST 展平后为 28×28=784
        hidden_dim : 中间隐层神经元数
        latent_dim : 隐空间维度（压缩后的向量长度）
        """
        super().__init__()
        # 共享的特征提取网络，将高维输入压缩为中间表示 h
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # 激活函数，引入非线性
        )
        # 两个独立的全连接层，分别预测分布的均值和对数方差
        # 使用 log σ² 而非 σ² 的原因：σ² 必须为正，log σ² 无约束，网络更好优化
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)  # 输出均值 μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 输出 log σ²（对数方差）

    def forward(self, x):
        h       = self.net(x)          # 提取中间特征
        mu      = self.fc_mu(h)        # 分布均值
        log_var = self.fc_logvar(h)    # 分布对数方差
        return mu, log_var             # 返回高斯分布参数，供重参数化使用


class VAEDecoder(nn.Module):
    """
    解码器：将隐向量 z 还原为重建图像 x̂。
    结构与编码器镜像对称。
    """
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # Sigmoid 将输出压缩到 [0,1]，与 MNIST 像素值范围一致，
            # 同时也是 BCE（二值交叉熵）损失函数的要求
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)  # z → x̂，输出重建图像（展平的784维向量）


class VAE(nn.Module):
    """
    完整 VAE 模型：编码器 + 重参数化 + 解码器。
    """
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧（Reparameterization Trick）
        ----------------------------------------
        问题：z = 从 N(μ, σ²) 中采样 → 采样操作不可微，梯度无法回传给编码器。
        解决：把随机性从网络参数中分离出去：
              z = μ + σ · ε,   ε ~ N(0, I)
        这样梯度可以通过 μ 和 σ 正常回传，ε 只是一个与参数无关的噪声。

        σ  = exp(log σ² / 2) = exp(0.5 * log_var)
        """
        std = torch.exp(0.5 * log_var)   # σ：从 log σ² 还原标准差
        eps = torch.randn_like(std)       # ε：与 std 同形状的标准正态噪声
        return mu + std * eps             # z：可微的采样结果

    def forward(self, x):
        # 1. 编码：x → (μ, log σ²)
        mu, log_var = self.encoder(x)
        # 2. 重参数化采样：从 N(μ, σ²) 采一个 z
        z = self.reparameterize(mu, log_var)
        # 3. 解码：z → x̂
        x_hat = self.decoder(z)
        # 返回重建图像和分布参数（损失计算需要 μ 和 log σ²）
        return x_hat, mu, log_var


def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    """
    VAE 损失函数 = 重建损失 + β × KL散度
    即最大化证据下界（ELBO，Evidence Lower BOund）的负值。

    重建损失（BCE）：
        衡量 x̂ 与原始 x 的像素级差异。
        使用 reduction="sum" 对一个批次内所有像素求和（而非平均），
        与 KL 散度的量纲保持一致，便于调节 β 权重。

    KL 散度（闭式解）：
        KL[q(z|x) || p(z)]，其中 p(z)=N(0,I) 是先验分布。
        当 q(z|x)=N(μ, σ²) 时，KL 有解析解：
        KL = -0.5 × Σ(1 + log σ² - μ² - σ²)
           = -0.5 × Σ(1 + log_var - mu² - exp(log_var))
        KL ≥ 0，它惩罚隐分布偏离标准正态的程度。

    β 参数（β-VAE）：
        β=1：标准 VAE
        β>1：更强的 KL 惩罚 → 隐空间更解耦（每个维度独立编码一个语义特征），
              但重建质量可能下降。
    """
    # 重建损失：像素级 BCE，对批次内所有样本所有像素求和
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    # KL 散度（闭式解），对批次内所有样本所有隐维度求和
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss


if __name__ == '__main__':
    # ── 设备选择 ──────────────────────────────────────────────
    # 优先使用 GPU（cuda），没有 GPU 则退回 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 数据集准备 ────────────────────────────────────────────
    # ToTensor(): PIL图片 → Tensor，像素值 [0,255] → [0.0,1.0]，维度 (H,W,C) → (C,H,W)
    transform = transforms.ToTensor()

    # MNIST：6万张训练图 + 1万张测试图，28×28 灰度手写数字
    # 参考文档：https://docs.pytorch.org/vision/0.22/generated/torchvision.datasets.MNIST.html
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # DataLoader 将数据集包装成可迭代的批次流
    # batch_size=128：每次喂给模型128张图；shuffle=True：每个 epoch 随机打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # ── 模型 / 优化器 ─────────────────────────────────────────
    latent_dim = 32   # 隐空间维度：压缩后每张图用32个数表示
    beta = 1.0        # KL 惩罚强度，1.0 为标准 VAE

    model = VAE(latent_dim=latent_dim).to(device)           # 构建模型并移到目标设备
    optimizer = optim.Adam(model.parameters(), lr=1e-3)     # Adam 自适应优化器，学习率 0.001

    # ── 训练循环 ──────────────────────────────────────────────
    epochs = 10
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0.0
        for image, cls in train_loader:
            # cls 是类别标签（0-9），VAE 无监督训练不使用标签，所以用 _ 接收也可以

            # 展平：(B, 1, 28, 28) → (B, 784)，并移到目标设备
            image = image.view(image.size(0), -1).to(device)

            # 前向传播：得到重建图像和隐分布参数
            image_hat, mu, log_var = model(image)

            # 计算 VAE 损失（重建 + KL）
            loss = vae_loss(image, image_hat, mu, log_var, beta=beta)

            optimizer.zero_grad()  # 清空上一步的梯度，避免累加
            loss.backward()        # 反向传播：计算各参数的梯度
            optimizer.step()       # 梯度下降：更新参数

            total_loss += loss.item()  # .item() 将单元素 Tensor 转为 Python float

        # 每样本平均损失（loss 用 sum reduction，所以除以样本总数）
        average_loss = total_loss / len(train_dataset)
        logger.info(f'Epoch: {epoch}, Loss: {average_loss:.4f}')

    # ── 保存模型权重 ──────────────────────────────────────────
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/vae_model.pth'
    # state_dict() 只保存参数权重，不保存模型结构（体积小，推荐做法）
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f'模型已保存到 {ckpt_path}')

    # ── 推理验证 ──────────────────────────────────────────────
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    test_loader  = DataLoader(test_dataset, batch_size=6, shuffle=True)

    # 重新构建模型并加载权重（模拟实际部署场景）
    infer_model = VAE(latent_dim=latent_dim).to(device)
    # weights_only=True：只恢复权重张量，禁止执行任意 pickle 代码（安全最佳实践）
    infer_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    infer_model.eval()  # 切换为推理模式（关闭 Dropout/BatchNorm 的训练行为）

    # 取一批测试图片
    images, labels = next(iter(test_loader))  # images: (B, 1, 28, 28)
    images_flat = images.view(images.size(0), -1).to(device)  # → (B, 784)

    with torch.no_grad():  # 推理时不需要计算梯度，节省显存和时间
        reconstructed, _, _ = infer_model(images_flat)  # reconstructed: (B, 784)

    # ── 可视化：原图 vs 重建图 ────────────────────────────────
    n = images.shape[0]  # 批次大小（6）
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        # 第一行：原始图像
        # squeeze() 去掉 channel 维度：(1, 28, 28) → (28, 28)
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')

        # 第二行：重建图像
        # view(28, 28)：(784,) → (28, 28)；.cpu() 将数据移回 CPU 再转 numpy
        axes[1, i].imshow(reconstructed[i].view(28, 28).cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title('原图', fontsize=10)
    axes[1, 0].set_title('重建', fontsize=10)
    plt.tight_layout()
    plt.savefig('vae_reconstruction.png', dpi=150)
    logger.info('推理验证完成，结果已保存到 vae_reconstruction.png')
