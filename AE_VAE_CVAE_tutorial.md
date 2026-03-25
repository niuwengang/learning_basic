# AE → VAE → CVAE 循序渐进教程

> 目标：从最简单的自编码器出发，一步步理解变分自编码器和条件变分自编码器的原理，并手写 PyTorch 实现。

---

## 目录

1. [AutoEncoder (AE)](#1-autoencoder-ae)
2. [Variational AutoEncoder (VAE)](#2-variational-autoencoder-vae)
3. [Conditional VAE (CVAE)](#3-conditional-vae-cvae)
4. [三者对比总结](#4-三者对比总结)

---

## 1. AutoEncoder (AE)

### 1.1 核心思想

AutoEncoder 的目标是**用一个低维向量（latent code）来表示高维数据**，同时保证从低维向量能还原出原始数据。

```
输入 x  ──► Encoder ──► z（瓶颈）──► Decoder ──► x̂（重建）
               ↓
         压缩/降维
```

整个过程可以理解为：

- **Encoder**：将输入 x 映射到低维隐空间 z，提取"本质特征"。
- **Decoder**：将 z 映射回原始空间，尽量还原 x。
- **Loss**：最小化重建误差（Reconstruction Loss），对于连续数据用 MSE，对于二值数据用 BCE。

### 1.2 数学表达

$$\mathcal{L}_{AE} = \| x - \hat{x} \|^2 \quad \text{（MSE）}$$

或

$$\mathcal{L}_{AE} = -\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)] \quad \text{（BCE）}$$

其中 $\hat{x} = \text{Decoder}(\text{Encoder}(x))$。

### 1.3 AE 的局限

AE 学到的隐空间 z **没有任何结构约束**：

- z 的分布是任意的，不同样本的 z 可能分散在空间各处。
- **无法做生成**：你不知道从哪里采样 z 才能解码出有意义的图像。
- 隐空间不连续：两个相邻的 z 解码出来的结果可能完全不同。

### 1.4 手写代码

```python
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
    criterion = nn.BCELoss()                               # 二元交叉熵损失，适合像素值 [0,1]

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
    x_samples, _ = next(iter(DataLoader(test_dataset, batch_size=n)))
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
```

**关键点回顾：**

| 组件 | 作用 |
|------|------|
| Encoder | x → z（确定性映射） |
| Decoder | z → x̂ |
| Loss | BCE/MSE（仅重建误差） |
| 隐空间 | 无约束，不适合生成 |

---

## 2. Variational AutoEncoder (VAE)

### 2.1 核心思想

VAE 在 AE 的基础上加了一个关键约束：**隐空间要服从标准正态分布 N(0, I)**。

这样做的好处：
- 隐空间是**连续且结构化**的，可以从 N(0, I) 随机采样来生成新样本。
- 相邻的 z 点解码后语义相近（插值有意义）。

AE 中 Encoder 直接输出 z，而 VAE 中 Encoder 输出的是**分布的参数**：

```
x  ──► Encoder ──► μ, log σ²
                      ↓
                   z ~ N(μ, σ²I)   ← 重参数化采样
                      ↓
              Decoder ──► x̂
```

### 2.2 ELBO 推导

VAE 的目标是最大化数据的对数似然 $\log p(x)$，通过变分推断引入近似后验 $q_\phi(z|x)$：

$$\log p(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 散度项}}$$

这就是 **ELBO（Evidence Lower BOund）**，我们最大化 ELBO 等价于最小化：

$$\mathcal{L}_{VAE} = \underbrace{\| x - \hat{x} \|^2}_{\text{重建 Loss}} + \underbrace{D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I))}_{\text{KL Loss}}$$

**KL 散度的闭式解（当先验为标准正态时）：**

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 2.3 重参数化技巧（Reparameterization Trick）

问题：采样操作 $z \sim \mathcal{N}(\mu, \sigma^2)$ 不可微，无法反向传播。

解决：将随机性"外包"给一个独立的噪声变量 $\epsilon$：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这样梯度可以顺利流过 $\mu$ 和 $\sigma$。

```
           ε ~ N(0, I)  ← 独立噪声（不参与梯度）
                ↓
x → Encoder → μ, σ  →  z = μ + σ·ε  → Decoder → x̂
      ↑_____________________________↑
              梯度正常流动
```

### 2.4 手写代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ─── 模型定义 ───────────────────────────────────────────────
class VAEEncoder(nn.Module):
    """将输入 x 映射为隐变量分布的参数 (μ, log σ²)"""
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log 方差

    def forward(self, x):
        h = self.shared(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        """重参数化采样：z = μ + σ·ε"""
        std = torch.exp(0.5 * log_var)   # σ = exp(log σ² / 2)
        eps = torch.randn_like(std)       # ε ~ N(0, I)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def sample(self, n_samples, device):
        """从先验 N(0, I) 采样生成新图像"""
        z = torch.randn(n_samples, self.encoder.fc_mu.out_features).to(device)
        with torch.no_grad():
            return self.decoder(z)


# ─── Loss 函数 ──────────────────────────────────────────────
def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    """
    ELBO Loss = 重建 Loss + beta * KL 散度
    beta=1 是标准 VAE；beta>1 是 β-VAE（更解耦的隐空间）
    """
    # 重建损失（BCE）
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    # KL 散度（闭式解）
    # -0.5 * sum(1 + log_var - mu² - exp(log_var))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (recon_loss + beta * kl_loss) / x.size(0)  # 对 batch 取均值


# ─── 训练 ───────────────────────────────────────────────────
def train_vae(epochs=20, latent_dim=32, batch_size=128, lr=1e-3, beta=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)

            x_hat, mu, log_var = model(x)
            loss = vae_loss(x, x_hat, mu, log_var, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        print(f"Epoch [{epoch}/{epochs}]  Loss: {total_loss / len(train_dataset):.4f}")

    return model


# ─── 可视化：生成 + 隐空间插值 ──────────────────────────────
def visualize_vae(model):
    device = next(model.parameters()).device

    # 1. 随机生成
    samples = model.sample(16, device).cpu().view(16, 28, 28)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(16):
        axes[i // 8, i % 8].imshow(samples[i].detach(), cmap="gray")
        axes[i // 8, i % 8].axis("off")
    plt.suptitle("VAE 随机生成样本")
    plt.tight_layout()
    plt.savefig("vae_generated.png", dpi=120)
    plt.show()

    # 2. 隐空间插值（仅 2D 隐变量时直观，latent_dim=2）
    # 将两个向量之间做线性插值，观察解码结果如何平滑过渡


if __name__ == "__main__":
    vae_model = train_vae(epochs=20, latent_dim=32)
    visualize_vae(vae_model)
```

**关键点回顾：**

| 组件 | AE | VAE |
|------|----|-----|
| Encoder 输出 | z（确定值） | μ, log σ²（分布参数） |
| 采样 | 无 | z = μ + σ·ε（重参数化） |
| 隐空间约束 | 无 | KL 散度拉向 N(0,I) |
| Loss | 重建 | 重建 + KL |
| 能否生成 | 不能 | 能（从 N(0,I) 采样） |

---

## 3. Conditional VAE (CVAE)

### 3.1 核心思想

VAE 能生成图像，但**无法控制生成什么**（随机采样得到随机数字）。

CVAE 引入**条件标签 c**，让编码和解码都知道"当前处理的是什么类别"：

```
x, c  ──► Encoder(x, c) ──► μ, log σ²
                                 ↓
                          z ~ N(μ, σ²)
                                 ↓
              Decoder(z, c) ──► x̂
```

这样可以实现**条件生成**：给定标签 c=3，就能生成数字 "3" 的图像。

### 3.2 数学表达

CVAE 的 ELBO 条件化在 c 上：

$$\mathcal{L}_{CVAE} = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - D_{KL}(q_\phi(z|x,c) \| p(z|c))$$

当先验 $p(z|c) = \mathcal{N}(0, I)$（不依赖 c）时，KL 项与 VAE 相同，只是 Encoder/Decoder 都额外接收 c：

$$\mathcal{L}_{CVAE} = \underbrace{\text{BCE}(x, \hat{x})}_{\text{重建}} + \underbrace{D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I))}_{\text{KL}}$$

### 3.3 条件如何注入

最简单的方式：**将 one-hot 编码的标签与输入拼接（concatenate）**。

```
Encoder 输入: [x (784-dim), c (10-dim)] → 794-dim
Decoder 输入: [z (32-dim), c (10-dim)]  → 42-dim
```

### 3.4 手写代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ─── One-Hot 编码工具 ────────────────────────────────────────
def one_hot(labels, num_classes=10):
    """将整数标签转换为 one-hot 向量"""
    return F.one_hot(labels, num_classes).float()


# ─── 模型定义 ───────────────────────────────────────────────
class CVAEEncoder(nn.Module):
    def __init__(self, input_dim=784, cond_dim=10, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),  # 拼接条件
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=-1)   # 拼接 x 和条件 c
        h = self.net(xc)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=32, cond_dim=10, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),  # 拼接条件
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=-1)   # 拼接 z 和条件 c
        return self.net(zc)


class CVAE(nn.Module):
    def __init__(self, input_dim=784, cond_dim=10, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = CVAEDecoder(latent_dim, cond_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, c)
        return x_hat, mu, log_var

    def generate(self, c, device):
        """给定条件标签 c，从先验 N(0,I) 采样生成图像"""
        z = torch.randn(c.size(0), self.latent_dim).to(device)
        with torch.no_grad():
            return self.decoder(z, c)


# ─── Loss 函数（与 VAE 相同）────────────────────────────────
def cvae_loss(x, x_hat, mu, log_var, beta=1.0):
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon + beta * kl) / x.size(0)


# ─── 训练 ───────────────────────────────────────────────────
def train_cvae(epochs=20, latent_dim=32, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, labels in train_loader:
            x = x.view(x.size(0), -1).to(device)
            c = one_hot(labels, num_classes=10).to(device)  # (B, 10)

            x_hat, mu, log_var = model(x, c)
            loss = cvae_loss(x, x_hat, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        print(f"Epoch [{epoch}/{epochs}]  Loss: {total_loss / len(train_dataset):.4f}")

    return model


# ─── 可视化：指定标签生成 ────────────────────────────────────
def visualize_cvae(model, num_classes=10, samples_per_class=8):
    """每个数字类别生成 samples_per_class 张图"""
    device = next(model.parameters()).device

    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 1.2, num_classes * 1.2))

    for digit in range(num_classes):
        labels = torch.full((samples_per_class,), digit, dtype=torch.long)
        c = one_hot(labels, num_classes=10).to(device)
        imgs = model.generate(c, device).cpu().view(samples_per_class, 28, 28)

        for j in range(samples_per_class):
            axes[digit, j].imshow(imgs[j].detach(), cmap="gray")
            axes[digit, j].axis("off")
        axes[digit, 0].set_ylabel(str(digit), fontsize=10, rotation=0, labelpad=15)

    plt.suptitle("CVAE 条件生成（每行为一个数字类别）", fontsize=12)
    plt.tight_layout()
    plt.savefig("cvae_conditional_generation.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    cvae_model = train_cvae(epochs=20, latent_dim=32)
    visualize_cvae(cvae_model)
```

**关键点回顾：**

| 组件 | VAE | CVAE |
|------|-----|------|
| Encoder 输入 | x | x + c（拼接） |
| Decoder 输入 | z | z + c（拼接） |
| 生成控制 | 无，纯随机 | 指定 c，可控生成 |
| 应用场景 | 无监督生成 | 条件生成、数据增强 |

---

## 4. 三者对比总结

### 4.1 架构演进

```
AE:
  x → [Encoder] → z (确定值) → [Decoder] → x̂
  Loss = Recon(x, x̂)

VAE:
  x → [Encoder] → μ, σ → z=μ+σε → [Decoder] → x̂
  Loss = Recon(x, x̂) + KL(N(μ,σ²) ‖ N(0,I))

CVAE:
  x,c → [Encoder] → μ, σ → z=μ+σε → [Decoder(z,c)] → x̂
  Loss = Recon(x, x̂) + KL(N(μ,σ²) ‖ N(0,I))
```

### 4.2 核心差异表

| 特性 | AE | VAE | CVAE |
|------|:--:|:---:|:----:|
| 隐空间约束 | ✗ | ✓ N(0,I) | ✓ N(0,I) |
| 可生成新样本 | ✗ | ✓ | ✓ |
| 可控生成 | ✗ | ✗ | ✓ |
| 需要标签 | ✗ | ✗ | ✓ |
| Loss 组成 | Recon | Recon + KL | Recon + KL |
| 重参数化 | 无需 | 需要 | 需要 |

### 4.3 应用场景

| 模型 | 典型应用 |
|------|---------|
| AE | 降维、去噪、异常检测（无需生成能力时） |
| VAE | 无监督图像生成、隐空间插值、表征学习 |
| CVAE | 类别可控生成、图像修复（给定条件）、数据增强 |

### 4.4 常见问题 & 调参技巧

**Q: VAE 生成的图像为什么模糊？**
> 原因：BCE/MSE 损失会对像素求平均，导致模糊。改进方向：使用感知损失（Perceptual Loss）或换用 GAN-based 方法。

**Q: KL 消失（KL Collapse/Posterior Collapse）怎么办？**
> 训练初期 Decoder 太强，直接忽略 z，导致 KL → 0、z 无意义。解决方案：
> - **KL Annealing**：训练初期让 β 从 0 逐渐增大到 1。
> - **Free Bits**：允许每个维度有一定 KL 自由度，不强制为 0。

**Q: β-VAE 是什么？**
> 将 Loss 中 KL 项的权重放大（β > 1），强制隐变量更加解耦（每个维度对应独立的语义因子）。

**Q: CVAE 和 Conditional GAN 有何区别？**
> CVAE 有明确的概率解释和 ELBO 优化目标，训练更稳定；CGAN 生成质量更高但训练不稳定。

---

### 4.5 一个文件运行三个模型

```python
# run_all.py
from ae import train_ae, visualize_ae
from vae import train_vae, visualize_vae
from cvae import train_cvae, visualize_cvae

print("=== 训练 AE ===")
ae  = train_ae(epochs=5)
visualize_ae(ae)

print("=== 训练 VAE ===")
vae = train_vae(epochs=10)
visualize_vae(vae)

print("=== 训练 CVAE ===")
cvae = train_cvae(epochs=10)
visualize_cvae(cvae)
```

---

> **学习路径建议：**
> 1. 先跑通 AE，理解 Encoder-Decoder 结构。
> 2. 改造成 VAE，理解重参数化和 ELBO。
> 3. 在 VAE 基础上加条件，得到 CVAE。
> 4. 尝试将 latent_dim 设为 2，用 t-SNE/PCA 可视化隐空间分布，直观感受约束的作用。
