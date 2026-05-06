import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms  # datasets: 内置数据集(MNIST等); transforms: 数据预处理/变换工具
from torch.utils.data import DataLoader       # 批量加载数据的工具
import torch.nn as nn
import torch.optim as optim    
from loguru import logger
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(), #输出范围 [0,1]，与 BCE 搭配；MNIST 像素值也在 [0,1]
        )
    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.encoder=Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder=Decoder(latent_dim, hidden_dim, input_dim)


    def forward(self, x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat

if __name__ == '__main__':
    # 自动选择设备: 有GPU用GPU(cuda), 否则用CPU
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ToTensor(): 将PIL图片/numpy数组转为Tensor, 同时把像素值从[0,255]归一化到[0.0,1.0], 并将维度从(H,W,C)转为(C,H,W)
    transform = transforms.ToTensor()
    #MNIST数据集见:https://docs.pytorch.org/vision/0.22/generated/torchvision.datasets.MNIST.html
    train_dataset=datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    latent_dim=32
    model = AutoEncoder(latent_dim=latent_dim).to(device)  #模型移到指定设备
    optimizer = optim.Adam(model.parameters(), lr=1e-3)      # Adam 自适应优化器
    criterion = nn.BCELoss(reduction='mean')             #用对数衡量概率差异 L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]   

    epochs=10
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0.0
        for image, cls in train_loader:
            image= image.view(image.size(0), -1).to(device) # 展平：(B,1,28,28) → (B,784)，并移到设备
            image_hat=model(image)  #前向传播
            loss = criterion(image_hat, image)  #预测值，真实值，计算损失

            optimizer.zero_grad() #梯度重置
            loss.backward() # 反向传播
            optimizer.step()  

            total_loss+=loss.item()
        
        average_loss = total_loss / len(train_loader)
        logger.info(f'Epoch: {epoch}, Loss: {average_loss}')
    
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/ae_model.pth'
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f'模型已保存到 {ckpt_path}')


    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    infer_model = AutoEncoder(latent_dim=latent_dim).to(device)
    infer_model.load_state_dict(torch.load(ckpt_path))
    infer_model.eval()

    # 取一批测试图片进行推理
    images, labels = next(iter(test_loader))  #(B, 1, 28, 28)
    images_flat=images.view(images.size(0), -1).to(device) #(B, 1, 28, 28)->(B, 784)
    with torch.no_grad():
        reconstructed = infer_model(images_flat) #[b, 784] 
    
    n=images.shape[0]  # 批次大小
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray') #! squeeze 去掉维度为1的 (1, 28, 28)->(28, 28)
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].view(28, 28).cpu().numpy(), cmap='gray')  #(784) -> (28, 28)
        axes[1, i].axis('off')
    axes[0, 0].set_title('原图', fontsize=10)
    axes[1, 0].set_title('重建', fontsize=10)
    plt.tight_layout()
    plt.savefig('ae_reconstruction.png', dpi=150)
    logger.info('推理验证完成，结果已保存到 ae_reconstruction.png')