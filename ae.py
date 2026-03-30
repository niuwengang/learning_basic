
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.network(x)




class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)
        


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def Inference(model):
    batch_size=8
    device=next(model.parameters()).device
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    #取第一个batch
    x_batch, _ = next(iter(test_loader))
    x_batch= x_batch.view(x_batch.size(0), -1).to(device) # 展平：(B,1,28,28)->(B,28*28)
    

    with torch.no_grad():
        x_batch_hat = model(x_batch)
        x_batch_hat = x_batch_hat.view(batch_size, 28, 28).cpu()
        x_batch = x_batch.view(batch_size, 28, 28).cpu()

        # 绘制原图(上排)和重建图(下排)
        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))
        for i in range(batch_size):
            # 上排：原图
            axes[0, i].imshow(x_batch[i], cmap='gray')
            axes[0, i].axis('off')


            # 下排：重建图
            axes[1, i].imshow(x_batch_hat[i], cmap='gray')
            axes[1, i].axis('off')


        plt.tight_layout()
        plt.savefig('ae_result.png')
        plt.show()








if __name__ == '__main__':
    #模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset= datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)  

    ae_model = AutoEncoder(28*28, 256, 20)
    ae_model.to(device)


    criterion=nn.BCELoss(reduction='mean')

    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

    num_epochs=10
    for epoch in range(1,num_epochs+1):
        total_loss = 0.0
        for x_batch, _ in data_loader:
            x_batch= x_batch.view(x_batch.size(0), -1).to(device) # 展平：(B,1,28,28)->(B,28*28)
            x_batch_hat = ae_model(x_batch)
            batch_loss=criterion(x_batch_hat, x_batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f'Epoch: {epoch}, Loss: {total_loss / len(data_loader.dataset)}')

    
    print('Finished Training')


    #推理
    Inference(model=ae_model)
    



  



        







