import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import itertools
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder

train_root_path = ''
test_root_path = ''
model_save_name = 'models/'

class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Sequential(
                nn.Conv2d(n_features, n_features, kernel_size=3),
                nn.BatchNorm2d(n_features),
                nn.ReLU()),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Sequential(
                nn.Conv2d(n_features, n_features, kernel_size=3),
                nn.BatchNorm2d(n_features))
        )

    def forward(self, input):
        return input + self.layers(input)

class G(nn.Module)
    def __init__(self, in_channels, out_channels, n_resnet_blocks = 6, n_filters=64):
        super(G, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, n_filters, kernel_size=7),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )

        self.downsampling_layer = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(),
        )

        blocks = []
        for _ in range(n_resnet_blocks):
            blocks.append(ResNetBlock(n_filters * 4))

        self.residual_blocks = nn.Sequential(*blocks)

        self.upsampling_layer = nn.Sequential(
            nn.Conv2d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, out_channels, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.initial_layer(input)
        output = self.downsampling_layer(output)
        output = self.residual_blocks(output)
        output = self.upsampling_layer(output)
        output = self.output_layer(output)
        return output


class D(nn.Module):
    def __init__(self, in_channels):
        super(D, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3),
        )
        self.classifier = nn.Conv2d(512, 1,kernel_size=4, padding=1)

    def forward(self, input):
        output = self.input_conv(input)
        output = self.conv_layers(output)
        output = self.classifier(output)
        return output


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataloader_A = DataLoader(ImageFolder(train_root_path + '/A', train_transforms), batch_size=64, num_workers=4)
dataloader_B = DataLoader(ImageFolder(train_root_path + '/B', train_transforms), batch_size=64, num_workers=4)
def train(G_A, G_B, D_A, D_B, epochs=20, is_need_GAN=True):
    GANLoss = nn.MSELoss().cuda()
    CycleLoss = nn.L1Loss().cuda()
    optimizer_G = torch.optim.Adam(itertools.chain(G_A.params(), G_B.params()), lr=0.0002, betas=(0.5, 0.99))
    optimizer_DA = torch.optim.Adam(D_A.params(), lr=0.0002, betas=(0.5, 0.99))
    optimizer_DB = torch.optim.Adam(D_B.params(), lr=0.0002, betas=(0.5, 0.99))
    for epoch_num in range(epochs):
        loss_trace = []
        for i, batch in tqdm(enumerate(zip(dataloader_A, dataloader_B))):
            A, B = batch
            optimizer_G.zero_grad()
            optimizer_DA.zero_grad()
            optimizer_DB.zero_grad()

            loss_A2B, loss_B2A = 0, 0
            gen_B = G_B(A)
            gen_A = G_A(B)

            if is_need_GAN:
                discriminate_A2B = D_B(gen_B)
                discriminate_B2A = D_A(gen_A)
                loss_A2B = GANLoss(discriminate_A2B, torch.ones_like(discriminate_A2B))
                loss_B2A = GANLoss(discriminate_B2A, torch.ones_like(discriminate_B2A))

            gen_BA = G_A(gen_B)
            gen_AB = G_B(gen_A)

            loss_A2B2A = CycleLoss(gen_BA, A) * 10
            loss_B2A2B = CycleLoss(gen_AB, B) * 10

            loss_G = loss_A2B + loss_B2A + loss_A2B2A + loss_B2A2B
            print("epoch: {}, iter: {}, loss_G: {}", epoch_num, i, loss_G)
            loss_trace.append(loss_G)
            loss_G.backward()
            optimizer_G.step()

            if is_need_GAN:
                discriminate_A = D_A(A)
                discriminate_B = D_B(B)
                discriminate_ABA = D_A(gen_BA)
                discriminate_BAB = D_B(gen_AB)
                loss_D_A = GANLoss(discriminate_A, torch.ones_like(discriminate_A)) + \
                           GANLoss(discriminate_ABA, torch.zeros_like(discriminate_ABA))
                loss_D_B = GANLoss(discriminate_B, torch.ones_like(discriminate_B)) + \
                           GANLoss(discriminate_BAB, torch.zeros_like(discriminate_BAB))
                loss_D_A.backward()
                print("epoch: {}, iter: {}, loss_D_A: {}, loss_D_B", epoch_num, i, loss_D_A, loss_D_B)
                optimizer_DA.step()
                loss_D_B.backward()
                optimizer_DB.step()

        f = plt.figure()
        plt.plot(len(loss_trace), loss_trace)
        plt.xlabel('iter_num')
        plt.ylabel('Generator Loss')
        plt.title('epochs num {}'.format(epoch_num))
        plt.grid()
        plt.show()
        f.savefig('plots/{}.pdf'.format(epoch_num))
        print('Loss at epoch {}:'.format(epoch_num), loss_trace[-1])


test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataloader = DataLoader(ImageFolder(test_root_path, test_transforms), batch_size=1, num_workers=4)
def test(G_A, G_B):
    B_output = 'generated/B/'
    A_output = 'generated/A/'

    if not os.path.exists(B_output):
        os.mkdir(B_output)
    if not os.path.exists(A_output):
        os.mkdir(A_output)

    for i, batch in enumerate(test_dataloader):
        gen_B = G_B(batch[0])
        plt.imsave('B_output/{}'.format(i), gen_B)
        gen_A = G_A(gen_B)
        plt.imsave('A_output/{}'.format(i), gen_A)


if __name__ == '__main__':
    G_A = G(3, 3)
    G_B = G(3, 3)
    D_A = D(3)
    D_B = D(3)
    train(G_A, G_B, D_A, D_B, is_need_GAN=False)
    torch.save(G_A.state_dict(), model_save_name + 'G_A')
    torch.save(G_B.state_dict(), model_save_name + 'G_B')
    torch.save(D_A.state_dict(), model_save_name + 'D_A')
    torch.save(D_B.state_dict(), model_save_name + 'D_B')
    G_A.eval()
    G_B.eval()
    test(G_A, G_B)