import torch

#single layer naural network

def activation(x):
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,5)) # 1 * 5
weight = torch.rand_like(features)
bias = torch.randn((1,1))

sum_act = activation(torch.sum(features * weight) + bias)
mm_act = activation(torch.mm(features, weight.view(5, 1)) + bias)
matual_act = activation(torch.matmul(features, weight.view(5,1)) + bias)

## 4. Networks Using Matrix Multiplization

n_input = features.shape[1]
n_hieeden = 2
n_output = 1

W1 = torch.randn(n_input, n_hieeden) #5 * 2
W2 = torch.randn(n_hieeden, n_output) #2 * 1

B1 = torch.randn((1, n_hieeden))
B2 = torch.randn((1, n_output))


hidden_layer = activation(torch.matmul(features, W1) + B1)
output_layer = activation(torch.matmul(hidden_layer, W2) + B2)

# 6. Neural Networks in Pytorch
from torchvision import datasets, transforms
import torch.utils.data
#defin a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
                                 ])

train_set = datasets.MNIST("MNIST_data/", download= True, train = True, transform= transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle= True)

data_iter = iter(train_loader)
image, labels = data_iter.next()

input = image.view(image.shape[0], -1)

W1 = torch.randn(784, 256)
W2 = torch.randn(256, 10)

B1 = torch.randn(256)
B2 = torch.randn(10)

hidden_layer = activation(torch.matmul(input, W1) + B1)
output_layer2 = activation(torch.matmul(hidden_layer, W2) + B2)

#print(output_layer2)

