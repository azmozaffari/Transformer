# get batch of inputs and then patchyfy them on GPU

import torch
from PIL import Image
from torchvision import transforms
from model import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR


num_epochs = 500
batch_size = 256
patch_size = 7
rate_learning = 0.0001



configs = {
    "n_patches": 16+1,
    "n_block": 3,
    "num_heads": 8,
    "p": 0.2,
    "bias": True,
    "hidden_features": 256,
    "in_channels":1,
    "emb_size":512,
    "patch_size" :patch_size,
    "n_classes": 10
}




# read the dataset MNIST
transforms = Compose(
    [
        Resize((28,28)),
        ToTensor()
    ]
)

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform= transforms, 
    download = True,            
)




loaders = {'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size, 
                                          shuffle=True, 
                                          num_workers=1)}




model = VIT(**configs).cuda()
loss_func = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=rate_learning, momentum=0.9)   
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)



total_step = len(loaders['train'])
for epoch in range(num_epochs):
    loss_ = torch.zeros(1).cuda()
    if epoch <40:
        scheduler.step()
    for i, (images, labels) in enumerate(loaders['train']):
        labels = labels.to('cuda')
        # gives batch data, normalize x when iterate train_loader
        images = images.to('cuda')
        y = model(images)

        loss = loss_func(y, labels)
        loss_ = loss_+loss  
        # clear gradients for this training step   
        optimizer.zero_grad()           
        
        # backpropagation, compute gradients 
        loss.backward()                # apply gradients             
        optimizer.step()                
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss_.item()))
            # print(loss.grad)

torch.save(model, './model/vit')



