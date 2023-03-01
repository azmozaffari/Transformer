import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F


patch_size = 7
batch_size = 100

transforms = Compose(
    [
        Resize((28,28)),
        ToTensor()
    ]
)


test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms
)


loaders = {
   
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
}


vit = torch.load('./model/vit')
vit.eval()
test_loss = 0
correct = 0
test_losses = []
with torch.no_grad():
    for data, target in loaders['test']:
        data = data.cuda()
        target = target.cuda()
        output = vit(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loaders['test'].dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(loaders['test'].dataset),
    100. * correct / len(loaders['test'].dataset)))