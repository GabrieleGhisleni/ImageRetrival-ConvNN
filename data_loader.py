import sys
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader(train_dir, test_dir):
    data_transforms = {'train': transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(root=test_dir,
                                   transform=data_transforms['test'])
    train_loader = DataLoader(train_dataset, batch_size=32,
                              num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             num_workers=2,shuffle=False)
    return (train_loader, test_loader)