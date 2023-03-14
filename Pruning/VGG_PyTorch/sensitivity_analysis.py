import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 加载 ImageNet 数据集
traindir = '/path/to/train/data'
valdir = '/path/to/validation/data'
train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]))
test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 加载预训练的 VGG-16 模型
model = models.vgg16(pretrained=True)

# 设置模型为评估模式
model.eval()

# 在测试集上进行测试
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the VGG-16 model on the ImageNet test images: %d %%' % (100 * correct / total))
