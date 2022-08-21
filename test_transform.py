import torch
from torchvision import transforms
import random

#this was an issue with torch 1.6; now fixed.

# random.seed(1)

x = torch.rand((3, 10, 10))

tf_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(4),
    transforms.ToTensor(),
])
tf_flip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
tf_rot = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
])

# Consistent x among the calls
print(x[:2, :2, :2])

# RandomRotation, RandomResizedCrop and Random HorizontalFlip changes stuff
# even if seed stays the same
# for idx in range(2):
#     torch.random.manual_seed(1)
#     print(f'Crop {idx + 1}')
#     print(tf_crop(x)[:2, :2, :2].numpy())
for idx in range(2):
    torch.random.manual_seed(1)
    print(f'Flip {idx + 1}')
    print(tf_flip(x)[:2, :2, :2].numpy())
# for idx in range(2):
#     torch.random.manual_seed(1)
#     print(f'Rotation {idx + 1}')
#     print(tf_rot(x)[:2, :2, :2].numpy())