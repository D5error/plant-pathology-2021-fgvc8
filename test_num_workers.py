"""
选择最佳的workers数量
"""

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import multiprocessing as mp
from time import time

if __name__ == "__main__":
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 将图像PIL转pytorch为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
    ])

    # 使用 ImageFolder 加载数据集
    trainset = torchvision.datasets.ImageFolder(root="./plant_dataset", transform=transform)

    print(f"num of CPU: {mp.cpu_count()}")
    
    # for num_workers in range(2, mp.cpu_count(), 2):
    for num_workers in range(0, 5, 1):
        # 初始化 DataLoader
        train_loader = DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=32, pin_memory=True)
        
        start = time()
        for i, data in enumerate(train_loader, 0):
            if i > 5: 
                break
            print(f"\r{i} / 5", end="")
        end = time()
        
        print("\rFinish with: {:.2f} seconds, num_workers={}".format(end - start, num_workers))
