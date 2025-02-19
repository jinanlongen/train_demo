import torch
from mnist_dataset import MNISTDataset
from torch import nn
from tqdm import tqdm
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3136, 10)  # 3136 = 64 * 7 * 7
        self.relu = nn.ReLU()

    def forward(self, x):
        # 确保输入数据格式正确
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # 添加batch维度
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 正确展平，保留batch维度
        x = self.fc1(x)
        return x

def main():
    mnist_dataset = MNISTDataset()
    train_data = mnist_dataset.train_data()
    test_data = mnist_dataset.test_data()
    
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    best_accuracy = 0
    
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        # 添加进度条
        train_pbar = tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for i, (images, labels) in enumerate(train_pbar):
            # 确保标签是tensor类型并且维度正确
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            # 确保标签是一维的
            labels = labels.view(-1)
            
            # 打印形状以进行调试
            # print(f"images shape: {images.shape}, labels shape: {labels.shape}")
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 更新进度条描述
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 评估模式
        model.eval()
        correct = 0
        total = 0
        # 添加测试进度条
        test_pbar = tqdm(test_data, desc=f'Epoch {epoch+1}/{epochs} [Test]')
        with torch.no_grad():
            for images, labels in test_pbar:
                # 确保标签是tensor类型并且维度正确
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                # 确保标签是一维的
                labels = labels.view(-1)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条描述
                accuracy = 100 * correct / total
                test_pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, 'models/best_model.pth')
                print(f"Saved best model with accuracy: {accuracy:.2f}%")
    
    # 训练结束后保存最终模型
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, 'models/final_model.pth')
    print(f"Saved final model with accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()