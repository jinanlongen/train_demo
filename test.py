import torch
from main import CNN
from PIL import Image
import torchvision.transforms as transforms

def predict_image(image_path, model, invert_colors=True):
    """预测单个图片文件"""
    # 定义图像预处理步骤
    transform_list = [
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.Resize((28, 28)),  # 调整大小为28x28
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]范围
    ]
    
    transform = transforms.Compose(transform_list)
    
    # 加载并预处理图片
    image = Image.open(image_path)
    image_tensor = transform(image)
    
    # 如果需要反转颜色
    if invert_colors:
        image_tensor = 1 - image_tensor
    
    # 预测
    model.eval()
    with torch.no_grad():
        # 确保输入维度正确
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
            
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        
        # 打印详细信息
        print("\nPrediction probabilities:")
        for digit, prob in enumerate(probabilities[0]):
            print(f"Digit {digit}: {prob.item()*100:.2f}%")
            
        print(f"\nImage tensor shape: {image_tensor.shape}")
        print(f"Value range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
        print(f"Mean value: {image_tensor.mean():.2f}")
        
        return predicted.item()

def main():
    model = CNN()
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 预测单张自定义图片
    image_path = './data/6.png'
    predicted_digit = predict_image(image_path, model, invert_colors=True)
    print(f"\nFinal prediction: {predicted_digit}")

if __name__ == "__main__":
    main()