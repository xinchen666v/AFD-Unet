# import torch
# import torch.nn as nn
# import segmentation_models_pytorch as smp

# def get_model(device):
#     """
#     构建 U-Net 模型并加载 ImageNet 预训练权重
#     """
#     # 使用 ResNet18 或 ResNet34 都可以，ResNet18 更快，34 稍微准一点
#     model = smp.Unet(
#         encoder_name="resnet18",        # 编码器（特征提取）
#         encoder_weights="imagenet",     # 自动下载并加载预训练权重
#         in_channels=1,                  # 原始输入是灰度图（单通道）
#         classes=36,                     # 输出 36 个通道
#         activation=None                 # 回归任务（热图），输出不加 Sigmoid
#     )
    
#     # 如果你想把模型冻住前面几层（微调），可以取消下面注释
#     # for param in model.encoder.parameters():
#     #     param.requires_grad = False
    
#     return model.to(device)

# if __name__ == "__main__":
#     # 简单测试模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(device)
    
#     # 假数据测试
#     dummy_input = torch.randn(2, 1, 512, 512).to(device) # Batch Size = 2
#     output = model(dummy_input)
    
#     print(f"模型已加载至: {device}")
#     print(f"输入尺寸: {dummy_input.shape}")
#     print(f"输出尺寸: {output.shape}")  # 预期: [2, 36, 512, 512]

# import torch
# import torch.nn as nn
# import segmentation_models_pytorch as smp
# import os

# def get_model(device):
#     """
#     构建 U-Net 模型并加载本地预训练权重
#     """
#     # 指定刚才放好的本地权重路径
#     weights_path = r"D:\PyProject\AFD-Unet\weights\resnet18.pth"
    
#     # 如果文件存在，就用本地路径；如果不存在，再让它去网上下载
#     if os.path.exists(weights_path):
#         encoder_weights = weights_path
#     else:
#         encoder_weights = "imagenet" # 兜底方案

#     model = smp.Unet(
#         encoder_name="resnet18",        
#         encoder_weights=encoder_weights, # 直接传本地文件路径！
#         in_channels=1,                  
#         classes=36,                     
#         activation='sigmoid'                 
#     )
    
#     return model.to(device)

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(device)
#     print(f"模型已加载至: {device}")


import torch
import segmentation_models_pytorch as smp
import os

def get_model(device, load_pretrained=True):
    """
    load_pretrained: 
        True - 训练时使用，加载本地预训练权重
        False - 推理时使用，直接构建空模型，绝不联网！
    """
    weights_path = r"D:\PyProject\AFD-Unet\weights\resnet18.pth"
    
    if not load_pretrained:
        encoder_weights = None # 推理时直接设为 None
    elif os.path.exists(weights_path):
        encoder_weights = weights_path
    else:
        encoder_weights = "imagenet"

    model = smp.Unet(
        encoder_name="resnet18",        
        encoder_weights=encoder_weights, 
        in_channels=1,                  
        classes=36,                     
        # activation='sigmoid'  # 千万别忘了之前加的 sigmoid
        activation=None
    )
    
    return model.to(device)