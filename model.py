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
        # V1 输入为单通道矩阵
        in_channels=1,       
        # # V2 在原本的矩阵后拼接X，Y两个通道
        # in_channels=3,           
        classes=36,                     
        # activation='sigmoid'  # 千万别忘了之前加的 sigmoid
        activation=None
    )
    
    return model.to(device)