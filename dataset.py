# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import albumentations as A

# class FloodmapDataset(Dataset):
#     def __init__(self, img_dir, coord_dir, img_size=512, sigma=6.0, is_train=True):
#         """
#         img_dir: 原图 .npy 所在目录
#         coord_dir: 坐标 .npy 所在目录
#         img_size: 图像尺寸 (512)
#         sigma: 高斯山包的胖瘦 (建议 3.0 ~ 4.0)
#         is_train: 是否为训练集（决定是否开启数据增强）
#         """
#         self.img_dir = img_dir
#         self.coord_dir = coord_dir
#         self.img_size = img_size
#         self.sigma = sigma
        
#         # 获取所有文件名，并排序保证对应关系
#         self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        
#         # 定义数据增强 (Albumentations)
#         if is_train:
#             self.transform = A.Compose([
#                 # # 轻微平移、缩放、旋转 (模拟不同位置的光斑)
#                 # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
#                 # 新版用 Affine 替代 ShiftScaleRotate
#                 A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
#                 # 亮度对比度变化 (模拟极亮或极暗)
#                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#                 # 高斯模糊 (极其重要：模拟光斑粘连)
#                 A.GaussianBlur(blur_limit=(3, 5), p=0.3),
#                 # # 添加噪点
#                 # A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
#                 # 新版 GaussNoise 直接给 std_range 或者不给参数用默认
#                 A.GaussNoise(std_range=(0.01, 0.05), p=0.3), 
#             ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
#         else:
#             # 验证集/测试集不做增强
#             self.transform = None

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         img_path = os.path.join(self.img_dir, fname)
#         coord_path = os.path.join(self.coord_dir, fname.replace('.npy', '_coords.npy'))

#         # 1. 读取图像和坐标
#         img = np.load(img_path).astype(np.float32)
#         coords = np.load(coord_path) # 形状 [36, 2]

#         # 2. 归一化 (关键：把图像拉伸到 0~1)
#         img = img / (np.max(img) + 1e-8)
#         # 确保你的 coords 是 (x, y) 格式，而不是 (y, x)
#         augmented = self.transform(image=img, keypoints=coords)

#         # 3. 数据增强 (注意：坐标必须和图像同步变化)
#         if self.transform:
#             # Albumentations 需要图像是 uint8 才能做某些操作，或者直接传 float32
#             # 这里直接传 float32
#             augmented = self.transform(image=img, keypoints=coords)
#             img = augmented['image']
#             coords = np.array(augmented['keypoints'], dtype=np.float32)
            
#             # # 边界保护：如果增强后点被移出了图像，强行拉回边界内
#             # coords[:, 0] = np.clip(coords[:, 0], 0, self.img_size - 1)
#             # coords[:, 1] = np.clip(coords[:, 1], 0, self.img_size - 1)

#         # 4. 实时生成 36 通道高斯热图标签
#         heatmap = np.zeros((36, self.img_size, self.img_size), dtype=np.float32)
#         yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
        
#         for ch in range(36):
#             x, y = coords[ch]
#             # 二维高斯公式
#             heatmap[ch] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))

#         # 5. 转成 PyTorch Tensor
#         # img: [1, 512, 512]
#         img_tensor = torch.from_numpy(img).unsqueeze(0)
#         # heatmap: [36, 512, 512]
#         heatmap_tensor = torch.from_numpy(heatmap)

#         return img_tensor, heatmap_tensor

# # ================= 测试代码 =================
# if __name__ == "__main__":
#     # 填入你的真实路径
#     NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
#     COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
    
#     # 创建 Dataset
#     dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=True)
#     print(f"数据集大小: {len(dataset)}")
    
#     # 取第一个样本测试
#     img, hm = dataset[0]
#     print(f"图像 Tensor 形状: {img.shape}")     # 应该是 [1, 512, 512]
#     print(f"热图 Tensor 形状: {hm.shape}")      # 应该是 [36, 512, 512]
    
#     # 检查一下标签的范围
#     print(f"热图最大值: {hm.max().item():.4f}") # 应该接近 1.0
#     print(f"热图最小值: {hm.min().item():.4f}") # 应该是 0.0


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class FloodmapDataset(Dataset):
    def __init__(self, img_dir, coord_dir, img_size=512, sigma=4.0, is_train=True):
        self.img_dir = img_dir
        self.coord_dir = coord_dir
        self.img_size = img_size
        self.sigma = sigma  # sigma 用 4.0 比较适中
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        
        if is_train:
            # 只做像素级增强，绝不做旋转、平移！
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.2), 
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        coord_path = os.path.join(self.coord_dir, fname.replace('.npy', '_coords.npy'))

        img = np.load(img_path).astype(np.float32)
        coords = np.load(coord_path)
        
        # 清理异常坐标（如果你之前的检查发现有点越界，这里强制拉回）
        coords = np.clip(coords, 0, self.img_size - 1)
        coords = np.nan_to_num(coords, nan=0.0)

        # 归一化
        img = img / (np.max(img) + 1e-8)

        # 数据增强（只处理图像，不处理坐标）
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        # 生成 36 通道高斯热图
        heatmap = np.zeros((36, self.img_size, self.img_size), dtype=np.float32)
        yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
        
        for ch in range(36):
            x, y = coords[ch]
            heatmap[ch] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        heatmap_tensor = torch.from_numpy(heatmap)

        return img_tensor, heatmap_tensor