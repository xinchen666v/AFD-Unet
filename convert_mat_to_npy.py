import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# ================= 配置路径 =================
# 你的原始 .mat 文件夹路径
# MAT_DIR = r"D:\PyProject\AFD-Unet\data\raw\floodmap"
MAT_DIR = r"D:\PyProject\AFD-Unet\test\floodmap\floodmap_1_2"
# 转换后存放 .npy 的文件夹路径（建议新建一个 npy 文件夹，和 raw 平级）
# NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
NPY_DIR = r"D:\PyProject\AFD-Unet\test\floodmap-npy1-2"

# 如果输出目录不存在，自动创建
os.makedirs(NPY_DIR, exist_ok=True)

def convert_all():
    # 找出所有 .mat 文件
    mat_files = [f for f in os.listdir(MAT_DIR) if f.endswith('.mat')]
    print(f"共找到 {len(mat_files)} 个 .mat 文件，开始转换...")

    for fname in tqdm(mat_files):
        mat_path = os.path.join(MAT_DIR, fname)
        
        # 构造输出的 .npy 文件名 (例如 floodmap_1_1.mat -> floodmap_1_1.npy)
        npy_name = fname.replace('.mat', '.npy')
        npy_path = os.path.join(NPY_DIR, npy_name)

        try:
            # 1. 读取 .mat 文件
            mat_data = sio.loadmat(mat_path)
            
            # 2. 提取变量名为 'flood_map' 的矩阵
            if 'flood_map' not in mat_data:
                print(f"\n警告: {fname} 中没有找到 'flood_map' 变量，跳过。")
                continue
                
            img = mat_data['flood_map']
            
            # 3. 类型转换与清理
            img = img.astype(np.float32) # 转 float32 省空间
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0) # 清理 NaN 和 Inf
            
            # 4. 保存为 .npy
            np.save(npy_path, img)
            
        except Exception as e:
            print(f"\n处理 {fname} 时出错: {e}")

    print(f"\n转换完成！文件已保存至: {NPY_DIR}")

if __name__ == "__main__":
    convert_all()