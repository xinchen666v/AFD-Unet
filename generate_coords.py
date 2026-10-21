import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入你原有的代码
# 请确保 afd_pet_v3.py 和这个脚本在同一个目录下
from afd_pet_v3 import CrystalImageProcessor, DetectionConfig 

# ================= 配置路径 =================
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
# 存放 36 个坐标的文件夹
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
os.makedirs(COORD_DIR, exist_ok=True)

def generate_all_coords():
    # 初始化你原来的处理器
    processor = CrystalImageProcessor(DetectionConfig())
    
    npy_files = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy')]
    print(f"开始为 {len(npy_files)} 张图生成坐标标签...")

    for fname in tqdm(npy_files):
        img_path = os.path.join(NPY_DIR, fname)
        # 读取我们转好的 float32 矩阵
        img = np.load(img_path) 
        
        # ================= 适配你原代码的关键 =================
        # 你的原代码是用 PIL 读图并归一化到 0~1 的。因为 MATLAB 的 double 矩阵
        # 本身可能已经归一化，或者最大值是几千。为了保证算法生效，我们强制缩放到 0~1
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img
        
        # 你的 processor.process() 原本是接收 image_path 的。
        # 为了不改动你原有代码，可以临时存成 PNG 跑，或者改写原代码的 process 接受 numpy array。
        # 这里最稳妥的方法：你的原代码是读取灰度图的，我们将 img_norm 乘 255 存成临时 PNG 跑。
        tmp_png = "temp_for_coord.png"
        Image.fromarray((img_norm * 255).astype(np.uint8)).save(tmp_png)
        
        try:
            # 跑你原来的检测算法
            points, _, log, _ = processor.process(tmp_png)
            
            # 提取 Q 值，你可以根据 Q 值过滤掉不准的图
            Qs = [info['quality']['Q'] for info in log]
            mean_Q = np.mean(Qs)
            
            # 保存坐标 [36, 2]
            coord_name = fname.replace('.npy', '_coords.npy')
            np.save(os.path.join(COORD_DIR, coord_name), points)
            
            # 如果 Q 值太低，打印一下提醒你
            if mean_Q < 0.8:
                print(f"\n注意: {fname} 的平均 Q 值为 {mean_Q:.2f}，检测可能不准。")
                
        except Exception as e:
            print(f"\n处理 {fname} 失败: {e}")
            
    # 清理临时文件
    if os.path.exists("temp_for_coord.png"):
        os.remove("temp_for_coord.png")

    print(f"\n坐标生成完毕！保存在: {COORD_DIR}")

if __name__ == "__main__":
    generate_all_coords()