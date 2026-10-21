import os
import numpy as np
import cv2
import shutil

# ================= 配置路径 =================
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
BAD_DIR = r"D:\PyProject\AFD-Unet\data\processed\bad_data"
os.makedirs(BAD_DIR, exist_ok=True)

def check_labels():
    npy_files = sorted([f for f in os.listdir(NPY_DIR) if f.endswith('.npy')])
    total = len(npy_files)
    
    print("="*50)
    print("Controls:")
    print("  [k] : Keep image and go to next")
    print("  [d] : Discard (move to bad_data) and go to next")
    print("  [q] : Quit")
    print("="*50)

    for idx, fname in enumerate(npy_files):
        img_path = os.path.join(NPY_DIR, fname)
        coord_path = os.path.join(COORD_DIR, fname.replace('.npy', '_coords.npy'))

        if not os.path.exists(coord_path):
            print(f"[{idx+1}/{total}] Skipped: No coord file for {fname}")
            continue

        # 1. 读取数据
        img = np.load(img_path)
        coords = np.load(coord_path)

        # 2. 图像格式转换 (OpenCV 需要 8-bit 格式)
        # 使用 99.5% 的百分位数作为最亮值，防止极个别极亮噪点导致整图太暗
        vmax = np.percentile(img, 99.5)
        if vmax == 0: vmax = 1.0 # 防止除0
        img_norm = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
        
        # 转成 3 通道彩色图，方便画红叉和绿字
        img_display = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        # 3. 画点和编号
        for i, (x, y) in enumerate(coords):
            # 坐标必须转 int
            pt = (int(x), int(y))
            # 画红叉 (BGR格式: (0,0,255)是红色)
            cv2.drawMarker(img_display, pt, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            # 写绿色编号 (0~35)
            cv2.putText(img_display, str(i), (pt[0]+5, pt[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # 4. 添加窗口标题信息
        info_text = f"[{idx+1}/{total}] {fname} | [k] Keep | [d] Discard | [q] Quit"
        cv2.putText(img_display, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 5. 显示窗口
        window_name = "Label Inspector (Press K/D/Q)"
        cv2.imshow(window_name, img_display)
        
        # 6. 等待键盘输入 (无限等待，直到按下一个键)
        key = cv2.waitKey(0) & 0xFF

        # 7. 根据按键执行操作
        if key == ord('d') or key == ord('D'):
            print(f"[{idx+1}/{total}] -> Moved to bad_data: {fname}")
            shutil.move(img_path, os.path.join(BAD_DIR, fname))
            shutil.move(coord_path, os.path.join(BAD_DIR, fname.replace('.npy', '_coords.npy')))
            
        elif key == ord('q') or key == ord('Q') or key == 27: # 27 是 ESC 键
            print("Exiting inspector...")
            break
            
        # 按 k 或其他键都会自动继续
        # (OpenCV 的 waitKey 在按完键后，如果继续循环，会自动更新 imshow 显示下一张)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_labels()