"""
AFD-PET v3
(Adaptive Floodmap Decoder for PET)
自适应晶体光斑检测模块
核心功能：根据光斑粘连程度自动选择检测方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, center_of_mass
from PIL import Image
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist


# ==================== 配置参数 ====================

class DetectionConfig:
    """检测参数配置"""
    # 网格设置
    GRID_SIZE = 3           # 3x3 网格
    ROI_SIZE = 70           # ROI半宽（实际120x120）
    # MARGIN = 60             # 边界留白

    # === 新增：自动边界探测参数 ===
    AUTO_BLUR_SIGMA = 10.0      # 用于全图模糊以寻找边界的强度
    AUTO_THRESHOLD_RATIO = 0.2  # 判定为有信号区域的亮度阈值
    
    # 自适应sigma搜索范围
    SIGMA_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    PEAK_THRESHOLD_HIGH = 0.15   # sigma <= 2时的阈值
    PEAK_THRESHOLD_LOW = 0.1     # sigma > 2时的阈值
    MIN_DISTANCE = 6             # 峰值最小间距
    
    # 方法选择阈值
    KMEANS_MIN_PEAKS = 4     # 使用K-Means的最小峰值数
    
    # 动态质心参数
    CENTROID_SIGMA = 4.0     # 强平滑sigma
    CENTROID_MASK_THRESHOLD = 0.3
    STEP_MIN = 8
    STEP_MAX = 20
    STEP_FACTOR = 0.8        # 标准差乘数
    
    # 局部精修参数
    REFINE_WINDOW = 7        # 7x7窗口
    REFINE_BLEND = 0.7       # 精修结果权重

    # 质量控制
    Q_RETRY_THRESHOLD = 0.80  # 质量评分低于此值时重试


'''
v3 更新日志：
- 新增评估模块，自动分割后给出置信度
'''
class QualityEvaluator:
    def evaluate_roi(self, img, points, info):
        """
        返回：
            dict:
              geo_score
              intensity_score
              spacing_score
              method_confidence
              Q (综合评分)
        """
        geo = geometry_consistency(points)
        inten = intensity_consistency(img, points)
        mconf = method_confidence(info)

        Q = combine_scores(geo, inten, mconf)

        return {
            "geo_score": geo,
            "intensity_score": inten,
            "method_confidence": mconf,
            "Q": Q
        }

def geometry_consistency(points):
        """
        points: (4,2) 已按 左上 右上 左下 右下 排序
        """
        p0, p1, p2, p3 = points

        dx = np.array([
            abs(p1[0] - p0[0]),
            abs(p3[0] - p2[0])
        ])
        dy = np.array([
            abs(p2[1] - p0[1]),
            abs(p3[1] - p1[1])
        ])

        dx_mean = np.mean(dx) + 1e-6
        dy_mean = np.mean(dy) + 1e-6

        geo_err = np.std(dx) / dx_mean + np.std(dy) / dy_mean
        geo_score = np.exp(-geo_err)   # 映射到 (0,1]

        return geo_score
    
def intensity_consistency(img, points, win=5):
        h, w = img.shape
        half = win // 2
        scores = []

        for x, y in points:
            ix, iy = int(x), int(y)
            x1 = max(0, ix - half)
            x2 = min(w, ix + half + 1)
            y1 = max(0, iy - half)
            y2 = min(h, iy + half + 1)

            local = img[y1:y2, x1:x2]
            scores.append(np.mean(local))

        scores = np.array(scores)
        score = np.mean(scores) / (np.mean(img) + 1e-6)
        return np.clip(score / 1.5, 0, 1)  # 经验归一
        
def method_confidence(info):
        if 'kmeans' in info['method'] and info['peak_count'] == 4:
            return 1.0
        if info['method'] == 'centroid':
            return 0.7
        return 0.4
    
def combine_scores(geo, inten, method):
        w_geo, w_int, w_m = 0.5, 0.3, 0.2
        Q = w_geo * geo + w_int * inten + w_m * method
        return Q

# ==================== 核心检测器 ====================

class AdaptiveSpotDetector:
    """
    自适应光斑检测器
    自动选择最优检测方法：K-Means（分离好）或动态质心（粘连）
    """
    
    def __init__(self, config=None):
        self.config = config or DetectionConfig()
        self.evaluator = QualityEvaluator()
    
    def detect(self, roi, offset_x=0, offset_y=0):
        """
        主检测接口
        
        Returns:
            points: (4, 2) 检测到的4个点坐标
            info: dict 检测信息（方法、sigma、峰值数等）
        """
        # Step 1: 自适应寻找最优平滑参数
        sigma, peaks, peak_count = self._find_optimal_sigma(roi)
        
        # Step 2: 根据峰值数量选择检测方法
        if peak_count >= self.config.KMEANS_MIN_PEAKS:
            points, method = self._detect_by_kmeans(roi, peaks, sigma, offset_x, offset_y)
        else:
            points, method = self._detect_by_centroid(roi, offset_x, offset_y)
        
        info = {
            'method': method,
            'sigma': sigma,
            'peak_count': peak_count,
            'output_points': len(points)
        }
        
        return points, info
    
    # def _find_optimal_sigma(self, roi):
    #     """
    #     自适应寻找最优高斯平滑参数
        
    #     策略：找到能检测出4个峰值的最小sigma，否则选最接近的
    #     """
    #     best_sigma = 2.0
    #     best_peaks = None
    #     best_count = 0
        
    #     for sigma in self.config.SIGMA_CANDIDATES:
    #         smoothed = gaussian_filter(roi.astype(float), sigma=sigma)
            
    #         # 保守策略：高阈值确保只有真正分离的光斑才能触发kmeans
    #         # 宁可peaks<4用centroid，也不让粘连光斑进入kmeans
    #         # 经验值：0.15经过验证，能过滤大部分噪声和粘连假象
    #         threshold = (self.config.PEAK_THRESHOLD_HIGH 
    #                     if sigma <= 2 else self.config.PEAK_THRESHOLD_LOW)

            
    #         peaks = peak_local_max(
    #             smoothed, 
    #             min_distance=self.config.MIN_DISTANCE,
    #             threshold_rel=threshold
    #         )
            
    #         # 优先选恰好4个峰值的最小sigma
    #         if len(peaks) == 4 and best_count != 4:
    #             return sigma, peaks, 4
    #         # 否则选最接近4个的
    #         elif abs(len(peaks) - 4) < abs(best_count - 4):
    #             best_sigma = sigma
    #             best_peaks = peaks
    #             best_count = len(peaks)

    #         # print(f"  sigma={sigma}: peaks={len(peaks)}")
        
    #     return best_sigma, best_peaks, best_count
    def _find_optimal_sigma(self, roi):
        best_sigma = 2.0
        best_peaks = None
        best_count = 0
        
        for sigma in self.config.SIGMA_CANDIDATES:
            smoothed = gaussian_filter(roi.astype(float), sigma=sigma)
            threshold = (self.config.PEAK_THRESHOLD_HIGH if sigma <= 2 else self.config.PEAK_THRESHOLD_LOW)

            peaks = peak_local_max(
                smoothed, 
                min_distance=self.config.MIN_DISTANCE,
                threshold_rel=threshold
            )
            
            current_count = len(peaks)
            
            # --- 关键优化：严格的质量预审 ---
            if current_count == 4:
                peak_values = smoothed[peaks[:, 0], peaks[:, 1]]
                # 亮度差异检测：最亮不能超过最暗的 3 倍 (经验值)
                if np.max(peak_values) > 3.0 * np.min(peak_values):
                    current_count = 1  # 强制降级，诱导逻辑走向 centroid
                
                # 变异系数检测：CV > 0.4 通常意味着包含噪点
                cv = np.std(peak_values) / (np.mean(peak_values) + 1e-6)
                if cv > 0.4:
                    current_count = 1

            # 只有通过质量预审的 count 才会参与最优选择
            if current_count == 4 and best_count != 4:
                return sigma, peaks, 4
            elif abs(current_count - 4) < abs(best_count - 4):
                best_sigma = sigma
                best_peaks = peaks
                best_count = current_count
        
        return best_sigma, best_peaks, best_count
    
    def _detect_by_kmeans(self, roi, peaks, sigma, offset_x, offset_y):
        """
        K-Means检测：适合分离良好的光斑
        
        使用亮度加权的K-Means聚类
        """

        h, w = roi.shape
        
        # 预处理：背景抑制 + 平滑
        roi_norm = roi - np.mean(roi)
        roi_smooth = gaussian_filter(roi_norm, sigma)
        
        # 找局部极大值作为候选点
        size = 15  # 固定窗口大小，经验值
        local_max = (roi_smooth == maximum_filter(roi_smooth, size))

        # 自适应窗口：基于峰值分布的密度
        # if len(peaks) >= 2:
        #     # 计算峰值间的平均距离

        #     avg_spacing = np.mean(pdist(peaks))
        #     size = max(5, int(avg_spacing * 0.8))  # 经验系数 0.8
        # else:
        #     size = 15  # 回退默认值

        # --- 稳定的自适应窗口 ---
        # 基于平滑尺度（主导）
        # size_sigma = int(6 * sigma)

        # # 基于最近邻峰距（辅助）
        # if len(peaks) >= 2:
        #     from scipy.spatial import KDTree
        #     tree = KDTree(peaks)
        #     dists, _ = tree.query(peaks, k=2)
        #     nn_dist = np.mean(dists[:, 1])
        #     size_peak = int(nn_dist * 0.6)
        # else:
        #     size_peak = size_sigma

        # # 融合 + 裁剪
        # size = int(np.clip(0.7 * size_sigma + 0.3 * size_peak, 9, 25))
        # size = size = 18
        # local_max = (roi_smooth == maximum_filter(roi_smooth, size=size))
        
        threshold = np.max(roi_smooth) * 0.1
        y_coords, x_coords = np.where(local_max & (roi_smooth > threshold))
        
        candidates = np.column_stack([x_coords, y_coords])
        
        # 候选点不足，回退到动态质心
        if len(candidates) < 4:
            if __name__ == "__main__":
                print(f"Not enough candidates for K-Means (found {len(candidates)}),size={size}, falling back to centroid.")
            return self._detect_by_centroid(roi, offset_x, offset_y)
        
        # 亮度加权
        weights = roi_smooth[y_coords, x_coords]
        
        # 多于4个点，取前4个最亮的
        if len(candidates) > 4:
            top4_idx = np.argsort(weights)[-4:]
            candidates = candidates[top4_idx]
            weights = weights[top4_idx]
        
        # K-Means聚类
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(candidates, sample_weight=weights / np.sum(weights))
        centers = kmeans.cluster_centers_
        
        # 排序为2x2网格并精修
        points = self._sort_and_refine(roi, centers, offset_x, offset_y)
        
        return points, f"kmeans_sigma{sigma}, candidates={len(candidates)}, size={size}"
    
    def _detect_by_centroid(self, roi, offset_x, offset_y):
        """
        动态质心检测：适合粘连严重的光斑
        
        基于整体亮度分布的几何中心自适应分裂
        """
        h, w = roi.shape
        cfg = self.config
        
        # 强平滑融合光斑
        smoothed = gaussian_filter(roi.astype(float), sigma=cfg.CENTROID_SIGMA)
        mask = smoothed > (np.max(smoothed) * cfg.CENTROID_MASK_THRESHOLD)
        
        # 计算质心和散布范围
        if not np.any(mask):
            cx, cy = w / 2, h / 2
            dx = dy = 15
        else:
            cy, cx = center_of_mass(mask)
            y_idx, x_idx = np.where(mask)
            dx = np.clip(np.std(x_idx) * cfg.STEP_FACTOR, cfg.STEP_MIN, cfg.STEP_MAX)
            dy = np.clip(np.std(y_idx) * cfg.STEP_FACTOR, cfg.STEP_MIN, cfg.STEP_MAX)
        
        # 生成2x2网格种子点
        centers = np.array([
            [cx - dx, cy - dy],
            [cx + dx, cy - dy],
            [cx - dx, cy + dy],
            [cx + dx, cy + dy]
        ])
        
        # 排序并精修
        points = self._sort_and_refine(roi, centers, offset_x, offset_y, is_centroid=True)
        
        return points, "centroid"
    
    def _sort_and_refine(self, roi, centers, offset_x, offset_y, is_centroid=False):
        """
        排序为2x2网格并进行局部精修
        
        排序：左上、右上、左下、右下
        精修：7x7窗口加权质心
        """
        h, w = roi.shape
        cfg = self.config
        
        # 按Y坐标排序，再按X排序，确保2x2结构
        centers = centers[np.argsort(centers[:, 1])]
        if len(centers) >= 4:
            top_two = centers[:2][np.argsort(centers[:2, 0])]
            bot_two = centers[2:4][np.argsort(centers[2:4, 0])]
            centers = np.vstack([top_two, bot_two])
        
        final_points = []
        for cx, cy in centers:
            # 提取7x7局部窗口
            x1 = max(0, int(cx) - cfg.REFINE_WINDOW)
            x2 = min(w, int(cx) + cfg.REFINE_WINDOW + 1)
            y1 = max(0, int(cy) - cfg.REFINE_WINDOW)
            y2 = min(h, int(cy) + cfg.REFINE_WINDOW + 1)
            
            local_roi = roi[y1:y2, x1:x2]
            
            if local_roi.size == 0 or np.sum(local_roi) <= 0:
                final_points.append([cx + offset_x, cy + offset_y])
                continue
            
            # 背景抑制
            local_norm = np.clip(local_roi - np.mean(local_roi), 0, None)
            
            # 加权质心精修
            Y, X = np.mgrid[0:local_norm.shape[0], 0:local_norm.shape[1]]
            refined_x = x1 + np.sum(X * local_norm) / np.sum(local_norm)
            refined_y = y1 + np.sum(Y * local_norm) / np.sum(local_norm)
            
            # 融合：精修结果 + 原值
            final_x = cfg.REFINE_BLEND * refined_x + (1 - cfg.REFINE_BLEND) * cx
            final_y = cfg.REFINE_BLEND * refined_y + (1 - cfg.REFINE_BLEND) * cy
            
            # 暗区回退（仅动态质心方法）
            if is_centroid:
                ix, iy = int(np.clip(final_x, 0, w-1)), int(np.clip(final_y, 0, h-1))
                if roi[iy, ix] < np.mean(roi):
                    final_x = final_x * 0.9 + cx * 0.1
                    final_y = final_y * 0.9 + cy * 0.1
            
            final_points.append([final_x + offset_x, final_y + offset_y])
        
        return np.array(final_points)
    


# ==================== 晶体图像处理器 ====================
class CrystalImageProcessor:
    """晶体图像处理器：具备自动边界探测功能"""
    def __init__(self, config=None):
        self.config = config or DetectionConfig()
        self.detector = AdaptiveSpotDetector(self.config)
        self.evaluator = QualityEvaluator()
    
    def _get_auto_bounds(self, img):
        """核心改进：自动计算晶体阵列的包络范围"""
        # 1. 深度平滑以消除点干扰，提取整体连通域
        blur = gaussian_filter(img, sigma=self.config.AUTO_BLUR_SIGMA)
        # 2. 生成掩膜（基于全局亮度的20%作为背景切断点）
        mask = blur > (np.max(blur) * self.config.AUTO_THRESHOLD_RATIO)
        y_idx, x_idx = np.where(mask)
        
        if len(x_idx) == 0:
            return 0, img.shape[1], 0, img.shape[0]
        
        # 3. 获取有信号区域的边界
        return np.min(x_idx), np.max(x_idx), np.min(y_idx), np.max(y_idx)

    def process(self, image_path):
        img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
        h, w = img.shape
        
        # --- 自动对齐逻辑 ---
        x_min, x_max, y_min, y_max = self._get_auto_bounds(img)
        if __name__ == "__main__":
            print(f"Auto-Bounds Detected: X({x_min}-{x_max}), Y({y_min}-{y_max})")

        # 动态分配 3x3 中心点，确保第一个点和最后一个点在信号区域内适当收缩
        # 收缩 ROI_SIZE / 2 是为了让 ROI 框尽可能对称地包住边缘的光斑
        inset = self.config.ROI_SIZE * 0.5
        x_pos = np.linspace(x_min + inset, x_max - inset, self.config.GRID_SIZE)
        y_pos = np.linspace(y_min + inset, y_max - inset, self.config.GRID_SIZE)
        # ------------------
        
        all_points = []
        log = []
        
        for i, cy in enumerate(y_pos):
            for j, cx in enumerate(x_pos):
                roi, ox, oy = self._extract_roi_with_offset(img, cx, cy)
                if roi is None: continue
                
                # points, info = self.detector.detect(roi, ox, oy)

                # # ===== 自动质量评估（v3 核心）=====
                # quality = self.evaluator.evaluate_roi(img, points, info)
                # info['quality'] = quality
                # # ==================================

                points, info = self.detector.detect(roi, ox, oy)

                quality = self.evaluator.evaluate_roi(img, points, info)
                info['quality'] = quality

                # ===== 新增：低质量ROI自动使用动态质心重跑 =====
                if quality['Q'] < self.config.Q_RETRY_THRESHOLD:
                    # if __name__ == "__main__":
                    print(f"  ↳ Q={quality['Q']:.2f} < {self.config.Q_RETRY_THRESHOLD}, retry with centroid")

                    # 强制用 centroid
                    retry_points, retry_method = self.detector._detect_by_centroid(roi, ox, oy)

                    retry_info = {
                        'method': retry_method + "_retry",
                        'sigma': info['sigma'],
                        'peak_count': info['peak_count'],
                        'output_points': len(retry_points)
                    }

                    retry_quality = self.evaluator.evaluate_roi(img, retry_points, retry_info)

                    # 如果重跑更好，则替换
                    if retry_quality['Q'] > quality['Q']:
                        points = retry_points
                        info = retry_info
                        info['quality'] = retry_quality
                    else:
                        info['quality'] = quality
                # ===============================================

                all_points.extend(points)
                info['cell'] = (i, j); info['center'] = (cx, cy)
                log.append(info)
                if __name__ == "__main__":
                    self._print_cell_result(i, j, info)


        Qs = [info['quality']['Q'] for info in log]
        if __name__ == "__main__":
            print(
                f"\n[Quality Summary] "
                f"mean={np.mean(Qs):.3f}, "
                f"min={np.min(Qs):.3f}, "
                f"bad(<0.85)={sum(q < 0.85 for q in Qs)}"
            )

        
        return np.array(all_points), img, log, (x_pos, y_pos)

    def _extract_roi_with_offset(self, img, cx, cy):
        """带安全偏移补偿的ROI提取"""
        cfg = self.config
        h, w = img.shape
        
        x1 = int(max(0, cx - cfg.ROI_SIZE))
        x2 = int(min(w, cx + cfg.ROI_SIZE))
        y1 = int(max(0, cy - cfg.ROI_SIZE))
        y2 = int(min(h, cy + cfg.ROI_SIZE))
        
        roi = img[y1:y2, x1:x2]
        return (roi, x1, y1) if roi.size > 0 else (None, 0, 0)


    # ==================== 日志打印 ====================
    '''
    增加了质量评分Q的打印
    便于用户了解每个ROI的检测置信度
    '''
    def _print_cell_result(self, i, j, info):
        q = info.get('quality', {}).get('Q', None)
        q_str = f"Q={q:.2f}" if q is not None else "Q=NA"
        
        print(
            f"Cell({i},{j}): "
            f"σ={info['sigma']:.1f}, "
            f"peaks={info['peak_count']}, "
            f"{info['method']:15s}, "
            f"{q_str}"
        )


# ==================== 可视化更新 ====================

def visualize_result(img, points, log, grid_pos, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    x_pos, y_pos = grid_pos
    cfg = DetectionConfig()

    for info in log:
        i, j = info['cell']
        cx, cy = info['center']
        
        # 计算 ROI 的左上角坐标 (x1, y1)
        x1 = cx - cfg.ROI_SIZE
        y1 = cy - cfg.ROI_SIZE
        
        # 绘制 ROI 边界 (保持不变)
        rect = plt.Rectangle((x1, y1), 
                             cfg.ROI_SIZE*2, cfg.ROI_SIZE*2, 
                             fill=False, edgecolor='cyan', linestyle='--', alpha=0.5)
        ax.add_patch(rect)
        
        # --- 修改位置：将文字放在 ROI 框的左上角上方 ---
        color = 'lime' if 'kmeans' in info['method'] else 'orange'
        method_name = info['method'].split('_')[0] # 简写：只显示 kmeans 或 centroid
        
        q = info['quality']['Q']
        # y1 - 2 确保文字稍微悬浮在框线垂直上方，不遮挡内部
        ax.text(x1, y1 - 2, 
                f"[{i},{j}] {method_name} Q={q:.2f}", 
                color=color,
                fontsize=8, ha='left', va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, lw=0, pad=1))

    # 绘制检测点 (红叉)
    ax.scatter(points[:, 0], points[:, 1], c='red', s=40, marker='x', linewidths=1.5)
    
    plt.title(f"Auto-Aligned Detection ({len(points)} pts)")
    plt.axis('off')
    
    if save_path: 
        # 使用 bbox_inches='tight' 防止标签被切掉
        plt.savefig(save_path, dpi=150, bbox_inches='tight') 
    if __name__ == "__main__":
        plt.show()

    plt.close(fig)



# ==================== 主程序 ====================

if __name__ == "__main__":
    # 配置（可选：自定义参数）
    config = DetectionConfig()

    
    # 处理图像
    
    processor = CrystalImageProcessor()
    # image_path = r"D:\aaaPreprocess\250909_human6\data_250909_brain_tdc_human_256_64mv_t1\singles\floodmap_png\floodmap_a27_b06.png"
    # image_path = r"D:\matlab_projects\data_251020_brain_tdc_line_256_64mv_t1\singles\floodmap_png\floodmap_a27_b01.png"
    # image_path = r"D:\aaaPreprocess\260304_test_floodmap_lut\floodmap\floodmap_1_7_png\floodmap_a27_b04.png"
    image_path = r"D:\aaaPreprocess\250909_human6\data_250909_brain_tdc_human_256_64mv_t1\data\output260304\flood_maps_png\floodmap_a27_b02.png"


    points, img, log, grid_pos = processor.process(image_path)
    
    # 可视化
    visualize_result(img, points, log, grid_pos, save_path='res/adaptive_result.png')
    
    # 保存结果
    np.save('res/adaptive_36_points.npy', points)
    print(f"\nSaved {len(points)} points to adaptive_36_points.npy")