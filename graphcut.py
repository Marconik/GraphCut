"""
GraphCut 图像分割与纹理合成核心模块
=======================================

基于 Boykov-Kolmogorov 最大流/最小割算法的图像处理实现。

分割算法：
  - 使用 PyMaxflow（封装了 BK maxflow C++ 库）构建 s-t 图
  - 基于前景/背景高斯混合模型(GMM)的数据项 + 边界平滑项
  - 迭代优化直到收敛

纹理合成算法：
  - 基于 "GraphCut Textures" (Kwatra et al., SIGGRAPH 2003)
  - 通过找最优缝合路径实现无缝纹理拼接

依赖:
  pip install PyMaxflow numpy opencv-python scikit-learn
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# 最大流引擎选择
# ---------------------------------------------------------------------------

_maxflow_engine = None  # 'pymaxflow' | 'grabcut' | 'fallback'


def _init_maxflow():
    """检测可用的最大流引擎并返回引擎名称。"""
    global _maxflow_engine
    if _maxflow_engine is not None:
        return _maxflow_engine

    # 优先使用 PyMaxflow（封装了 Boykov-Kolmogorov maxflow C++ 库）
    try:
        import maxflow  # PyMaxflow 包
        _ = maxflow.GraphFloat()  # 验证可用
        _maxflow_engine = 'pymaxflow'
        return _maxflow_engine
    except ImportError:
        pass

    # 回退：使用 OpenCV 内置的 GrabCut
    _maxflow_engine = 'grabcut'
    return _maxflow_engine


# ---------------------------------------------------------------------------
# 图像分割
# ---------------------------------------------------------------------------

def segment_image(img: np.ndarray, rect: tuple,
                  max_iters: int = 5,
                  border_trim: int = 2) -> np.ndarray:
    """
    使用 GraphCut 对图像进行交互式前景分割。

    参数:
        img: BGR 图像 (H, W, 3)，uint8
        rect: 用户框选的前景区域 (x, y, w, h)，在原图坐标中
        max_iters: 最大迭代次数（GMM 估计 + GraphCut 交替）
        border_trim: 从边界收缩多少像素来构建硬约束

    返回:
        mask: 二值掩码 (H, W)，uint8。1=前景，0=背景
    """
    h, w = img.shape[:2]

    # ---- 解析矩形 ----
    x, y, rw, rh = rect
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = min(rw, w - x)
    rh = min(rh, h - y)

    engine = _init_maxflow()

    if engine == 'pymaxflow':
        return _segment_pymaxflow(img, (x, y, rw, rh), max_iters, border_trim)
    else:
        return _segment_grabcut(img, (x, y, rw, rh), max_iters)


def _segment_grabcut(img: np.ndarray, rect: tuple, max_iters: int) -> np.ndarray:
    """使用 OpenCV GrabCut 进行分割（回退方案）。"""
    x, y, rw, rh = rect
    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut 需要的矩形格式
    cv2.grabCut(img, mask, (x, y, rw, rh),
                bgd_model, fgd_model,
                max_iters, cv2.GC_INIT_WITH_RECT)

    # 将结果转换为二值掩码（0=背景, 1=前景）
    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return result


def _segment_pymaxflow(img: np.ndarray, rect: tuple,
                       max_iters: int, border_trim: int) -> np.ndarray:
    """
    使用 PyMaxflow（Boykov-Kolmogorov maxflow）的自定义 GraphCut 分割。

    图结构:
      - 每个像素对应一个图节点
      - 源边 (s→p): 像素属于背景的代价
      - 汇边 (p→t): 像素属于前景的代价
      - 邻接边 (p↔q): 相邻像素标签不一致的平滑代价

    算法流程:
      1. 根据矩形初始化硬约束（内部=可能前景，外部=可能背景）
      2. 对前景/背景像素分别拟合 GMM（K=5 个高斯分量）
      3. 根据 GMM 计算每个像素的数据项（负对数似然）
      4. 构建图并求解最小割
      5. 用分割结果重新估计 GMM，迭代至收敛
    """
    import maxflow

    h, w = img.shape[:2]
    x, y, rw, rh = rect

    # ---- 初始化掩码 ----
    # 矩形外部 = 确定背景 (0)
    # 矩形内部边界 = 确定前景 (1)
    # 其余 = 未知 (2)
    mask = np.full((h, w), 2, dtype=np.uint8)
    mask[:] = 0  # 默认背景
    mask[y + border_trim:y + rh - border_trim,
         x + border_trim:x + rw - border_trim] = 2  # 内部：未知
    # 矩形中心区域作为确定前景种子
    center_margin_y = max(rh // 6, border_trim)
    center_margin_x = max(rw // 6, border_trim)
    mask[y + center_margin_y:y + rh - center_margin_y,
         x + center_margin_x:x + rw - center_margin_x] = 1  # 中心=前景

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # ---- 颜色特征：使用 LAB + 归一化 RGB ----
    img_float = img.astype(np.float64)
    b, g, r = cv2.split(img_float)
    intensity = (b + g + r) / 3.0 + 1e-6
    features = np.dstack([
        img_lab[..., 0], img_lab[..., 1], img_lab[..., 2],
        b / intensity, g / intensity, r / intensity,
    ]).reshape(-1, 6).astype(np.float64)

    # ---- 迭代优化 ----
    gamma = 50.0       # 数据项权重
    lambda_smooth = 15.0  # 平滑项权重
    sigma_smooth = 5.0    # 平滑项中对比度的敏感度参数
    k = 5                 # GMM 分量数

    for iteration in range(max_iters):
        fg_pixels = features[mask.reshape(-1) == 1]
        bg_pixels = features[mask.reshape(-1) == 0]

        if len(fg_pixels) < 50 or len(bg_pixels) < 50:
            break

        # ---- 用 GMM 拟合前景/背景颜色分布 ----
        from sklearn.mixture import GaussianMixture

        fg_gmm = GaussianMixture(n_components=min(k, len(fg_pixels) // 50 + 1),
                                 covariance_type='full', reg_covar=1e-4,
                                 random_state=42).fit(fg_pixels)
        bg_gmm = GaussianMixture(n_components=min(k, len(bg_pixels) // 50 + 1),
                                 covariance_type='full', reg_covar=1e-4,
                                 random_state=42).fit(bg_pixels)

        # ---- 计算数据项（负对数似然） ----
        fg_logprob = fg_gmm.score_samples(features)
        bg_logprob = bg_gmm.score_samples(features)

        # 限制对数似然范围，避免数值问题
        fg_logprob = np.clip(fg_logprob, -100, 100)
        bg_logprob = np.clip(bg_logprob, -100, 100)

        # 数据项：属于前景的代价 = 不在前景的似然
        # unary_fg = -log P(pixel|fg), unary_bg = -log P(pixel|bg)
        unary_fg = -fg_logprob.reshape(h, w) * gamma
        unary_bg = -bg_logprob.reshape(h, w) * gamma

        # ---- 构建 s-t 图 ----
        g = maxflow.GraphFloat()
        node_ids = g.add_grid_nodes((h, w))

        # 添加边界权重（平滑项）
        # 水平边
        diff_h = np.sum(np.abs(img[:, 1:] - img[:, :-1]), axis=2)
        weight_h = lambda_smooth * np.exp(-diff_h ** 2 / (2 * sigma_smooth ** 2))
        # 转换为边结构
        structure = np.zeros((3, 3))
        structure[1, 2] = 1
        g.add_grid_edges(node_ids, structure=structure, weights=weight_h, symmetric=True)

        # 垂直边
        diff_v = np.sum(np.abs(img[1:, :] - img[:-1, :]), axis=2)
        weight_v = lambda_smooth * np.exp(-diff_v ** 2 / (2 * sigma_smooth ** 2))
        structure = np.zeros((3, 3))
        structure[2, 1] = 1
        g.add_grid_edges(node_ids, structure=structure, weights=weight_v, symmetric=True)

        # 添加源/汇边（数据项）
        g.add_grid_tedges(node_ids, unary_bg, unary_fg)

        # 添加硬约束
        hard_fg = np.where(mask == 1, np.inf, 0).astype(np.float64)
        hard_bg = np.where(mask == 0, np.inf, 0).astype(np.float64)
        g.add_grid_tedges(node_ids, hard_bg, hard_fg)

        # ---- 求解最大流 ----
        g.maxflow()

        # ---- 获取分割结果 ----
        new_mask = g.get_grid_segments(node_ids).astype(np.uint8)  # True=前景

        # 检查是否收敛
        changed = np.sum(new_mask != mask)
        mask = new_mask
        if changed < (h * w * 0.001):
            break

    return mask


# ---------------------------------------------------------------------------
# 纹理合成（GraphCut Textures）
# ---------------------------------------------------------------------------

def synthesize_texture(texture: np.ndarray,
                       out_width: int, out_height: int,
                       patch_size: int = 48,
                       overlap: int = 8) -> np.ndarray:
    """
    基于 GraphCut 的纹理合成。

    参考: "GraphCut Textures: Image and Video Synthesis Using Graph Cuts"
          Kwatra et al., SIGGRAPH 2003

    采用 Image Quilting 的思想：
      1. 将纹理切成固定大小的 patch
      2. 以光栅扫描顺序放置 patch
      3. 在重叠区域用 GraphCut 找最优缝合路径
      4. 沿缝合线混合两个 patch

    参数:
        texture: BGR 纹理图像 (H, W, 3)
        out_width, out_height: 输出尺寸
        patch_size: 每个 patch 的边长
        overlap: 相邻 patch 之间的重叠像素数

    返回:
        合成后的 BGR 图像 (out_height, out_width, 3)
    """
    import maxflow

    tex_h, tex_w = texture.shape[:2]

    # 确保 patch 和 overlap 合理
    patch_size = min(patch_size, tex_h, tex_w, out_height, out_width)
    overlap = min(overlap, patch_size // 4)
    step = patch_size - overlap

    # 计算网格布局
    cols = int(np.ceil((out_width - overlap) / step))
    rows = int(np.ceil((out_height - overlap) / step))

    # 调整输出尺寸
    out_height = rows * step + overlap
    out_width = cols * step + overlap

    result = np.zeros((out_height, out_width, 3), dtype=np.float64)
    result_mask = np.zeros((out_height, out_width), dtype=np.uint8)

    # 可选的 patch 起始位置
    max_ty = max(0, tex_h - patch_size)
    max_tx = max(0, tex_w - patch_size)

    for row in range(rows):
        for col in range(cols):
            out_y = row * step
            out_x = col * step

            # 当前目标区域
            patch_h = min(patch_size, out_height - out_y)
            patch_w = min(patch_size, out_width - out_x)

            # 选择最佳 patch（基于重叠区域相似度）
            if row == 0 and col == 0:
                # 第一个 patch：随机选择
                ty = np.random.randint(0, max_ty + 1) if max_ty > 0 else 0
                tx = np.random.randint(0, max_tx + 1) if max_tx > 0 else 0
            else:
                # 后续 patch：在重叠区域找最佳匹配
                ty, tx = _find_best_patch(
                    texture, result, result_mask,
                    out_y, out_x, patch_h, patch_w, overlap,
                    row, col
                )

            patch = texture[ty:ty + patch_h, tx:tx + patch_w].astype(np.float64)

            if row == 0 and col == 0:
                # 第一个 patch 直接拷贝
                result[out_y:out_y + patch_h, out_x:out_x + patch_w] = patch
                result_mask[out_y:out_y + patch_h, out_x:out_x + patch_w] = 1
            else:
                # 用 GraphCut 找最优缝合路径
                _blend_with_graphcut(
                    result, result_mask, patch,
                    out_y, out_x, patch_h, patch_w, overlap,
                    row, col
                )

    return np.clip(result, 0, 255).astype(np.uint8)


def _find_best_patch(texture, result, result_mask,
                     out_y, out_x, patch_h, patch_w, overlap,
                     row, col):
    """在重叠区域找到最匹配的纹理 patch。"""
    tex_h, tex_w = texture.shape[:2]
    max_ty = max(0, tex_h - patch_h)
    max_tx = max(0, tex_w - patch_w)

    # 确定重叠区域
    overlap_top = overlap if row > 0 else 0
    overlap_left = overlap if col > 0 else 0

    if overlap_top == 0 and overlap_left == 0:
        return (np.random.randint(0, max_ty + 1) if max_ty > 0 else 0,
                np.random.randint(0, max_tx + 1) if max_tx > 0 else 0)

    # 计算重叠区域的误差
    best_error = np.inf
    best_ty, best_tx = 0, 0

    # 在纹理上随机采样候选位置
    n_candidates = min(40, (max_ty + 1) * (max_tx + 1))
    candidates = []
    for _ in range(n_candidates * 3):  # 尝试足够多次
        ty = np.random.randint(0, max_ty + 1) if max_ty > 0 else 0
        tx = np.random.randint(0, max_tx + 1) if max_tx > 0 else 0
        candidates.append((ty, tx))
        if len(candidates) >= n_candidates:
            break

    for ty, tx in candidates:
        error = 0.0
        count = 0

        # 上方重叠
        if overlap_top > 0:
            existing = result[out_y:out_y + overlap_top, out_x:out_x + patch_w]
            candidate_patch = texture[ty:ty + overlap_top, tx:tx + patch_w].astype(np.float64)
            mask_region = result_mask[out_y:out_y + overlap_top, out_x:out_x + patch_w]
            diff = np.sum((existing - candidate_patch) ** 2, axis=2)
            if mask_region.sum() > 0:
                error += diff[mask_region > 0].sum()
                count += mask_region.sum()

        # 左方重叠
        if overlap_left > 0:
            start_y = max(0, overlap_top)
            existing = result[out_y + start_y:out_y + patch_h,
                              out_x:out_x + overlap_left]
            candidate_patch = texture[ty + start_y:ty + patch_h,
                                      tx:tx + overlap_left].astype(np.float64)
            mask_region = result_mask[out_y + start_y:out_y + patch_h,
                                       out_x:out_x + overlap_left]
            diff = np.sum((existing - candidate_patch) ** 2, axis=2)
            if mask_region.sum() > 0:
                error += diff[mask_region > 0].sum()
                count += mask_region.sum()

        if count > 0:
            error /= count
            if error < best_error:
                best_error = error
                best_ty, best_tx = ty, tx

    return best_ty, best_tx


def _blend_with_graphcut(result, result_mask, patch,
                         out_y, out_x, patch_h, patch_w, overlap,
                         row, col):
    """
    使用 GraphCut 在重叠区域找最优缝合路径并混合 patch。
    """
    import maxflow

    overlap_top = overlap if row > 0 else 0
    overlap_left = overlap if col > 0 else 0

    if overlap_top == 0 and overlap_left == 0:
        result[out_y:out_y + patch_h, out_x:out_x + patch_w] = patch
        result_mask[out_y:out_y + patch_h, out_x:out_x + patch_w] = 1
        return

    # ---- 水平缝合（与上方 patch 的重叠） ----
    if overlap_top > 0:
        existing = result[out_y:out_y + overlap_top, out_x:out_x + patch_w]
        new_patch = patch[:overlap_top, :patch_w]
        existing_mask = result_mask[out_y:out_y + overlap_top, out_x:out_x + patch_w]

        # 计算重叠区域每个像素的差异
        diff = np.sum((existing - new_patch) ** 2, axis=2)  # (overlap_top, patch_w)

        # 构建图：在重叠区域找一条从上到下的切割路径
        g = maxflow.GraphFloat()
        nodes = g.add_grid_nodes((overlap_top, patch_w))

        # 边权重 = 相邻像素差异之和（鼓励沿低差异区域切割）
        lambda_seam = 20.0

        # 水平边
        if patch_w > 1:
            h_weight = lambda_seam + diff[:, 1:] + diff[:, :-1]
            structure = np.zeros((3, 3))
            structure[1, 2] = 1
            g.add_grid_edges(nodes, structure=structure, weights=h_weight, symmetric=True)

        # 垂直边
        if overlap_top > 1:
            v_weight = lambda_seam + diff[1:, :] + diff[:-1, :]
            structure = np.zeros((3, 3))
            structure[2, 1] = 1
            g.add_grid_edges(nodes, structure=structure, weights=v_weight, symmetric=True)

        # 源/汇约束：顶部强制为 0(使用旧patch)，底部强制为 1(使用新patch)
        top_vals = np.full((1, patch_w), np.inf)
        bottom_vals = np.full((1, patch_w), np.inf)
        g.add_grid_tedges(nodes[:1, :], top_vals, 0)
        g.add_grid_tedges(nodes[-1:, :], 0, bottom_vals)

        g.maxflow()
        seam_mask = g.get_grid_segments(nodes).astype(np.uint8)  # 1=使用新patch

        # 沿缝合线混合
        for y in range(overlap_top):
            for x in range(patch_w):
                if seam_mask[y, x]:
                    result[out_y + y, out_x + x] = new_patch[y, x]
                # else: 保留原有值
            result_mask[out_y + y, out_x:out_x + patch_w] = 1

        # 非重叠部分直接拷贝
        if patch_h > overlap_top:
            result[out_y + overlap_top:out_y + patch_h, out_x:out_x + patch_w] = \
                patch[overlap_top:patch_h, :patch_w]
            result_mask[out_y + overlap_top:out_y + patch_h, out_x:out_x + patch_w] = 1

    # ---- 垂直缝合（与左方 patch 的重叠） ----
    if overlap_left > 0:
        existing = result[out_y:out_y + patch_h, out_x:out_x + overlap_left]
        # 对于水平缝合已经覆盖的部分，用最新的 result
        new_patch = patch[:patch_h, :overlap_left]
        existing_mask = result_mask[out_y:out_y + patch_h, out_x:out_x + overlap_left]

        diff = np.sum((existing - new_patch) ** 2, axis=2)  # (patch_h, overlap_left)

        g = maxflow.GraphFloat()
        nodes = g.add_grid_nodes((patch_h, overlap_left))

        lambda_seam = 20.0

        # 水平边
        if overlap_left > 1:
            h_weight = lambda_seam + diff[:, 1:] + diff[:, :-1]
            structure = np.zeros((3, 3))
            structure[1, 2] = 1
            g.add_grid_edges(nodes, structure=structure, weights=h_weight, symmetric=True)

        # 垂直边
        if patch_h > 1:
            v_weight = lambda_seam + diff[1:, :] + diff[:-1, :]
            structure = np.zeros((3, 3))
            structure[2, 1] = 1
            g.add_grid_edges(nodes, structure=structure, weights=v_weight, symmetric=True)

        # 左边界使用旧patch，右边界使用新patch
        left_vals = np.full((patch_h, 1), np.inf)
        right_vals = np.full((patch_h, 1), np.inf)
        g.add_grid_tedges(nodes[:, :1], left_vals, 0)
        g.add_grid_tedges(nodes[:, -1:], 0, right_vals)

        g.maxflow()
        seam_mask = g.get_grid_segments(nodes).astype(np.uint8)

        for y in range(patch_h):
            for x in range(overlap_left):
                if seam_mask[y, x]:
                    result[out_y + y, out_x + x] = new_patch[y, x]
            result_mask[out_y + y, out_x:out_x + overlap_left] = 1

        # 非重叠部分
        if patch_w > overlap_left:
            result[out_y:out_y + patch_h, out_x + overlap_left:out_x + patch_w] = \
                patch[:patch_h, overlap_left:patch_w]
            result_mask[out_y:out_y + patch_h, out_x + overlap_left:out_x + patch_w] = 1


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def extract_foreground(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    根据二值掩码提取前景，返回带透明背景的 BGRA 图像。

    参数:
        img: BGR 图像 (H, W, 3)
        mask: 二值掩码 (H, W)。1=前景，0=背景

    返回:
        BGRA 图像 (H, W, 4)，背景为透明
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[mask == 0, 3] = 0
    return result


def refine_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """对掩码进行形态学后处理（开运算 + 闭运算）。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 开运算去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 闭运算填充小孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
