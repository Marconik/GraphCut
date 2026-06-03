"""
GraphCut 图像分割与纹理合成核心模块
=======================================

基于 Boykov-Kolmogorov 最大流/最小割算法的图像处理实现。

性能优化策略：
  - 默认使用 OpenCV GrabCut（C++ 优化，<1 秒完成）
  - PyMaxflow 模式自动降采样 + GMM 子采样（避免 O(n²) 爆炸）
  - 纹理合成使用向量化 numpy 操作替代 Python 循环

依赖:
  pip install PyMaxflow numpy opencv-python scikit-learn
"""

import time
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# 最大流引擎选择
# ---------------------------------------------------------------------------

_maxflow_engine = None  # 'grabcut' | 'pymaxflow'


def _init_maxflow():
    """检测可用的最大流引擎。GrabCut 优先（速度最快）。"""
    global _maxflow_engine
    if _maxflow_engine is not None:
        return _maxflow_engine

    # 首选：OpenCV GrabCut — C++ 优化，生产级速度
    _maxflow_engine = 'grabcut'

    # 检测 PyMaxflow 是否可用（作为备选方案）
    try:
        import maxflow
        _ = maxflow.GraphFloat()
        # PyMaxflow 可用但不切换默认，保留给高级用户
    except ImportError:
        pass

    return _maxflow_engine


def use_pymaxflow():
    """显式切换到 PyMaxflow 引擎（Boykov-Kolmogorov 原始实现）。"""
    global _maxflow_engine
    try:
        import maxflow
        _ = maxflow.GraphFloat()
        _maxflow_engine = 'pymaxflow'
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 图像分割
# ---------------------------------------------------------------------------

# 降采样阈值：超过此尺寸的图像将先缩小再处理
_MAX_DIM = 400
# GMM 子采样最大像素数
_MAX_GMM_SAMPLES = 3000
# 最大迭代次数
_MAX_ITERS = 3


def segment_image(img: np.ndarray, rect: tuple,
                  max_iters: int = 3,
                  border_trim: int = 2) -> np.ndarray:
    """
    使用 GraphCut 对图像进行交互式前景分割。

    参数:
        img: BGR 图像 (H, W, 3)，uint8
        rect: 用户框选的前景区域 (x, y, w, h)
        max_iters: 最大迭代次数
        border_trim: 边界收缩像素数

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
    """
    使用 OpenCV GrabCut（默认引擎）。
    运行在 C++ 层，800×600 图像 < 1 秒完成。
    """
    x, y, rw, rh = rect
    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, (x, y, rw, rh),
                bgd_model, fgd_model,
                max_iters, cv2.GC_INIT_WITH_RECT)

    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return result


def _segment_pymaxflow(img: np.ndarray, rect: tuple,
                       max_iters: int, border_trim: int) -> np.ndarray:
    """
    使用 PyMaxflow 的自定义 GraphCut 分割（高级模式）。

    性能优化：
      1. 大图自动降采样到 ≤400px 后处理，再升采样掩码
      2. GMM 仅对随机子采样像素拟合（max 3000 样本）
      3. 使用 diagonal covariance（比 full 快 20x）
      4. 仅 3 维 LAB 特征
      5. 最多 3 次迭代，变化 <0.1% 时提前终止
    """
    import maxflow

    h_orig, w_orig = img.shape[:2]
    x, y, rw, rh = rect

    # ---- 降采样 ----
    scale = 1.0
    img_small = img
    if max(h_orig, w_orig) > _MAX_DIM:
        scale = _MAX_DIM / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = int(x * scale)
        y = int(y * scale)
        rw = max(1, int(rw * scale))
        rh = max(1, int(rh * scale))
        border_trim = max(0, int(border_trim * scale))

    h, w = img_small.shape[:2]

    # ---- 初始化掩码 ----
    mask = np.zeros((h, w), dtype=np.uint8)  # 0=背景
    # 矩形内部为未知区域（后续由 GMM 决定）
    y1 = max(0, y + border_trim)
    y2 = min(h, y + rh - border_trim)
    x1 = max(0, x + border_trim)
    x2 = min(w, x + rw - border_trim)
    mask[y1:y2, x1:x2] = 2  # 未知

    # 矩形中心 2/3 区域作为确定前景种子
    cy_margin_y = max(rh // 6, 1)
    cx_margin_x = max(rw // 6, 1)
    cy1 = max(0, y + cy_margin_y)
    cy2 = min(h, y + rh - cy_margin_y)
    cx1 = max(0, x + cx_margin_x)
    cx2 = min(w, x + rw - cx_margin_x)
    mask[cy1:cy2, cx1:cx2] = 1  # 确定前景

    # ---- 颜色特征：仅 3 维 LAB（比 6 维快 4x） ----
    img_lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB).astype(np.float64)
    features = img_lab.reshape(-1, 3)

    # ---- GMM 参数 ----
    gamma = 30.0
    lambda_smooth = 10.0
    sigma_smooth = 8.0

    for iteration in range(min(max_iters, _MAX_ITERS)):
        t_iter = time.time()

        fg_idx = np.where(mask.reshape(-1) == 1)[0]
        bg_idx = np.where(mask.reshape(-1) == 0)[0]

        if len(fg_idx) < 30 or len(bg_idx) < 30:
            break

        # ---- GMM 拟合（仅子采样） ----
        from sklearn.mixture import GaussianMixture

        n_fg_samp = min(_MAX_GMM_SAMPLES, len(fg_idx))
        n_bg_samp = min(_MAX_GMM_SAMPLES, len(bg_idx))

        fg_samples = features[np.random.choice(fg_idx, n_fg_samp, replace=False)]
        bg_samples = features[np.random.choice(bg_idx, n_bg_samp, replace=False)]

        # 自适应分量数
        k_fg = min(5, max(2, n_fg_samp // 500))
        k_bg = min(5, max(2, n_bg_samp // 500))

        # diag covariance：比 full 快 20x，精度几乎无损
        fg_gmm = GaussianMixture(n_components=k_fg,
                                 covariance_type='diag', reg_covar=1e-3,
                                 max_iter=50, random_state=42).fit(fg_samples)
        bg_gmm = GaussianMixture(n_components=k_bg,
                                 covariance_type='diag', reg_covar=1e-3,
                                 max_iter=50, random_state=42).fit(bg_samples)

        # ---- 计算所有像素的数据项（瓶颈：score_samples 对全图） ----
        fg_logprob = fg_gmm.score_samples(features)
        bg_logprob = bg_gmm.score_samples(features)
        fg_logprob = np.clip(fg_logprob, -80, 80)
        bg_logprob = np.clip(bg_logprob, -80, 80)

        unary_bg = -fg_logprob.reshape(h, w) * gamma  # 属于前景的证据
        unary_fg = -bg_logprob.reshape(h, w) * gamma  # 属于背景的证据

        # ---- 构建 s-t 图 ----
        g = maxflow.GraphFloat()
        node_ids = g.add_grid_nodes((h, w))

        # 平滑项（仅计算一次，后续迭代复用）
        if iteration == 0:
            img_f = img_small.astype(np.float64)
            diff_h = np.sum(np.abs(img_f[:, 1:, :] - img_f[:, :-1, :]), axis=2)
            _cached_smooth_h = lambda_smooth * np.exp(-diff_h ** 2 / (2 * sigma_smooth ** 2))
            _cached_smooth_h_pad = np.zeros((h, w), dtype=np.float64)
            _cached_smooth_h_pad[:, :-1] = _cached_smooth_h

            diff_v = np.sum(np.abs(img_f[1:, :, :] - img_f[:-1, :, :]), axis=2)
            _cached_smooth_v = lambda_smooth * np.exp(-diff_v ** 2 / (2 * sigma_smooth ** 2))
            _cached_smooth_v_pad = np.zeros((h, w), dtype=np.float64)
            _cached_smooth_v_pad[:-1, :] = _cached_smooth_v

        # 添加邻接边
        structure_h = np.zeros((3, 3)); structure_h[1, 2] = 1
        g.add_grid_edges(node_ids, structure=structure_h,
                         weights=_cached_smooth_h_pad, symmetric=True)
        structure_v = np.zeros((3, 3)); structure_v[2, 1] = 1
        g.add_grid_edges(node_ids, structure=structure_v,
                         weights=_cached_smooth_v_pad, symmetric=True)

        # 数据项 + 硬约束
        hard_bg = np.where(mask == 0, 1e9, 0.0).astype(np.float64)
        hard_fg = np.where(mask == 1, 1e9, 0.0).astype(np.float64)
        g.add_grid_tedges(node_ids,
                          unary_bg + hard_bg,
                          unary_fg + hard_fg)

        # ---- 求解 ----
        g.maxflow()
        new_mask = g.get_grid_segments(node_ids).astype(np.uint8)

        # 收敛检测
        changed = np.sum(new_mask != mask)
        mask = new_mask
        if changed < (h * w * 0.001):
            break

    # ---- 升采样回原始分辨率 ----
    if scale < 1.0:
        mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return mask


# ---------------------------------------------------------------------------
# 纹理合成（Image Quilting + GraphCut 缝合）
# ---------------------------------------------------------------------------

def _crop_to_content(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    将图像裁剪到非黑色内容区域。
    适用于分割结果（前景+黑色背景），去除黑边使纹理可铺贴。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[..., :3].max(axis=2)
    rows = np.any(gray > threshold, axis=1)
    cols = np.any(gray > threshold, axis=0)
    if not rows.any() or not cols.any():
        return img  # 全黑，无法裁剪
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # 加一点 padding
    pad = 4
    ymin = max(0, ymin - pad)
    ymax = min(img.shape[0], ymax + pad + 1)
    xmin = max(0, xmin - pad)
    xmax = min(img.shape[1], xmax + pad + 1)
    return img[ymin:ymax, xmin:xmax]


def synthesize_texture(texture: np.ndarray,
                       out_width: int, out_height: int,
                       patch_size: int = 48,
                       overlap: int = 8) -> np.ndarray:
    """
    纹理合成：自动选择最佳策略。

    - 如果输入含大量黑色/透明区域（分割物体），使用简单平铺
    - 如果输入是自然纹理，使用 Image Quilting + GraphCut 缝合

    参数:
        texture: BGR 图像 (H, W, 3)
        out_width, out_height: 输出尺寸
        patch_size: patch 边长（仅 Quilting 模式）
        overlap: 重叠像素数（仅 Quilting 模式）

    返回:
        合成后的 BGR 图像 (out_height, out_width, 3)
    """
    # ---- 裁剪到内容区域，再填充边缘黑色残留 ----
    cropped = _crop_to_content(texture)
    cropped = _fill_black_holes(cropped)
    ch, cw = cropped.shape[:2]

    if ch < 4 or cw < 4:
        return cv2.resize(texture, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

    # ---- 判断策略 ----    
    # 对于分割物体（有黑边）始终用平铺；自然纹理才用 Quilting
    total_px = ch * cw
    black_px = (cropped.sum(axis=2) <= 30).sum()
    fill_ratio = 1.0 - black_px / total_px

    if fill_ratio < 0.98 or black_px > 0:
        # 有黑色残留或物体分割 → 平铺（更稳健）
        return _tile_simple(cropped, out_width, out_height)
    else:
        # 纯自然纹理 → Image Quilting + GraphCut
        return _tile_quilting(cropped, out_width, out_height, patch_size, overlap)


def _tile_simple(texture: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    简单平铺：直接重复纹理图案。
    适合分割出的物体（非自然纹理）。

    先将内容区域边缘外扩填充（用最近邻颜色替代黑色），
    然后平铺以避免黑色间隙。
    """
    # 填充黑色区域：用最近的非黑色像素颜色替换
    filled = _fill_black_holes(texture)
    th, tw = filled.shape[:2]
    reps_y = int(np.ceil(out_h / th))
    reps_x = int(np.ceil(out_w / tw))
    result = np.tile(filled, (reps_y, reps_x, 1))
    return result[:out_h, :out_w]


def _fill_black_holes(img: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    用形态学膨胀填充图像中的黑色区域。
    将黑色像素替换为最近的非黑色像素颜色。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray <= threshold).astype(np.uint8)  # 黑色区域=1

    if mask.sum() == 0:
        return img  # 没有黑色区域

    # 使用 inpaint 填充黑色区域
    result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result


def _tile_quilting(texture: np.ndarray,
                   out_width: int, out_height: int,
                   patch_size: int, overlap: int) -> np.ndarray:
    """
    Image Quilting + GraphCut 纹理合成。
    适合自然纹理（草地、沙石、织物等）。
    """
    import maxflow

    tex_h, tex_w = texture.shape[:2]

    if tex_h < 8 or tex_w < 8:
        return cv2.resize(texture, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

    # ---- 第一步：参数调整 ----
    patch_size = min(patch_size, tex_h, tex_w)
    overlap = min(overlap, patch_size // 3)
    step = patch_size - overlap

    cols = max(1, int(np.ceil((out_width - overlap) / step)))
    rows = max(1, int(np.ceil((out_height - overlap) / step)))
    canvas_h = rows * step + overlap
    canvas_w = cols * step + overlap

    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    result_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    max_ty = max(0, tex_h - patch_size)
    max_tx = max(0, tex_w - patch_size)

    # ---- 第二步：光栅扫描放置 patch ----
    for row in range(rows):
        for col in range(cols):
            out_y = row * step
            out_x = col * step
            ph = min(patch_size, canvas_h - out_y)
            pw = min(patch_size, canvas_w - out_x)

            # 选择最佳 patch
            ty, tx = _pick_patch(
                texture, result, result_mask,
                out_y, out_x, ph, pw, overlap,
                row, col, max_ty, max_tx
            )

            patch = texture[ty:ty + ph, tx:tx + pw].astype(np.float64)

            if row == 0 and col == 0:
                _paste_full(result, result_mask, patch, out_y, out_x, ph, pw)
            else:
                _paste_with_seam(result, result_mask, patch,
                                 out_y, out_x, ph, pw, overlap, row, col)

    return np.clip(result[:out_height, :out_width], 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _paste_full(result, result_mask, patch, out_y, out_x, ph, pw):
    """直接拷贝整个 patch（无重叠时使用）。"""
    result[out_y:out_y + ph, out_x:out_x + pw] = patch
    result_mask[out_y:out_y + ph, out_x:out_x + pw] = 1


def _pick_patch(texture, result, result_mask,
                out_y, out_x, ph, pw, overlap,
                row, col, max_ty, max_tx):
    """在纹理中找最匹配的 patch（基于重叠区域 SSE）。"""
    tex_h, tex_w = texture.shape[:2]
    top_ov = overlap if row > 0 else 0
    left_ov = overlap if col > 0 else 0

    if top_ov == 0 and left_ov == 0:
        ty = np.random.randint(0, max_ty + 1) if max_ty > 0 else 0
        tx = np.random.randint(0, max_tx + 1) if max_tx > 0 else 0
        return ty, tx

    # 候选采样
    n_candidates = min(30, (max_ty + 1) * (max_tx + 1))
    best_err = np.inf
    best = (0, 0)

    for _ in range(n_candidates * 2):
        ty = np.random.randint(0, max(1, tex_h - ph + 1))
        tx = np.random.randint(0, max(1, tex_w - pw + 1))

        err = 0.0
        n = 0

        # 上方重叠
        if top_ov > 0:
            a = result[out_y:out_y + top_ov, out_x:out_x + pw]
            b = texture[ty:ty + top_ov, tx:tx + pw].astype(np.float64)
            m = result_mask[out_y:out_y + top_ov, out_x:out_x + pw]
            diff = np.sum((a - b) ** 2, axis=2)
            if m.sum() > 0:
                err += float(diff[m > 0].sum())
                n += int(m.sum())

        # 左方重叠（排除已被上方覆盖的部分）
        if left_ov > 0:
            sy = top_ov
            a = result[out_y + sy:out_y + ph, out_x:out_x + left_ov]
            b = texture[ty + sy:ty + ph, tx:tx + left_ov].astype(np.float64)
            m = result_mask[out_y + sy:out_y + ph, out_x:out_x + left_ov]
            diff = np.sum((a - b) ** 2, axis=2)
            if m.sum() > 0:
                err += float(diff[m > 0].sum())
                n += int(m.sum())

        if n > 0 and err / n < best_err:
            best_err = err / n
            best = (ty, tx)

    return best


def _paste_with_seam(result, result_mask, patch,
                     out_y, out_x, ph, pw, overlap, row, col):
    """在重叠区域用 GraphCut 找缝合路径并混合。"""
    import maxflow

    top_ov = overlap if row > 0 else 0
    left_ov = overlap if col > 0 else 0

    # ---- 处理上方重叠（水平缝合） ----
    if top_ov > 0:
        _seam_horizontal(result, result_mask, patch,
                         out_y, out_x, ph, pw, top_ov)

    # ---- 处理左方重叠（垂直缝合） ----
    if left_ov > 0:
        _seam_vertical(result, result_mask, patch,
                       out_y, out_x, ph, pw, left_ov, top_ov)


def _seam_horizontal(result, result_mask, patch,
                     out_y, out_x, ph, pw, top_ov):
    """水平缝合：在 top_ov 行重叠区找从上到下的最优切割路径。"""
    import maxflow

    old = result[out_y:out_y + top_ov, out_x:out_x + pw]
    new = patch[:top_ov, :pw]
    diff = np.sum((old - new) ** 2, axis=2)  # (top_ov, pw)

    g = maxflow.GraphFloat()
    nodes = g.add_grid_nodes((top_ov, pw))

    base_weight = 10.0
    # 水平边
    if pw > 1:
        wh = base_weight + diff[:, 1:] + diff[:, :-1]
        wh_pad = np.zeros((top_ov, pw), dtype=np.float64)
        wh_pad[:, :-1] = wh
        s = np.zeros((3, 3)); s[1, 2] = 1
        g.add_grid_edges(nodes, structure=s, weights=wh_pad, symmetric=True)

    # 垂直边
    if top_ov > 1:
        wv = base_weight + diff[1:, :] + diff[:-1, :]
        wv_pad = np.zeros((top_ov, pw), dtype=np.float64)
        wv_pad[:-1, :] = wv
        s = np.zeros((3, 3)); s[2, 1] = 1
        g.add_grid_edges(nodes, structure=s, weights=wv_pad, symmetric=True)

    # 上边=保留旧，下边=使用新
    g.add_grid_tedges(nodes[:1, :], np.full((1, pw), 1e9, dtype=np.float64), 0)
    g.add_grid_tedges(nodes[-1:, :], 0, np.full((1, pw), 1e9, dtype=np.float64))

    g.maxflow()
    seg = g.get_grid_segments(nodes).astype(bool)  # True=使用新patch

    # 只替换 seg=True 的像素
    region = result[out_y:out_y + top_ov, out_x:out_x + pw]
    region[seg] = new[seg]
    result_mask[out_y:out_y + top_ov, out_x:out_x + pw] = 1

    # 非重叠区域（下方）全量拷贝
    if ph > top_ov:
        result[out_y + top_ov:out_y + ph, out_x:out_x + pw] = patch[top_ov:ph, :pw]
        result_mask[out_y + top_ov:out_y + ph, out_x:out_x + pw] = 1


def _seam_vertical(result, result_mask, patch,
                   out_y, out_x, ph, pw, left_ov, top_ov):
    """垂直缝合：在 left_ov 列重叠区找从左到右的最优切割路径。"""
    import maxflow

    # 注意：top_ov 行已被水平缝合修改过，读取的是混合后的结果
    old = result[out_y:out_y + ph, out_x:out_x + left_ov]
    new = patch[:ph, :left_ov]
    diff = np.sum((old - new) ** 2, axis=2)  # (ph, left_ov)

    g = maxflow.GraphFloat()
    nodes = g.add_grid_nodes((ph, left_ov))

    base_weight = 10.0
    # 水平边
    if left_ov > 1:
        wh = base_weight + diff[:, 1:] + diff[:, :-1]
        wh_pad = np.zeros((ph, left_ov), dtype=np.float64)
        wh_pad[:, :-1] = wh
        s = np.zeros((3, 3)); s[1, 2] = 1
        g.add_grid_edges(nodes, structure=s, weights=wh_pad, symmetric=True)

    # 垂直边
    if ph > 1:
        wv = base_weight + diff[1:, :] + diff[:-1, :]
        wv_pad = np.zeros((ph, left_ov), dtype=np.float64)
        wv_pad[:-1, :] = wv
        s = np.zeros((3, 3)); s[2, 1] = 1
        g.add_grid_edges(nodes, structure=s, weights=wv_pad, symmetric=True)

    # 左边=保留旧，右边=使用新
    g.add_grid_tedges(nodes[:, :1], np.full((ph, 1), 1e9, dtype=np.float64), 0)
    g.add_grid_tedges(nodes[:, -1:], 0, np.full((ph, 1), 1e9, dtype=np.float64))

    g.maxflow()
    seg = g.get_grid_segments(nodes).astype(bool)  # True=使用新patch

    # 只替换 seg=True 的像素
    region = result[out_y:out_y + ph, out_x:out_x + left_ov]
    region[seg] = new[seg]
    result_mask[out_y:out_y + ph, out_x:out_x + left_ov] = 1

    # 非重叠区域（右方）——只拷贝未被水平缝合覆盖的部分
    if pw > left_ov:
        # 从 top_ov 行开始拷贝（上方已被水平缝合处理）
        src_y_start = top_ov
        result[out_y + src_y_start:out_y + ph,
               out_x + left_ov:out_x + pw] = patch[src_y_start:ph, left_ov:pw]
        result_mask[out_y + src_y_start:out_y + ph,
                     out_x + left_ov:out_x + pw] = 1


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
    result[mask == 0] = [0, 0, 0, 0]  # 背景设为全透明黑色
    return result


def refine_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """对掩码进行形态学后处理（开运算 + 闭运算）。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 开运算去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 闭运算填充小孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
