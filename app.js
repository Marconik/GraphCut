/* ============================================================
   GraphCut Textures — 交互式图像分割与纹理合成
   前端交互逻辑
   ============================================================ */

// ---------- API 基础配置 ----------
const API_BASE = '/api';

// ---------- DOM 引用 ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    // 文件选择
    fileInput: $('#fileInput'),

    // 源图像
    sourceWrapper: $('#sourceWrapper'),
    sourcePlaceholder: $('#sourcePlaceholder'),
    sourceImage: $('#sourceImage'),
    selectionCanvas: $('#selectionCanvas'),
    selectionInfo: $('#selectionInfo'),
    selectionInfoText: $('#selectionInfoText'),

    // 按钮
    btnReselect: $('#btnReselect'),
    btnCancelSelection: $('#btnCancelSelection'),
    btnSegment: $('#btnSegment'),

    // 结果图像
    resultWrapper: $('#resultWrapper'),
    resultPlaceholder: $('#resultPlaceholder'),
    resultImage: $('#resultImage'),
    segmentLoading: $('#segmentLoading'),

    // 纹理控制
    textureWidth: $('#textureWidth'),
    textureHeight: $('#textureHeight'),
    btnTexture: $('#btnTexture'),

    // 纹理结果
    textureWrapper: $('#textureWrapper'),
    texturePlaceholder: $('#texturePlaceholder'),
    textureImage: $('#textureImage'),
    textureLoading: $('#textureLoading'),

    // 下载
    btnDownload: $('#btnDownload'),
    downloadHint: $('#downloadHint'),
};

// ---------- 应用状态 ----------
const state = {
    sourceImage: null,          // 已加载的源图像 File/DataURL
    selection: null,            // { x, y, w, h } 图像坐标系中的选区
    segmentResult: null,        // 分割结果 DataURL
    textureResult: null,        // 贴图结果 DataURL
    isDrawing: false,
    drawStart: { x: 0, y: 0 },
    canvasScaleX: 1,
    canvasScaleY: 1,
};

// ---------- 工具函数 ----------

/** 显示 Toast 消息 */
function showToast(message, duration = 2500) {
    const existing = $('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/** 更新按钮状态 */
function updateButtonStates() {
    // 分割箭头按钮：仅当有选区时可用
    dom.btnSegment.disabled = !state.selection;

    // 取消选区按钮：仅当有选区时可用
    dom.btnCancelSelection.disabled = !state.selection;

    // 贴图按钮：仅当两个尺寸都填写且有效时可用
    const w = parseInt(dom.textureWidth.value, 10);
    const h = parseInt(dom.textureHeight.value, 10);
    dom.btnTexture.disabled = !(w > 0 && h > 0);

    // 下载按钮：仅当有贴图结果时可用
    dom.btnDownload.disabled = !state.textureResult;
    dom.downloadHint.textContent = state.textureResult
        ? '点击下载合成的纹理贴图'
        : '完成贴图生成后即可下载';
}

/** 从 DataURL 创建一个 Blob */
function dataURLToBlob(dataURL) {
    const parts = dataURL.split(',');
    const mime = parts[0].match(/:(.*?);/)[1];
    const bytes = atob(parts[1]);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
        arr[i] = bytes.charCodeAt(i);
    }
    return new Blob([arr], { type: mime });
}

// ---------- Canvas 选区绘制 ----------

/** 获取鼠标在图像坐标系中的位置 */
function getImageCoords(e) {
    const rect = dom.selectionCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * state.canvasScaleX;
    const y = (e.clientY - rect.top) * state.canvasScaleY;
    return {
        x: Math.max(0, Math.min(x, state.sourceImage.naturalWidth)),
        y: Math.max(0, Math.min(y, state.sourceImage.naturalHeight)),
    };
}

/** 绘制选区矩形和遮罩 */
function drawSelection() {
    const canvas = dom.selectionCanvas;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    if (!state.selection) return;

    const scaleX = W / state.sourceImage.naturalWidth;
    const scaleY = H / state.sourceImage.naturalHeight;

    const sx = state.selection.x * scaleX;
    const sy = state.selection.y * scaleY;
    const sw = state.selection.w * scaleX;
    const sh = state.selection.h * scaleY;

    // 选区外的半透明遮罩
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    ctx.fillRect(0, 0, W, sy);                          // 上
    ctx.fillRect(0, sy, sx, sh);                         // 左
    ctx.fillRect(sx + sw, sy, W - sx - sw, sh);          // 右
    ctx.fillRect(0, sy + sh, W, H - sy - sh);            // 下

    // 选区边框
    ctx.strokeStyle = '#4f46e5';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(sx, sy, sw, sh);
    ctx.setLineDash([]);

    // 四角标记
    const cornerLen = 18;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);

    const corners = [
        [sx, sy, 1, 1], [sx + sw, sy, -1, 1],
        [sx, sy + sh, 1, -1], [sx + sw, sy + sh, -1, -1],
    ];
    corners.forEach(([cx, cy, dx, dy]) => {
        ctx.beginPath();
        ctx.moveTo(cx, cy + dy * cornerLen);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + dx * cornerLen, cy);
        ctx.stroke();
    });
}

/** 设置 Canvas 尺寸以匹配图像的渲染大小 */
function syncCanvasSize() {
    const img = dom.sourceImage;
    const canvas = dom.selectionCanvas;
    const wrapper = dom.sourceWrapper;

    const rect = wrapper.getBoundingClientRect();
    const wrapperW = rect.width;
    const wrapperH = rect.height;

    // 计算图像在 contain 模式下的实际渲染区域
    const imgW = img.naturalWidth;
    const imgH = img.naturalHeight;
    const scale = Math.min(wrapperW / imgW, wrapperH / imgH);
    const renderW = imgW * scale;
    const renderH = imgH * scale;

    canvas.style.width = renderW + 'px';
    canvas.style.height = renderH + 'px';
    canvas.style.left = ((wrapperW - renderW) / 2) + 'px';
    canvas.style.top = ((wrapperH - renderH) / 2) + 'px';
    canvas.width = renderW;
    canvas.height = renderH;

    // 坐标缩放比例：canvas像素 → 原始图像像素
    state.canvasScaleX = img.naturalWidth / renderW;
    state.canvasScaleY = img.naturalHeight / renderH;
}

/** 更新选区信息标签及 wrapper 样式 */
function updateSelectionInfo() {
    if (state.selection) {
        const s = state.selection;
        dom.selectionInfoText.textContent =
            `选区: ${Math.round(s.x)}, ${Math.round(s.y)} — ${Math.round(s.w)}×${Math.round(s.h)}`;
        dom.selectionInfo.style.display = 'block';
        dom.sourceWrapper.classList.add('has-selection');
    } else {
        dom.selectionInfo.style.display = 'none';
        dom.sourceWrapper.classList.remove('has-selection');
    }
}

// ---------- Canvas 鼠标事件 ----------

function onCanvasMouseDown(e) {
    if (!state.sourceImage) return;
    e.preventDefault();

    state.isDrawing = true;
    const coords = getImageCoords(e);
    state.drawStart = coords;
    state.selection = null;
    drawSelection();
    updateSelectionInfo();
    updateButtonStates();
}

function onCanvasMouseMove(e) {
    if (!state.isDrawing) return;
    e.preventDefault();

    const coords = getImageCoords(e);
    const sx = Math.min(state.drawStart.x, coords.x);
    const sy = Math.min(state.drawStart.y, coords.y);
    const sw = Math.abs(coords.x - state.drawStart.x);
    const sh = Math.abs(coords.y - state.drawStart.y);

    state.selection = { x: sx, y: sy, w: sw, h: sh };
    drawSelection();
    updateSelectionInfo();
}

function onCanvasMouseUp(e) {
    if (!state.isDrawing) return;
    state.isDrawing = false;

    // 太小的选区视为无效（最小 10px）
    if (state.selection && (state.selection.w < 10 || state.selection.h < 10)) {
        state.selection = null;
        drawSelection();
        updateSelectionInfo();
        updateButtonStates();
        return;
    }

    updateButtonStates();
    updateSelectionInfo();

    if (state.selection) {
        showToast('选区已就绪，点击箭头开始分割');
    }
}

// ---------- 图像加载 ----------

/** 触发文件选择器 */
function triggerFileSelect() {
    dom.fileInput.click();
}

/** 加载图像文件 */
function loadImage(file) {
    if (!file || !file.type.startsWith('image/')) {
        showToast('请选择有效的图像文件');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const dataURL = e.target.result;
        dom.sourceImage.src = dataURL;
        dom.sourceImage.onload = () => {
            state.sourceImage = dom.sourceImage;
            state.selection = null;
            state.segmentResult = null;
            state.textureResult = null;

            // 图像框长宽比自适应图像
            const img = dom.sourceImage;
            dom.sourceWrapper.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;

            // 显示图像，隐藏占位符
            dom.sourceImage.style.display = 'block';
            dom.sourcePlaceholder.style.display = 'none';
            dom.selectionCanvas.style.display = 'block';
            dom.sourceWrapper.classList.add('has-image');
            dom.sourceWrapper.classList.remove('has-selection');

            // 清除结果: 隐藏结果图 + 重置右侧图像框为默认 4:3
            dom.resultImage.style.display = 'none';
            dom.resultPlaceholder.style.display = 'flex';
            dom.resultWrapper.style.aspectRatio = '4 / 3';
            dom.textureImage.style.display = 'none';
            dom.texturePlaceholder.style.display = 'flex';

            // 等待布局完成后再同步 Canvas 尺寸
            requestAnimationFrame(() => {
                syncCanvasSize();
                drawSelection();
                updateSelectionInfo();
                updateButtonStates();
            });

            showToast('图像已加载，请在图像上拖拽框选目标区域');
        };
    };
    reader.readAsDataURL(file);
}

/** 取消选区 */
function cancelSelection() {
    state.selection = null;
    drawSelection();
    updateSelectionInfo();
    updateButtonStates();
    dom.sourceWrapper.classList.remove('has-selection');
    showToast('选区已取消');
}

// ---------- 图像分割 ----------

async function performSegmentation() {
    if (!state.selection || !state.sourceImage) return;

    // 显示加载状态
    dom.segmentLoading.style.display = 'flex';
    dom.btnSegment.disabled = true;
    dom.btnCancelSelection.disabled = true;

    try {
        // 构建 FormData
        const formData = new FormData();
        const blob = dataURLToBlob(dom.sourceImage.src);
        formData.append('image', blob, 'source.png');
        formData.append('x', Math.round(state.selection.x).toString());
        formData.append('y', Math.round(state.selection.y).toString());
        formData.append('w', Math.round(state.selection.w).toString());
        formData.append('h', Math.round(state.selection.h).toString());

        const response = await fetch(`${API_BASE}/segment`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        const resultBlob = await response.blob();
        const dataURL = URL.createObjectURL(resultBlob);

        // 显示结果
        dom.resultImage.src = dataURL;
        dom.resultImage.onload = () => {
            const rImg = dom.resultImage;
            dom.resultWrapper.style.aspectRatio = `${rImg.naturalWidth} / ${rImg.naturalHeight}`;
        };
        dom.resultImage.style.display = 'block';
        dom.resultPlaceholder.style.display = 'none';
        state.segmentResult = dataURL;

        showToast('✅ 分割完成！');
    } catch (err) {
        console.error('分割失败:', err);
        showToast('❌ 分割失败，请检查后端服务是否运行');
    } finally {
        dom.segmentLoading.style.display = 'none';
        updateButtonStates();
    }
}

// ---------- 纹理贴图 ----------

async function performTextureSynthesis() {
    const w = parseInt(dom.textureWidth.value, 10);
    const h = parseInt(dom.textureHeight.value, 10);

    if (!w || !h || w < 64 || h < 64 || w > 4096 || h > 4096) {
        showToast('请设置有效的贴图尺寸 (64–4096 px)');
        return;
    }

    // 需要有分割结果才能贴图
    if (!state.segmentResult) {
        showToast('请先在 Step 1 完成图像分割');
        return;
    }

    // 显示加载状态
    dom.textureLoading.style.display = 'flex';
    dom.btnTexture.disabled = true;

    try {
        const formData = new FormData();
        const blob = dataURLToBlob(state.segmentResult);
        formData.append('texture', blob, 'texture.png');
        formData.append('width', w.toString());
        formData.append('height', h.toString());

        const response = await fetch(`${API_BASE}/texture`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        const resultBlob = await response.blob();
        const dataURL = URL.createObjectURL(resultBlob);

        // 显示结果，图像框按用户设定尺寸比例自适应
        dom.textureImage.src = dataURL;
        dom.textureImage.onload = () => {
            const tImg = dom.textureImage;
            dom.textureWrapper.style.aspectRatio = `${tImg.naturalWidth} / ${tImg.naturalHeight}`;
        };
        dom.textureImage.style.display = 'block';
        dom.texturePlaceholder.style.display = 'none';
        state.textureResult = dataURL;

        showToast('✅ 贴图生成完成！');
    } catch (err) {
        console.error('贴图生成失败:', err);
        showToast('❌ 贴图生成失败，请检查后端服务是否运行');
    } finally {
        dom.textureLoading.style.display = 'none';
        updateButtonStates();
    }
}

/** 下载贴图结果 */
function downloadTexture() {
    if (!state.textureResult) return;

    const a = document.createElement('a');
    a.href = state.textureResult;
    a.download = `texture_${dom.textureWidth.value}x${dom.textureHeight.value}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    showToast('📥 贴图已开始下载');
}

// ---------- 事件绑定 ----------

/** 点击源图像区域触发文件选择（仅在无图像时） */
dom.sourceWrapper.addEventListener('click', (e) => {
    // 如果有图像且点击在 canvas 上，不触发文件选择
    if (state.sourceImage && e.target === dom.selectionCanvas) {
        return;
    }
    if (!state.sourceImage) {
        triggerFileSelect();
    }
});

/** 文件选择器变化 */
dom.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        loadImage(file);
    }
    // 重置以允许重复选择同一文件
    dom.fileInput.value = '';
});

/** Canvas 鼠标事件 */
dom.selectionCanvas.addEventListener('mousedown', onCanvasMouseDown);
dom.selectionCanvas.addEventListener('mousemove', onCanvasMouseMove);
document.addEventListener('mouseup', (e) => {
    if (state.isDrawing) {
        onCanvasMouseUp(e);
    }
});

/** 窗口大小变化时重新同步 Canvas */
window.addEventListener('resize', () => {
    if (state.sourceImage) {
        syncCanvasSize();
        drawSelection();
    }
});

/** 重新选择图像 */
dom.btnReselect.addEventListener('click', (e) => {
    e.stopPropagation();
    triggerFileSelect();
});

/** 取消选区 */
dom.btnCancelSelection.addEventListener('click', (e) => {
    e.stopPropagation();
    cancelSelection();
});

/** 执行分割 */
dom.btnSegment.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!dom.btnSegment.disabled) {
        performSegmentation();
    }
});

/** 纹理尺寸输入变化 */
dom.textureWidth.addEventListener('input', updateButtonStates);
dom.textureHeight.addEventListener('input', updateButtonStates);

/** 生成贴图 */
dom.btnTexture.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!dom.btnTexture.disabled) {
        performTextureSynthesis();
    }
});

/** 下载贴图 */
dom.btnDownload.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!dom.btnDownload.disabled) {
        downloadTexture();
    }
});

/** 键盘快捷键 */
document.addEventListener('keydown', (e) => {
    // Escape 取消选区
    if (e.key === 'Escape' && state.selection) {
        e.preventDefault();
        cancelSelection();
    }
    // Enter 执行分割
    if (e.key === 'Enter' && !dom.btnSegment.disabled) {
        e.preventDefault();
        performSegmentation();
    }
});

// ---------- 拖拽文件支持 ----------

dom.sourceWrapper.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dom.sourceWrapper.style.borderColor = '#4f46e5';
    dom.sourceWrapper.style.background = '#eef2ff';
});

dom.sourceWrapper.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dom.sourceWrapper.style.borderColor = '';
    dom.sourceWrapper.style.background = '';
});

dom.sourceWrapper.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dom.sourceWrapper.style.borderColor = '';
    dom.sourceWrapper.style.background = '';

    const file = e.dataTransfer.files[0];
    if (file) {
        loadImage(file);
    }
});

// ---------- 初始化 ----------
function init() {
    updateButtonStates();
    console.log('GraphCut Textures 前端已就绪');
    console.log('API 地址:', API_BASE);
}

init();
