# GraphCut Textures — 交互式图像分割与纹理合成

基于 **GraphCut（图割）** 算法的 Web 应用，支持交互式前景提取与纹理合成。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端
python server.py

# 3. 打开浏览器访问
http://localhost:5000
```

## 功能

| 功能 | 说明 |
|------|------|
| **交互式分割** | 上传图像 → 框选目标 → 一键提取前景（透明背景 PNG） |
| **纹理合成** | 将分割结果平铺/合成为任意尺寸的纹理图像 |
| **双引擎** | 默认 OpenCV GrabCut（极速），可选 PyMaxflow（Boykov-Kolmogorov 原始实现） |

## 项目结构

```
GraphCut/
├── server.py          # Flask 后端（API 服务）
├── graphcut.py        # 核心算法模块
├── index.html         # 前端页面
├── app.js             # 前端交互逻辑
├── styles.css         # 样式
├── requirements.txt   # Python 依赖
└── README.md
```

## API

### `POST /api/segment`

图像分割接口。

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | File | 原始图像 |
| `x`, `y`, `w`, `h` | int | 前景选区坐标与尺寸 |

返回：透明背景的 PNG 图像（自动裁剪至前景区域）。

### `POST /api/texture`

纹理合成接口。

| 参数 | 类型 | 说明 |
|------|------|------|
| `texture` | File | 分割结果图像 |
| `width`, `height` | int | 输出纹理尺寸 |

返回：合成后的 PNG 纹理图像。

## 原理简介

**GraphCut** 将图像分割建模为 s-t 图上的最小割问题：

- 每个像素是图中的一个节点
- **数据项（unary term）**：像素属于前景/背景的代价，由 GMM 颜色模型估计
- **平滑项（pairwise term）**：相邻像素颜色越接近，被分开的代价越大
- 求解最小 s-t 割即得到最优分割边界

纹理合成采用 **Image Quilting**：从源纹理逐一取 patch 拼贴，重叠区域用 GraphCut 找最优缝合路径，消除拼接痕迹。对于分割物体则自动切换为简单平铺模式。

$$
E(\mathbf{x}) = \underbrace{\sum_{i} \theta_i(x_i)}_{\text{数据项}} + \underbrace{\sum_{(i,j)\in\mathcal{N}} \phi_{ij}(x_i, x_j)}_{\text{平滑项}}
$$

> 引擎默认使用 OpenCV 的 GrabCut（C++ 实现），分割一张 800×600 图像不到 1 秒。

## 依赖

- **Flask** — Web 框架
- **OpenCV** — 图像 I/O、GrabCut、形态学操作
- **NumPy** — 数值计算
- **PyMaxflow** — Boykov-Kolmogorov 最大流算法（可选）
- **scikit-learn** — GMM 颜色建模（PyMaxflow 模式）

## License

MIT
