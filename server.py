"""
GraphCut Textures — Python Flask 后端服务
==========================================
提供图像分割与纹理合成的 API 接口。

启动方式:
    python server.py
    或
    flask run --host=0.0.0.0 --port=5000

API 端点:
    POST /api/segment   — 图像分割（接收图像 + 矩形选区，返回分割结果）
    POST /api/texture   — 纹理贴图（接收纹理 + 尺寸，返回合成纹理）
"""

import io
import os
import traceback

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS  # 跨域支持
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Flask 应用初始化
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder='.',       # 静态文件目录（index.html, styles.css, app.js）
    static_url_path=''
)

# 允许跨域请求（前端可能在别的端口运行）
CORS(app)

# 上传文件大小限制（16 MB）
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否合法."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# 路由：首页
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """返回前端主页面."""
    return app.send_static_file('index.html')


# ---------------------------------------------------------------------------
# API：图像分割（占位 — 待实现）
# ---------------------------------------------------------------------------

@app.route('/api/segment', methods=['POST'])
def api_segment():
    """
    交互式图像分割端点。

    请求 (multipart/form-data):
        image : File      — 原始图像文件
        x     : int       — 选区左上角 x 坐标
        y     : int       — 选区左上角 y 坐标
        w     : int       — 选区宽度
        h     : int       — 选区高度

    返回:
        image/png          — 分割后的透明背景 PNG 图像
    """
    # ---- 参数校验 ----
    if 'image' not in request.files:
        return jsonify({'error': '缺少 image 参数'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '无效的图像文件'}), 400

    try:
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y', 0))
        w = int(request.form.get('w', 0))
        h = int(request.form.get('h', 0))
    except (TypeError, ValueError):
        return jsonify({'error': '选区坐标需为整数'}), 400

    if w <= 0 or h <= 0:
        return jsonify({'error': '选区尺寸无效'}), 400

    # ---- 图像处理 (TODO: 实现 GraphCut / GrabCut 分割) ----
    try:
        import cv2
        import numpy as np

        # 读取上传的图像
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': '无法解码图像'}), 400

        # ============================================================
        # TODO: 在此处实现 GraphCut / GrabCut 分割算法
        # 目前仅将选区区域裁剪出来作为占位结果
        # ============================================================

        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        # 占位：裁剪选区
        result = img[y:y + h, x:x + w].copy()

        # 转换为 PNG 字节流
        _, buffer = cv2.imencode('.png', result)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/png',
            as_attachment=False
        )

    except ImportError:
        return jsonify({'error': 'OpenCV (cv2) 未安装，无法处理图像'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# API：纹理贴图（占位 — 待实现）
# ---------------------------------------------------------------------------

@app.route('/api/texture', methods=['POST'])
def api_texture():
    """
    纹理贴图合成端点。

    请求 (multipart/form-data):
        texture : File      — 输入纹理图像
        width   : int       — 输出宽度 (px)
        height  : int       — 输出高度 (px)

    返回:
        image/png            — 合成后的纹理图像
    """
    # ---- 参数校验 ----
    if 'texture' not in request.files:
        return jsonify({'error': '缺少 texture 参数'}), 400

    file = request.files['texture']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '无效的图像文件'}), 400

    try:
        out_w = int(request.form.get('width', 0))
        out_h = int(request.form.get('height', 0))
    except (TypeError, ValueError):
        return jsonify({'error': '尺寸需为整数'}), 400

    if out_w < 64 or out_h < 64 or out_w > 4096 or out_h > 4096:
        return jsonify({'error': '尺寸需在 64–4096 之间'}), 400

    # ---- 纹理合成 (TODO: 实现 GraphCut 纹理合成) ----
    try:
        import cv2
        import numpy as np

        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        texture = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if texture is None:
            return jsonify({'error': '无法解码图像'}), 400

        # ============================================================
        # TODO: 在此处实现 GraphCut Textures 纹理合成算法
        # 目前仅将输入纹理 resize 到目标尺寸作为占位结果
        # ============================================================

        result = cv2.resize(texture, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        _, buffer = cv2.imencode('.png', result)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/png',
            as_attachment=False
        )

    except ImportError:
        return jsonify({'error': 'OpenCV (cv2) 未安装，无法处理图像'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def api_health():
    """服务健康检查."""
    return jsonify({
        'status': 'ok',
        'service': 'GraphCut Textures API',
        'endpoints': {
            'segment': 'POST /api/segment',
            'texture': 'POST /api/texture',
        }
    })


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 56)
    print("  🎨 GraphCut Textures — API Server")
    print("=" * 56)
    print(f"  前端页面:  http://localhost:5000/")
    print(f"  健康检查:  http://localhost:5000/api/health")
    print(f"  图像分割:  POST /api/segment")
    print(f"  纹理贴图:  POST /api/texture")
    print("=" * 56)
    print()
    print("  ⚠️  图像处理算法尚未实现，当前为占位版本。")
    print("     分割端点返回裁剪区域，贴图端点返回缩放结果。")
    print()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
    )
