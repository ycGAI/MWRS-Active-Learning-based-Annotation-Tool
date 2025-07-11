# Dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

# 更新 conda
RUN conda update conda -y

# 安装系统依赖
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y \
        git wget curl \
        g++ freeglut3-dev build-essential \
        libx11-dev libxmu-dev libxi-dev \
        libglu1-mesa libglu1-mesa-dev libfreeimage-dev \
        ffmpeg libsm6 libxext6 libffi-dev \
        python3-dev python3-pip gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# 升级 pip
RUN pip install --upgrade pip

# 复制 requirements 文件
COPY requirements-base.txt requirements.txt requirements-clean.txt requirements-test.txt* ./

# 安装依赖（按优先级）
# 1. 基础依赖
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ -f requirements-base.txt ]; then \
        pip install -r requirements-base.txt; \
    fi

# 2. 主要依赖
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ -f requirements.txt ]; then \
        pip install -r requirements.txt; \
    fi

# 3. 使用清理后的完整依赖列表（确保版本一致性）
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-deps -r requirements-clean.txt || true

# 4. 测试依赖（仅在测试环境）
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ] && [ -f requirements-test.txt ]; then \
        pip install -r requirements-test.txt; \
    fi

# 复制项目文件
COPY . ./

# 创建必要的目录
RUN mkdir -p /app/models /app/cache /app/cache_dir /app/uncertainty_cache \
    /app/training_data /app/results /app/logs

# 下载 YOLO 模型（如果不存在）
RUN python -c "from ultralytics import YOLO; \
    import os; \
    models = ['yolov8n.pt', 'yolov8m.pt', 'yolov8n-cls.pt', 'yolov8n-seg.pt']; \
    for model in models: \
        model_path = f'/app/models/{model}'; \
        if not os.path.exists(model_path): \
            print(f'Downloading {model}...'); \
            YOLO(model)"

# 设置执行权限
RUN chmod +x /app/start.sh || true

# 设置 Python 路径
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 暴露端口
EXPOSE ${PORT}

# 启动命令
CMD ["/app/start.sh"]