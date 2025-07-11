# ──────────────────────────────── ① 基础镜像 ────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV=false          # 构建时可传 --build-arg TEST_ENV=true

# ──────────────────────────────── ② 工作目录 ────────────────────────────────
WORKDIR /app

# ─────────────────────── ③ conda + Python 3.8 虚拟环境 ──────────────────────
RUN conda update -y conda && \
    conda create -y -n yolo python=3.8 && \
    conda clean -afy
SHELL ["conda", "run", "-n", "yolo", "/bin/bash", "-c"]

# ───────────────────────────── ④ 系统依赖 ─────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git wget curl build-essential g++ \
        ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ──────────────────── ⑤ 预装 scikit-learn 0.24.1 ────────────────────
RUN conda install -y scikit-learn=0.24.1 && conda clean -afy

# ⑤½ 预装官方 cu121 wheel，避免 PyPI 依赖炸裂
RUN pip install --no-cache-dir \
      torch==2.1.2+cu121 \
      torchvision==0.16.2+cu121 \
      torchaudio==2.1.2+cu121 \
      --index-url https://download.pytorch.org/whl/cu121

# 把启动脚本放进镜像并赋 +x
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh


# ───────────────────────────── ⑥ Python 依赖 ─────────────────────────────
ENV PIP_CACHE_DIR=/.cache
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir -r requirements-base.txt

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir -r requirements.txt

COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then pip install --no-cache-dir -r requirements-test.txt ; fi

# ───────────────────────────── ⑦ 拷贝代码 ─────────────────────────────
COPY . .

# ───────────────────────────── ⑧ 运行参数 ─────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    PYTHONPATH=/app

CMD ["conda", "run", "-n", "yolo", "/app/start.sh"]
