#!/bin/bash
# start.sh

# 打印环境信息
echo "🚀 Starting YOLO ML Backend Enhanced..."
echo "📦 Python version: $(python --version)"
echo "🔧 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 启动应用
if [ -f "/app/_wsgi.py" ]; then
    echo "✅ Starting with gunicorn..."
    exec gunicorn --bind :${PORT} \
        --workers ${WORKERS} \
        --threads ${THREADS} \
        --timeout 0 \
        --preload \
        _wsgi:app
else
    echo "✅ Starting with python..."
    exec python /app/ml_backend_enhanced.py
fi