#!/bin/bash
set -e

echo "🚀 启动 YOLO ML Backend 项目..."

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 未安装 Docker，请访问 https://docs.docker.com/get-docker/"
    exit 1
fi

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p models training_data cache uncertainty_cache cache_dir label-studio-data logs results

# 设置脚本权限
chmod +x start.sh 2>/dev/null || true

# 构建并启动
echo "🔨 构建 Docker 镜像..."
docker-compose build

echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
if docker-compose ps | grep -q "Up"; then
    echo "✅ 服务已成功启动！"
    echo ""
    echo "📊 Label Studio: http://localhost:8081"
    echo "   默认账号: admin@example.com"
    echo "   默认密码: password"
    echo ""
    echo "🤖 ML Backend: http://localhost:9090"
    echo ""
    echo "📝 查看日志: docker-compose logs -f"
    echo "🛑 停止服务: docker-compose down"
else
    echo "❌ 服务启动失败，请查看日志："
    docker-compose logs
fi