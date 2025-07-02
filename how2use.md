YOLO Fine-tune with Label Studio 完整使用指南
本指南将帮助您从零开始部署和使用YOLO自动标注系统，支持模型fine-tune功能。

目录
系统要求
安装步骤
配置Label Studio
配置ML Backend
使用Fine-tune功能
常见问题
系统要求
Docker 和 Docker Compose
Python 3.8+
至少8GB内存
GPU（可选，但推荐用于训练）
安装步骤
1. 克隆项目
bash
git clone https://github.com/yourusername/MWRS-Active-Learning-based-Annotation-Tool.git
cd MWRS-Active-Learning-based-Annotation-Tool
2. 启动Label Studio
bash
# 使用Docker启动Label Studio
docker run -d \
  --name label-studio \
  -p 8081:8080 \
  -v ~/label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest
访问 http://localhost:8081 创建账号并登录。

3. 获取Label Studio API Token
登录Label Studio
点击右上角用户头像 → "Account & Settings"
在"Personal Access Token"部分，点击"Create New Token"
复制生成的token（格式类似：eyJhbGciOiJIUzI1NiIs...）
配置ML Backend
1. 配置docker-compose.yml
进入YOLO示例目录：

bash
cd examples/yolo
编辑docker-compose.yml：

yaml
version: '3.8'
services:
  yolo:
    build: .
    container_name: yolo
    ports:
      - "9090:9090"
    environment:
      - MODEL_DIR=/app/models
      - LABEL_STUDIO_URL=http://host.docker.internal:8081
      - LABEL_STUDIO_API_KEY=你的API_TOKEN  # 替换为实际的token
      - LOG_LEVEL=DEBUG
    volumes:
      - ./models:/app/models
      - ./training_data:/app/training_data
      - ./cache:/app/cache
      # 挂载Label Studio数据目录（根据实际路径调整）
      - ~/label-studio-data/media:/label-studio-data:ro
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
2. 准备模型文件
确保在models目录下有您的YOLO模型：

bash
# 如果使用自定义模型
cp /path/to/your/best.pt ./models/

# 或下载预训练模型
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
cd ..
3. 启动ML Backend
bash
# 构建并启动
docker-compose up --build -d

# 查看日志确认启动成功
docker logs -f yolo
配置Label Studio
1. 创建项目
在Label Studio中创建新项目，使用以下标注配置：

xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image" 
                   model_path="best.pt"
                   model_score_threshold="0.3"
                   model_finetune="true"
                   model_min_annotations="10"
                   model_retrain_interval="5"
                   model_finetune_epochs="20"
                   model_finetune_batch="8">
    <Label value="类别1" background="#FF6B6B"/>
    <Label value="类别2" background="#4ECDC4"/>
    <Label value="类别3" background="#45B7D1"/>
    <!-- 添加更多类别 -->
  </RectangleLabels>
</View>
重要参数说明：

model_path: YOLO模型文件名
model_finetune="true": 启用fine-tune功能
model_min_annotations: 开始训练的最小标注数
model_retrain_interval: 每增加多少标注重新训练
model_finetune_epochs: 训练轮数
model_finetune_batch: 批次大小
2. 连接ML Backend
在项目设置中，进入"Model"页面
点击"Connect Model"
输入URL: http://localhost:9090
选择"No Authentication"
启用"Use for interactive preannotations"
点击"Validate and Save"
3. 上传图片
上传需要标注的图片到项目中。

使用Fine-tune功能
1. 自动预标注
上传图片后，ML Backend会自动生成预测结果。

2. 人工修正
打开任务
修正或添加边界框
点击"Submit"提交
3. 监控训练状态
bash
# 查看标注进度
docker exec -it yolo cat /app/training_data/project_1_counter.json

# 监控训练日志
docker logs -f yolo | grep -E "(Processing|Saved|Training|epochs)"
4. 自动训练触发
当满足以下条件时，系统会自动开始fine-tune：

标注数量达到model_min_annotations（默认10个）
新增标注数量是model_retrain_interval的倍数（默认每5个）
5. 查看训练结果
bash
# 查看训练进度
docker exec -it yolo ls -la /app/models/runs/project_1/finetune/

# 查看训练指标
docker exec -it yolo cat /app/models/runs/project_1/finetune/results.csv

# 查看训练图表
docker exec -it yolo ls /app/models/runs/project_1/finetune/*.jpg
6. 使用Fine-tuned模型
训练完成后，系统会自动使用新模型进行预测。新模型保存在：

/app/models/finetune_project_1.pt
常见问题
1. 找不到图片文件
问题：日志显示 "Local image path does not exist"

解决方案：

检查Label Studio数据目录位置：
bash
find ~ -name "*label-studio*" -type d | grep -v cache
更新docker-compose.yml中的挂载路径
2. API认证失败
问题：401 Unauthorized错误

解决方案：

这是正常的，系统会自动使用本地文件访问
确保正确挂载了Label Studio数据目录
3. 训练不触发
检查步骤：

确认配置中model_finetune="true"
查看标注计数：
bash
docker exec -it yolo cat /app/training_data/project_1_counter.json
检查是否达到最小标注数量
4. 快速测试
如果想快速测试训练功能，可以降低触发门槛：

xml
model_min_annotations="2"
model_retrain_interval="1"
高级配置
1. 使用GPU加速
修改docker-compose.yml添加GPU支持：

yaml
services:
  yolo:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
2. 调整训练参数
在标注配置中调整：

model_finetune_lr: 学习率（默认0.001）
model_finetune_patience: 早停耐心值（默认5）
3. 重置训练数据
如需重新开始：

bash
docker exec -it yolo rm -rf /app/training_data/project_1/*
docker exec -it yolo rm -f /app/models/finetune_project_1.pt
总结
准备阶段：安装Docker，启动Label Studio，获取API Token
配置阶段：配置docker-compose.yml，准备模型文件
使用阶段：上传图片，修正标注，自动触发训练
监控阶段：查看训练日志和结果
系统会自动管理训练流程，您只需要专注于标注工作！

