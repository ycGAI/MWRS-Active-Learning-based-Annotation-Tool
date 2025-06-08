第1部分：
first part:
1.1： 
bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/label_studio_ml/examples/yolo

1.2:
把模型权重(.pt文件)拷贝到label-studio-ml-backend/label_studio_ml/examples/yolo下的/models/路径下
copy the model weight(.pt file) to label-studio-ml-backend/label_studio_ml/examples/yolo/models/
cp /path/to/your/model.pt ./models/best.pt

1.3：
启动 Docker Compose
start docker compose
docker-compose up
# 验证启动成功（看到以下信息表示成功）：
# Verify that the startup is successful (seeing the following information indicates success)
[INFO] Starting gunicorn
[INFO] Listening at: http://0.0.0.0:9090
[INFO] Loading yolo model: /app/models/best.pt

1.4：
# 在新终端中验证模型类别
# Verify the model class in a new terminal
docker exec -it yolo python3 -c "
from ultralytics import YOLO
model = YOLO('/app/models/best.pt')
print('模型类别:', model.names)
print('类别数量:', len(model.names))
"
docker exec -it yolo python3 -c "
from ultralytics import YOLO
model = YOLO('/app/models/best.pt')
print('Model Category:', model.names)
print('Number of categories:', len(model.names))
"


# 验证 ML Backend 健康状态
# Verify ML Backend Health
curl http://localhost:9090/health
# 期望返回: {"status": "UP"}
# Expected return: {"status": "UP"}

第二部分 启动label studio:
second part start label studio:
# 准备数据目录
# Prepare the data directory
mkdir -p ~/label-studio-data
sudo chown -R $USER:$USER ~/label-studio-data
chmod -R 755 ~/label-studio-data

# 启动 Label Studio（使用 8081 端口避免冲突）
# Start Label Studio (use port 8081 to avoid conflicts)
docker run -d -p 8081:8080 \
  -v ~/label-studio-data:/label-studio/data \
  --user $(id -u):$(id -g) \
  --name label-studio-plant \
  heartexlabs/label-studio:latest

# 验证启动
# Verify Boot
docker logs -f label-studio-plant
# 看到 "Starting development server at http://0.0.0.0:8080/" 表示成功
# Seeing "Starting development server at http://0.0.0.0:8080/" indicates success

第三部分 配置 Label Studio 项目：
third part Configure Label Studio Project:
Create Project
Computer Vision → Object Detection
Labeling Interface" → "Code"

<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image" 
                   model_path="best.pt"
                   model_score_threshold="0.5"
                   model_conf_threshold="0.25">
    <Label value="Armeria Maritima" predicted_values="armeria-maritima" background="#FF6B6B"/>
    <Label value="Centaurea Jacea" predicted_values="centaurea-jacea" background="#4ECDC4"/>
    <Label value="Cirsium Oleraceum" predicted_values="cirsium-oleraceum" background="#45B7D1"/>
    <Label value="Daucus Carota" predicted_values="daucus-carota" background="#96CEB4"/>
    <Label value="Knautia Arvensis" predicted_values="knautia-arvensis" background="#FFEAA7"/>
    <Label value="Lychnis Flos Cuculi" predicted_values="lychnis-flos-cuculi" background="#DDA0DD"/>
  </RectangleLabels>
</View>

project-settings-model-connect model
http://host.docker.internal:9090
(http://localhost:9090
http://172.17.0.1:9090)
如果第一个URL不行就尝试后面两个
if the first URL don't work, try the following two
No Authentication
Interactive preannotations