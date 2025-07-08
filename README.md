**Label Studio ML Backend Local Deployment Guide**
```bash
git clone https://github.com/ycGAI/MWRS-Active-Learning-based-Annotation-Tool
cd MWRS-Active-Learning-based-Annotation-Tool
```

**Configure the username, password and token in .env file**

```bash
docker-compose up
```

<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.25" model_show_confidence="true">
    <Label value="Armeria Maritima" background="#FF0000"/>
    <Label value="Centaurea Jacea" background="#00FF00"/>
    <Label value="Cirsium Oleraceum" background="#0000FF"/>
    <Label value="Daucus Carota" background="#FFFF00"/>
    <Label value="Knautia Arvensis" background="#FF00FF"/>
    <Label value="Lychnis Flos Cuculi" background="#00FFFF"/>
  </RectangleLabels>
  <Text name="confidence_text" toName="image" readonly="true"/>
  <TextArea name="scores" toName="image" placeholder="Confidence scores will appear here" rows="1" maxSubmissions="0" editable="false"/>
</View>

**Verify service operation status**
```bash
$ curl http://localhost:9090/
```
{"status":"UP"}

**Adding models in Label Studio project settings**
Enter the URLs (try in order):

http://host.docker.internal:9090
http://localhost:9090
http://172.17.0.1:9090