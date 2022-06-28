# Converting ONNX model to Dynamic Batch Size & FP16

- This script will modify the onnx model to fp16 & dynamic batch.
- Inside the ultralytics/yolov5 repo, it provides script to export onnx model
    - It allows either export fp16 onnx model or fp32 onnx model with dynamic batch size.
    - But it does not support export model with both dynamic and fp16.
- One easy way is export the model to fp16 onnx first, and modify the model to dynamic batch size. Or export the model with dynamic batch size first and convert to fp16.
- In this script, we will show both without using the export.py
- Before we start, we neend to install some dependencies

```
pip3 install -r requirements.txt
```

- Now we can execute the onnx_model_modifier.py to modify the batch 8 fp32 yolov5s model to dynamic batch fp16 model.

- Before conversion, the batch8 model looks like: 
    - ![before](img/before.png)
- After conversion, it should be like:
    - ![after](img/after.png)
