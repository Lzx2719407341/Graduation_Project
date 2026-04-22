from ultralytics import YOLO

model = YOLO('LV_Pole_Project/exp_finetune_v2/weights/best.pt')
# 在验证集上跑推理，设置低阈值看看能不能抓回漏掉的杆子
results = model.predict(source='data/final_dataset/val/images', save=True, conf=0.15, iou=0.45)