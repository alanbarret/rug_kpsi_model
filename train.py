from ultralytics import YOLO

# Load a model
model = YOLO("base_models\yolo11m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"C:\\Users\DELL\Documents\\MVP\\3ds_decryption\\rug_kpsi_model\\dataset\\data.yaml", epochs=100, imgsz=640)
print(results)
