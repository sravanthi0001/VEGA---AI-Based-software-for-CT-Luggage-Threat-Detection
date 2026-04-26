from ultralytics import YOLO

# Path to the last checkpoint (update if your checkpoint is elsewhere)
checkpoint_path = 'runs/train/yolov8n_custom/weights/last.pt'

# Load the model from the last checkpoint
model = YOLO(checkpoint_path)

# Resume training for 10 more epochs
results = model.train(
    data='data.yaml',
    epochs=10,  # Train for 10 more epochs
    imgsz=640,
    resume=True,
    project='runs/train',
    name='yolov8n_custom_resume',
    verbose=True
)

print("Training resumed and complete.")
print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")