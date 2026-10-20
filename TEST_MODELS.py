
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from Model_att import FruitFreshnessModel

YOLO_PATH = r"C:\Users\PC\Desktop\MODEL\Fruit Model\runs\detect\train-6\weights\best.pt"
FRESHNESS_PATH = r"C:\Users\PC\Desktop\MODEL\Fruit Model\best_model.pth"
IMAGE_PATH = r"C:\Users\PC\Desktop\MODEL\Fruit Model\random_image\images (2).jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = YOLO(YOLO_PATH)

freshness_model = FruitFreshnessModel(pretrained=False).to(device)

checkpoint = torch.load(
    FRESHNESS_PATH,
    map_location=device,
    weights_only=True
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

freshness_model.load_state_dict(state_dict)
freshness_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

results = detector.predict(image, verbose=False)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)
        tensor = transform(crop).unsqueeze(0).to(device)

        with torch.inference_mode():
            score = freshness_model(tensor).item()

        class_id = int(box.cls[0])
        fruit_name = result.names[class_id]
        confidence = float(box.conf[0])

        condition = "Fresh" if score >= 0.5 else "Rotten"

        label = (
            f"Det: {confidence:.2f} | "
            f"Freshness: {score:.2f}"
        )

        color = (0, 200, 0) if score >= 0.5 else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            2
        )

        print(
            f"Detection: {confidence:.3f} | "
            f"Freshness: {score:.3f} | "
            f"Condition: {condition}"
        )

h, w = image.shape[:2]
scale = 1
image = cv2.resize(image, (int(w * scale), int(h * scale)))
cv2.imshow("Fruit Freshness Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
