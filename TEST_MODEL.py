
import cv2
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

from Model_2 import FruitFreshnessModel


YOLO_PATH = r"Classification\Weights\YOLO Weight\best.pt"
MODEL_PATH = r"Classification\Weights\resNet Weight\best_model.pth"
IMAGE_PATH = r"C:\Users\PC\Desktop\MODEL\Fruit Model\random_image\alotapple.png"

IMAGE_SIZE = 224
CONFIDENCE = 0.2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FRESHNESS_LABELS = {
    "Fresh": (0.75, (0, 200, 0)),
    "Mild deterioration": (0.25, (0, 200, 255)),
    "Rotten": (0.0, (0, 0, 255))
}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def freshness_label(score):
    if score >= 0.75:
        return "Fresh", (0, 200, 0)
    elif score >= 0.25:
        return "Mild deterioration", (0, 200, 255)
    return "Rotten", (0, 0, 255)


def main():
    if not Path(YOLO_PATH).is_file():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")

    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"Classifier not found: {MODEL_PATH}")

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = FruitFreshnessModel(
        num_produce_classes=len(class_names),
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    detector = YOLO(YOLO_PATH)

    results = detector.predict(
        source=image,
        conf=CONFIDENCE,
        verbose=False
    )

    counts = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits, score_tensor = model(tensor)
                probabilities = torch.softmax(logits, dim=1)

            class_id = int(probabilities.argmax(dim=1).item())
            produce = class_names[class_id]
            class_confidence = probabilities[0, class_id].item()

            freshness = float(score_tensor.item())
            condition, color = freshness_label(freshness)

            fruit_or_vegetable = "Unknown"

            label = (
                f"{produce} ({class_confidence:.2f}) | "
                f"{condition} ({freshness:.2f})"
            )
            print(produce ,"freshness", freshness)
            cv2.rectangle(
                image, (x1, y1), (x2, y2), color, 2
            )

            text_y = max(25, y1 - 10)

            cv2.putText(
                image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

            counts[condition] = counts.get(condition, 0) + 1

    print("Detected item counts by freshness:")
    print(counts)
    scale = 1
    image = cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
    cv2.imshow("Produce Classification and Freshness", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
