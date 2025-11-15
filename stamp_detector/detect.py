"""
Простой скрипт для детекции печатей (stamp)
Требуется только: модель и изображение
"""
import cv2
import os
import sys
import json
from ultralytics import YOLO


def detect_stamps_no_save(image_path, model_path="stamp_model.pt", conf=0.25):
    """
    Detect stamps without saving images.
    
    Args:
        image_path: Path to input image
        model_path: Path to model (or will download from HF Hub if not found)
        conf: Confidence threshold
        
    Returns:
        dict: Detection results with detections and image_size
    """
    # Load model - try to download from HF Hub if not found locally
    if not os.path.exists(model_path):
        # Try to download from Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
            print(f"Model not found locally, attempting to download from HF Hub...")
            # You can upload your model to HF Hub and use it here
            # For now, try the default path in stamp_detector directory
            default_path = os.path.join("stamp_detector", "stamp_model.pt")
            if os.path.exists(default_path):
                model_path = default_path
            else:
                raise FileNotFoundError(f"Stamp model not found: {model_path}. Please upload stamp_model.pt to the Space.")
        except ImportError:
            raise FileNotFoundError(f"Stamp model not found: {model_path}")
    
    model = YOLO(model_path)
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Detection
    results = model(image, conf=conf, verbose=False)
    
    # Collect detections
    detections = []
    image_height, image_width = image.shape[:2]
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Filter only stamp (class_id == 0)
            if class_id == 0 and confidence >= conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detection = {
                    "class": "stamp",
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    },
                    "bbox_normalized": {
                        "x1": round(x1 / image_width, 6),
                        "y1": round(y1 / image_height, 6),
                        "x2": round(x2 / image_width, 6),
                        "y2": round(y2 / image_height, 6),
                        "width": round((x2 - x1) / image_width, 6),
                        "height": round((y2 - y1) / image_height, 6)
                    }
                }
                detections.append(detection)
    
    return {
        "image_size": {
            "width": image_width,
            "height": image_height
        },
        "detections_count": len(detections),
        "detections": detections
    }


def detect_stamps(image_path, model_path="stamp_model.pt", output_path=None, conf=0.25, return_json=False):
    """
    Детектирует печати на изображении

    Args:
        image_path: путь к входному изображению
        model_path: путь к модели (по умолчанию: stamp_model.pt)
        output_path: путь для сохранения результата (если None, создается автоматически)
        conf: порог уверенности (по умолчанию: 0.25)
        return_json: если True, возвращает также JSON с координатами

    Returns:
        если return_json=False: путь к сохраненному изображению
        если return_json=True: словарь с 'image_path' и 'detections' (JSON структура)
    """
    # Загружаем модель
    if not os.path.exists(model_path):
        print(f"❌ Ошибка: модель не найдена: {model_path}")
        sys.exit(1)

    print(f"📥 Загружаю модель: {model_path}")
    model = YOLO(model_path)
    print("✅ Модель загружена")

    # Загружаем изображение
    if not os.path.exists(image_path):
        print(f"❌ Ошибка: изображение не найдено: {image_path}")
        sys.exit(1)

    print(f"📷 Загружаю изображение: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Ошибка: не удалось загрузить изображение")
        sys.exit(1)

    # Детекция
    print(f"🔍 Выполняю детекцию (порог: {conf})...")
    results = model(image, conf=conf, verbose=False)

    # Собираем детекции и рисуем рамки
    result_image = image.copy()
    detections = []
    image_height, image_width = image.shape[:2]

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Фильтруем только stamp (class_id == 0)
            if class_id == 0 and confidence >= conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Сохраняем детекцию в JSON формате
                detection = {
                    "class": "stamp",
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    },
                    "bbox_normalized": {
                        "x1": round(x1 / image_width, 6),
                        "y1": round(y1 / image_height, 6),
                        "x2": round(x2 / image_width, 6),
                        "y2": round(y2 / image_height, 6),
                        "width": round((x2 - x1) / image_width, 6),
                        "height": round((y2 - y1) / image_height, 6)
                    }
                }
                detections.append(detection)

                # Рисуем рамку (красная)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Подпись
                label = f"stamp {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    result_image,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

    # Сохраняем результат
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")

    cv2.imwrite(output_path, result_image)
    print(f"✅ Найдено печатей: {len(detections)}")
    print(f"📁 Результат сохранен: {output_path}")

    # Возвращаем результат
    if return_json:
        result_data = {
            "image_path": output_path,
            "image_size": {
                "width": image_width,
                "height": image_height
            },
            "detections_count": len(detections),
            "detections": detections
        }
        return result_data
    else:
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Детекция печатей на изображениях")
    parser.add_argument("image_path", help="Путь к изображению")
    parser.add_argument(
        "--model",
        default="stamp_model.pt",
        help="Путь к модели (по умолчанию: stamp_model.pt)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Путь для сохранения результата (по умолчанию: output/{имя_файла}_result.jpg)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности (по умолчанию: 0.25)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Сохранить JSON с координатами детекций"
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Путь для сохранения JSON файла (по умолчанию: output/{имя_файла}_result.json)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 Детекция печатей (stamp)")
    print("=" * 60)

    result = detect_stamps(
        args.image_path,
        args.model,
        args.output,
        args.conf,
        return_json=args.json or args.json_output is not None
    )

    # Сохраняем JSON если нужно
    if args.json or args.json_output is not None:
        if isinstance(result, dict):
            json_data = {
                "image_path": result["image_path"],
                "image_size": result["image_size"],
                "detections_count": result["detections_count"],
                "detections": result["detections"]
            }
        else:
            # Если result - это путь, нужно пересчитать
            result = detect_stamps(
                args.image_path,
                args.model,
                args.output,
                args.conf,
                return_json=True
            )
            json_data = {
                "image_path": result["image_path"],
                "image_size": result["image_size"],
                "detections_count": result["detections_count"],
                "detections": result["detections"]
            }

        # Определяем путь для JSON
        if args.json_output:
            json_path = args.json_output
        else:
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"{base_name}_result.json")

        # Сохраняем JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"📄 JSON сохранен: {json_path}")

    print("=" * 60)
