# Stamp Detector

Простой инструмент для детекции печатей (stamp) на изображениях с использованием YOLOv8.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Базовое использование

```bash
python detect.py path/to/image.jpg
```

### С кастомным порогом уверенности

```bash
python detect.py path/to/image.jpg --conf 0.20
```

### С указанием пути к модели

```bash
python detect.py path/to/image.jpg --model stamp_model.pt
```

### С указанием выходного файла

```bash
python detect.py path/to/image.jpg --output result.jpg
```

### Сохранение JSON с координатами

```bash
# Сохранить JSON в output/{имя_файла}_result.json
python detect.py path/to/image.jpg --json

# Сохранить JSON в указанный файл
python detect.py path/to/image.jpg --json-output results.json
```

## Параметры

- `image_path` (обязательный) - путь к входному изображению
- `--model` - путь к модели (по умолчанию: `stamp_model.pt`)
- `--output` - путь для сохранения результата (по умолчанию: `output/{имя_файла}_result.jpg`)
- `--conf` - порог уверенности (по умолчанию: 0.25)
- `--json` - сохранить JSON с координатами детекций
- `--json-output` - путь для сохранения JSON файла

## Структура

```
stamp_detector/
├── stamp_model.pt      # Обученная модель YOLOv8
├── detect.py           # Скрипт детекции
├── requirements.txt    # Зависимости
└── README.md          # Документация
```

## Примеры

```bash
# Детекция с порогом 0.25
python detect.py image.jpg

# Более чувствительная детекция (ниже порог)
python detect.py image.jpg --conf 0.15

# Менее чувствительная детекция (выше порог)
python detect.py image.jpg --conf 0.35

# Детекция с сохранением JSON координат
python detect.py image.jpg --json
```

## Формат JSON

При использовании флага `--json` создается JSON файл со следующей структурой:

```json
{
  "image_path": "output/image_result.jpg",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections_count": 2,
  "detections": [
    {
      "class": "stamp",
      "confidence": 0.8542,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400,
        "width": 200,
        "height": 200
      },
      "bbox_normalized": {
        "x1": 0.052083,
        "y1": 0.185185,
        "x2": 0.15625,
        "y2": 0.37037,
        "width": 0.104167,
        "height": 0.185185
      }
    }
  ]
}
```

- `bbox` - абсолютные координаты в пикселях
- `bbox_normalized` - нормализованные координаты (0.0 - 1.0) относительно размера изображения

