# Классификатор грибов

Проект машинного обучения для идентификации съедобных и ядовитых грибов на основе табличных данных.

## Описание проекта

Цель этого проекта — создание классификатора, способного определять съедобность или ядовитость грибов на основе их характеристик. Модель обучается на наборе данных UCI Mushroom, содержащем категориальные признаки, описывающие различные характеристики грибов.

Проект использует:
- PyTorch и PyTorch Lightning для обучения модели
- Hydra для управления конфигурацией
- DVC для версионирования и управления данными
- MLflow для отслеживания экспериментов
- ONNX и TensorRT для оптимизации модели
- Triton Inference Server для развертывания модели

## Настройка

### Предварительные требования

- Python 3.8+

### Установка

1. Клонировать репозиторий:
```bash
git clone https://github.com/yourusername/mushroom-classifier.git
cd mushroom-classifier
```

2. Создать и активировать виртуальное окружение:
```bash
# Использование venv
python -m venv .venv
source .venv/bin/activate  # На Windows: .venv\Scripts\activate

# ИЛИ использование conda
conda create -n mushroom-classifier python=3.8
conda activate mushroom-classifier
```

3. Установить пакет и зависимости с помощью Poetry:
```bash
# Установить Poetry, если его нет
pip install poetry

# Установить зависимости проекта
poetry install
```

4. Настроить pre-commit хуки:
```bash
pre-commit install
```

## Управление данными с DVC

Проект использует DVC для управления и версионирования набора данных. Данные по умолчанию хранятся в Google Drive.

1. Загрузить данные из настроенного удаленного хранилища:
```bash
dvc pull
```

2. Если вы хотите использовать другое хранилище, вы можете обновить конфигурацию DVC:
```bash
# Для Google Drive
dvc remote add -d storage gdrive://your-gdrive-folder-id
dvc remote modify storage gdrive_acknowledge_abuse true

# Для S3
dvc remote add -d s3storage s3://your-bucket-name
dvc remote modify s3storage endpointurl https://your-endpoint
dvc remote modify s3storage access_key_id your-access-key
dvc remote modify s3storage secret_access_key your-secret-key
```

## Обучение

Обучение модели с использованием конфигурации по умолчанию:

```bash
python -m mushroom_classifier.train
```

Или с пользовательскими параметрами:

```bash
python -m mushroom_classifier.train trainer.max_epochs=50 model.hidden_sizes=[256,128,64]
```

Метрики обучения будут записываться в TensorBoard. Вы можете просмотреть их с помощью:

```bash
tensorboard --logdir logs/tensorboard
```

## Отслеживание экспериментов с MLflow

Запустите сервер MLflow для отслеживания экспериментов:

```bash
python scripts/start_mlflow_server.py --port 8080
```

Затем откройте браузер и перейдите по адресу `http://127.0.0.1:8080`, чтобы просмотреть интерфейс MLflow.

Метрики обучения, параметры и артефакты будут автоматически записываться в MLflow при запуске скрипта обучения.

## Оптимизация модели

### Преобразование в ONNX

Конвертация обученной модели PyTorch в формат ONNX:

```bash
python scripts/convert_to_onnx.py --model_path models/checkpoints/best_model.ckpt --output_path models/exported/model.onnx
```

### Преобразование в TensorRT

Конвертация модели ONNX в TensorRT для более быстрого вывода:

```bash
python scripts/convert_to_tensorrt.py --onnx_path models/exported/model.onnx --tensorrt_path models/exported/model.engine --precision fp16
```

## Инференс

Запуск инференса с использованием обученной модели:

```bash
python -m mushroom_classifier.infer --input path/to/mushroom/features.csv --output predictions.json
```

Для использования модели ONNX для инференса:

```bash
python -m mushroom_classifier.infer model.use_onnx=true --input path/to/mushroom/features.csv
```

## Развертывание с Triton Inference Server

Настройка и запуск Triton Inference Server с вашей моделью:

```bash
python scripts/setup_triton_server.py --model_path models/exported/model.onnx --model_format onnx
```

Это действие:
1. Подготовит репозиторий модели с правильной структурой
2. Настроит модель для Triton
3. Запустит сервер Triton

После запуска сервера вы можете отправлять запросы на инференс по адресу `http://localhost:8000/v2/models/mushroom_classifier/infer` с соответствующей полезной нагрузкой JSON.

Пример запроса на инференс:

```bash
curl -X POST http://localhost:8000/v2/models/mushroom_classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3, 224, 224],
        "datatype": "FP32",
        "data": [...]
      }
    ]
  }'
```

## Как мы проверим вашу работу (шаги верификации)

1. **Настройка и установка**:
   - Клонирование репозитория
   - Создание виртуального окружения
   - Запуск `poetry install`
   - Запуск `pre-commit install`

2. **Управление данными**:
   - Запуск `dvc pull` для получения набора данных
   - Проверка структуры данных в директории `data/`

3. **Обучение**:
   - Запуск `python -m mushroom_classifier.train`
   - Проверка логов TensorBoard в `logs/tensorboard/`
   - Проверка экспериментов MLflow по адресу `http://127.0.0.1:8080`

4. **Экспорт модели**:
   - Проверка создания модели ONNX в `models/exported/`
   - Запуск конвертации в TensorRT

5. **Инференс**:
   - Запуск инференса на тестовых данных
   - Запуск сервера Triton и тестирование API инференса

6. **Качество кода**:
   - Запуск `pre-commit run -a` для проверки качества кода

## Структура проекта

```
mushroom_classifier/
├── configs/               # Файлы конфигурации Hydra
│   ├── config.yaml        # Основная конфигурация
│   ├── model.yaml         # Конфигурация модели
│   ├── preprocessing.yaml # Конфигурация предобработки данных
│   └── training.yaml      # Конфигурация обучения
├── data/                  # Директория данных
│   ├── raw/               # Исходные файлы данных
│   └── processed/         # Обработанные файлы данных
├── logs/                  # Файлы логов
│   ├── mlflow/            # Логи MLflow
│   └── tensorboard/       # Логи TensorBoard
├── models/                # Файлы моделей
│   ├── checkpoints/       # Контрольные точки модели
│   └── exported/          # Экспортированные модели (ONNX, TensorRT)
├── mushroom_classifier/   # Основной пакет
│   ├── __init__.py        # Инициализация пакета
│   ├── data.py            # Загрузка и предобработка данных
│   ├── model.py           # Определение модели
│   ├── train.py           # Скрипт обучения
│   ├── infer.py           # Скрипт инференса
│   └── utils.py           # Вспомогательные функции
├── scripts/               # Вспомогательные скрипты
│   ├── convert_to_onnx.py         # Конвертация модели в ONNX
│   ├── convert_to_tensorrt.py     # Конвертация ONNX в TensorRT
│   ├── setup_triton_server.py     # Настройка сервера Triton
│   └── start_mlflow_server.py     # Запуск сервера MLflow
├── .dvc/                  # Конфигурация DVC
├── .pre-commit-config.yaml # Конфигурация хуков pre-commit
├── pyproject.toml         # Конфигурация Poetry
├── prepare_data.py        # Скрипт подготовки данных
└── README.md              # Документация проекта
```

## Информация о наборе данных

Набор данных UCI Mushroom содержит характеристики 8,124 образцов грибов, каждый из которых классифицирован как съедобный или ядовитый. Характеристики включают:

- Форма, поверхность и цвет шляпки
- Наличие синяков
- Запах
- Прикрепление, расстановка, размер и цвет жабр
- Форма и корень ножки
- Тип и цвет вуали
- Количество и тип колец
- Цвет споровой массы
- Популяция и среда обитания

Для получения дополнительной информации см. [Репозиторий машинного обучения UCI](https://archive.ics.uci.edu/ml/datasets/mushroom).