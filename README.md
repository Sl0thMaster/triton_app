## Описание

Проект разворачивает модель классификации токсичности текста с использованием Triton Inference Server.
Клиент отправляет текст на сервер, сервер обрабатывает его через модель и возвращает вероятности классов.

Модель загружается через Python backend Triton и использует библиотеку transformers и PyTorch.

---

# Структура проекта

triton_app/

model_repository/
    toxicity_classifier/
        config.pbtxt
        1/
            model.py

client/
    client.py

requirements.txt
Dockerfile
README.md

---

# Требования

Перед запуском необходимо установить:

- Python 3.10+
- Docker
- Git

---

# Установка проекта

Склонировать репозиторий

git clone https://github.com/Sl0thMaster/triton_app.git

Перейти в папку проекта

cd triton_app

---

# Создание виртуального окружения (Windows)

python -m venv venv

Активация окружения

venv\\Scripts\\activate

Установка зависимостей

pip install -r requirements.txt

---

# Сборка Docker образа

docker build -t triton_toxicity .

---

# Запуск сервера Triton

docker run --rm ^
-p 8000:8000 ^
-p 8001:8001 ^
-p 8002:8002 ^
-v %cd%/model_repository:/models ^
triton_toxicity ^
tritonserver --model-repository=/models

После запуска сервер будет доступен по адресу

http://localhost:8000

Проверка состояния сервера

http://localhost:8000/v2/health/ready

Если сервер работает, он вернет

OK

---

# Запуск клиента

Открыть новый терминал в папке проекта.

Активировать виртуальное окружение

venv\\Scripts\\activate

Запустить клиент

python client\\client.py

---

# Пример запроса

Клиент отправляет список текстов:

sample = [
"Привет, как дела?",
"Ты идиот"
]

Ответ сервера содержит:

- исходный текст
- вероятности классов
- предсказанный класс

Пример ответа

[
{'text': 'Привет, как дела?', 'label': 0, 'probs': [...]},
{'text': 'Ты идиот', 'label': 1, 'probs': [...]}
]

---

# Частые ошибки

Ошибка формы входа

unexpected shape for input

Решение:

в config.pbtxt установить

max_batch_size: 0

---

Ошибка PyTorch

AutoModelForSequenceClassification requires the PyTorch library

Решение:

установить PyTorch версии 2.4 или выше

pip install torch>=2.4

---

Ошибка CUDA

CUDA driver version is insufficient for CUDA runtime version

Можно игнорировать, если используется CPU версия PyTorch.

---

# Тестирование

После запуска сервера выполнить

python client\\client.py

Если модель работает корректно, клиент вернет предсказания для тестовых строк.

---

# Автор

Лабораторная работа по развертыванию модели в Triton Inference Server.
"""

path = "/mnt/data/README.md"
pypandoc.convert_text(content, 'md', format='md', outputfile=path, extra_args=['--standalone'])

path
