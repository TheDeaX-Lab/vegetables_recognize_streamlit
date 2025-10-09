import numpy as np
import cv2
import streamlit as st
import pandas as pd
import time

import torch
from torchvision import transforms, models
import torch.nn as nn

num_classes = 15
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model.load_state_dict(torch.load('vegetables_cnn.pth', map_location=torch.device('cpu')))  # или 'cuda'
model.eval()

# Загружаем все модели при первом запуске (опционально — можно загружать по требованию)
# Но лучше загружать по требованию, чтобы не тратить память
# models = {name: load_model_by_name(name) for name in MODEL_PATHS.keys()}  # если хочешь все сразу

# Определяем метки классов — ПОРЯДОК ВАЖЕН!
class_labels = list({0: 'Бобы', 1: 'Горький огурец', 2: 'Кабачок', 3: 'Баклажан', 4: 'Брокколи', 5: 'Капуста', 6: 'Сладкий перец', 7: 'Морковь', 8: 'Цветная капуста', 9: 'Огурец', 10: 'Папайя', 11: 'Картофель', 12: 'Тыква', 13: 'Редис', 14: 'Помидор'}.values())

# =============================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ (с передачей модели)
# =============================
from PIL import Image
def predict_image(image_array: np.ndarray, model) -> tuple:
    """
    Принимает numpy-массив изображения (H, W, 3) в BGR или RGB (предпочтительно RGB),
    возвращает (метку, уверенность, все вероятности).
    """
    # Убедитесь, что изображение в RGB (OpenCV читает как BGR)
    if image_array.shape[-1] == 3:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_array  # уже grayscale или RGB

    # Преобразуем numpy -> PIL Image (обязательно для transforms.ToTensor())
    pil_img = Image.fromarray(img_rgb.astype('uint8'), 'RGB')

    # Трансформы (точно такие же, как при валидации!)
    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Применяем трансформы → получаем тензор (C, H, W)
    tensor_img = val_test_transforms(pil_img)

    # Добавляем batch dimension: (C, H, W) → (1, C, H, W)
    tensor_img = tensor_img.unsqueeze(0)

    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # [num_classes]

    # Получаем результаты
    predicted_class_index = np.argmax(probabilities)
    confidence = probabilities[predicted_class_index]
    predicted_label = class_labels[predicted_class_index]

    return predicted_label, confidence, probabilities

# =============================
# STREAMLIT ИНТЕРФЕЙС
# =============================
# Заголовок и описание
st.title("🥦 Классификация овощей")
st.write("Загрузите изображение овоща, чтобы определить его вид.")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Преобразуем в массив с помощью OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR формат

    if image is None:
        st.error("Не удалось загрузить изображение. Попробуйте другой файл.")
    else:
        # Показываем изображение
        st.image(image, channels="BGR", caption="Загруженное изображение", use_column_width=True)

        # Делаем предсказание
        with st.spinner("Анализируем изображение..."):
            start_time = time.time()
            label, confidence, probs = predict_image(image, model)
            elapsed_time = time.time() - start_time

        # Выводим результат
        st.info(f"**Время прогноза**: {elapsed_time:.2f} секунд")
        st.success(f"**Овощ:** {label}")
        st.info(f"**Уверенность модели:** {confidence:.4f}")

        # Вывод распределения вероятностей
        st.write("### Вероятности для всех классов:")
        prob_df = pd.DataFrame({
            'Овощ': class_labels,
            'Вероятность': probs
        }).sort_values('Вероятность', ascending=False).reset_index(drop=True)
        prob_df['Овощ'] = pd.Categorical(prob_df['Овощ'], categories=prob_df['Овощ'], ordered=True)
        st.bar_chart(prob_df.set_index('Овощ'))