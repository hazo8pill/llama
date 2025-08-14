import os
import numpy as np
import time
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine

# ==== НАСТРОЙКИ ====
QUERY_IMAGE_PATH = "query.jpg"       # путь к изображению-запросу
IMAGES_FOLDER = "images"      # папка с изображениями для поиска
TOP_N = 3                            # сколько наиболее похожих вывести

# ==== 1. Загружаем модель MobileNetV3Small ====
print("[INFO] Загружаю модель MobileNetV3Small...")
model = MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg")

start = time.time()

# ==== 2. Функция получения эмбеддинга ====
def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array, verbose=0)
    return embedding[0]

# ==== 3. Получаем эмбеддинг для query ====
print("[INFO] Обрабатываю query-изображение...", os.path.join(os.getcwd(), 'src/llvis', QUERY_IMAGE_PATH))
query_emb = get_embedding(os.path.join(os.getcwd(), 'src/llvis', QUERY_IMAGE_PATH))

# ==== 4. Сравниваем с изображениями в папке ====
results = []
print("[INFO] Сравниваю с изображениями в папке...")
for filename in os.listdir(os.path.join(os.getcwd(), 'src/llvis', IMAGES_FOLDER)):
    path = os.path.join(os.path.join(os.getcwd(), 'src/llvis', IMAGES_FOLDER, filename))
    if not os.path.isfile(path):
        continue
    try:
        emb = get_embedding(path)
        similarity = 1 - cosine(query_emb, emb)  # чем ближе к 1, тем больше похожи
        results.append((filename, similarity))
    except Exception as e:
        print(f"[WARN] Ошибка при обработке {filename}: {e}")

# ==== 5. Сортировка по схожести ====
results.sort(key=lambda x: x[1], reverse=True)

# ==== 6. Вывод результата ====
print("\n=== Наиболее похожие изображения ===", results)
for filename, sim in results[:TOP_N]:
    print(f"{filename} — схожесть {sim:.4f}")

print(f"[INFO] Общее время обработки: {time.time() - start:.2f} секунд")
