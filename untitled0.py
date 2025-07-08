# main.py
from tensorflow.keras.models import load_model
import joblib
import pickle
import numpy as np
from text_cleaner import clean_text  # dış dosyadan fonksiyon içe aktarılır

# 📌 1. Girdi verisi (tahmin edilecek haber başlıkları)
new_data = [
    "Cumhurbaşkanı yeni kabineyi açıkladı",
    "Dolar kuru rekor tazeledi",
    "5G altyapısı için çalışmalar başladı",
    "Uluslararası film festivali İstanbul’da başladı",
    "Polis uyuşturucu operasyonunda 12 kişiyi gözaltına aldı"
]

# 📌 2. Temizle
new_data_cleaned = [clean_text(text) for text in new_data]

# 📌 3. Model, TF-IDF ve LabelEncoder yükle
model = load_model("tr_news_ann_model.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 📌 4. Vektörleştir
X_input = tfidf.transform(new_data_cleaned)

# 📌 5. Tahmin et
predictions = model.predict(X_input)
predicted_indices = np.argmax(predictions, axis=1)

# 📌 6. Etiketlere çevir
# Eğer label_encoder bir np.array gibi görünüyorsa:
if isinstance(label_encoder, np.ndarray):
    predicted_labels = label_encoder[predicted_indices]
else:
    predicted_labels = label_encoder.inverse_transform(predicted_indices)

# 📌 7. Yazdır
for original, label in zip(new_data, predicted_labels):
    print(f"{original} --> Tahmin: {label}")
