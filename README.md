TÜRKÇE HABER BAŞLIĞI SINIFLANDIRMA PROJESİ (ANN + TF-IDF)

Bu proje, Türkçe haber başlıklarını otomatik olarak kategorilere ayırmak için yapay sinir ağı (ANN) kullanır. Veriler temizlenip TF-IDF ile vektörleştirilir ve çok katmanlı bir sinir ağı modeliyle sınıflandırılır. Sonuçlar hem metrikler hem de görselleştirmelerle değerlendirilmiştir.

📌 KULLANILAN YÖNTEMLER

TF-IDF vektörleştirme (bigram dahil, 5000 özellik)
Label Encoding + One-Hot Encoding
3 katmanlı Keras Sequential ANN
Eğitim / doğrulama ayrımı ve performans ölçümü
Tahmin çıktısı, grafikler, karışıklık matrisi
📁 PROJE DOSYA YAPISI

TurkishHeadlines.csv → Veri seti (haber başlıkları + etiket) text_cleaner.py → Metin ön işleme fonksiyonu tr_news_ann_main.py → Ana Python scripti tr_news_ann_model.keras → Eğitilmiş ANN modeli tfidf_vectorizer.pkl → Eğitilmiş TF-IDF vektörleştirici label_encoder.pkl → Etiket kodlayıcı (LabelEncoder) requirements.txt → Gerekli pip paketleri README_Yasemin.txt → Bu dosya

🚀 KURULUM VE ÇALIŞTIRMA

Gerekli Python paketlerini kur: pip install -r requirements.txt

Ana Python dosyasını çalıştır: python tr_news_ann_main.py

📊 MODEL ÖZETİ

Model: "tr_news_ann" Toplam parametre: 648,839 Katmanlar:

Dense (128) + Dropout(0.5)
Dense (64)
Çıkış: Softmax
🎯 EĞİTİM SONUÇLARI (Epoch 20)

Eğitim Doğruluğu : %99.84+ Doğrulama Doğruluğu : %95.54 Test Doğruluğu : %97.48 Test Kayıp (Loss) : 0.10

📈 EĞİTİM GRAFİKLERİ

Eğitim vs. Doğrulama Kayıp (Loss) Eğitim vs. Doğrulama Doğruluk (Accuracy)

→ Otomatik olarak matplotlib ile çizilir: plt.plot(hist.history['loss']) plt.plot(hist.history['val_loss'])

🧪 TAHMİN ÖRNEĞİ

Test cümleleri:

"Merkez Bankası faiz oranlarında değişikliğe gitmedi."
"Fenerbahçe Galatasaray derbisi nefes kesti."
"Apple yeni iPhone modelini tanıttı." ...
Modelin çıktısı:

HABER: Merkez Bankası faiz oranlarında değişikliğe gitmedi. TAHMİN: Ekonomi

HABER: Ünlü oyuncu kırmızı halıda verdiği pozlarla gündeme geldi. TAHMİN: Magazin

HABER: Yeni koronavirüs aşısı klinik denemeleri başarıyla tamamladı. TAHMİN: Sağlık

HABER: Uzayda gözlem yapan James Webb teleskopu ilginç görüntüler kaydetti. TAHMİN: Teknoloji

HABER: Fenerbahçe Galatasaray derbisi nefes kesti. TAHMİN: Spor

HABER: Apple yeni iPhone modelini tanıttı. TAHMİN: Teknoloji

HABER: Çiftçiler kuraklıktan dolayı büyük zarar gördü. TAHMİN: Ekonomi

📚 KULLANILAN KÜTÜPHANELER

pandas scikit-learn tensorflow matplotlib seaborn joblib nltk

🧠 GELİŞTİRİCİ

Yasemin
NLP & Görüntü İşleme
GitHub: https://github.com/karacayase

