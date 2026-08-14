# Meme Kanseri Teşhisi - Makine Öğrenmesi Final Ödevi

Bu proje, **Breast Cancer Wisconsin (Diagnostic)** veri setini kullanarak hastaların tümör özelliklerinden kanserin **İyi Huylu (Benign)** veya **Kötü Huylu (Malignant)** olup olmadığını tahmin eden uçtan uca bir Makine Öğrenmesi (ML) sınıflandırma projesidir.

Projede veri analizi, veri ön işleme, öznitelik mühendisliği, model eğitimi, hiperparametre ayarlama ve model açıklanabilirliği adımları ML prensiplerine uygun olarak gerçekleştirilmiştir.

---

## Proje Adımları ve Metodoloji

1. **Veri İnceleme (EDA):** 569 satır ve 31 sütundan oluşan veri seti incelenmiş, 357 Benign (1) ve 212 Malignant (0) sınıf dağılımı tespit edilmiştir.
2. **Eksik ve Aykırı Değer Yönetimi:** Veri setinde eksik değer bulunmamaktadır. IQR (Interquartile Range) yöntemi kullanılarak uç değerler alt ve üst sınırlara baskılanmıştır (Winsorization).
3. **Öznitelik Mühendisliği (Feature Engineering):**
   - `radius_to_texture_ratio` (Yarıçap / Doku oranı)
   - `worst_to_mean_radius_ratio` (En kötü yarıçap / Ortalama yarıçap oranı)
   - `area_category` (Tümör alanına göre Small/Medium/Large kategorisi) öznitelikleri türetilmiştir.
4. **Encoding & Scaling:** Kategorik değişkenler One-Hot Encoding ile dönüştürülmüş, sayısal değişkenler `StandardScaler` ile ölçeklenmiştir (Data leakage önlenmesi amacıyla ölçekleyici yalnızca train setinde fit edilmiştir).
5. **Öznitelik Seçimi (Feature Selection):** `SelectKBest` (ANOVA F-value) kullanılarak 34 öznitelik arasından en anlamlı **15 öznitelik** seçilmiştir.
6. **Veri Bölme:** Veri seti %60 Train (341), %20 Validation (114) ve %20 Test (114) olacak şekilde Stratified (sınıf oranları korunarak) olarak ayrılmıştır.

---

## Model Performansı ve Karşılaştırma

### Validation Kümesi Sonuçları
Eğitilen 3 farklı modelin Validation kümesindeki performansları:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9474** | **0.9452** | **0.9718** | **0.9583** |
| **Random Forest** | **0.9474** | **0.9452** | **0.9718** | **0.9583** |
| **K-Nearest Neighbors** | 0.9298 | 0.9315 | 0.9577 | 0.9444 |

### Hiperparametre Optimizasyonu (GridSearch)
Random Forest modeli üzerinde yapılan 5-Fold Cross Validation GridSearch sonucunda en iyi hiperparametreler belirlenmiştir:
- `max_depth`: `None`
- `min_samples_split`: `5`
- `n_estimators`: `100`
- **Best Cross-Validation F1-Score:** `0.9584`

---

## Sonuçlar

Optimize edilen modelin 114 örneklik hiç görülmemiş Test kümesi üzerindeki metrikleri:

- **Test Accuracy:** `0.9649` (%96.49)
- **Test Precision:** `0.9722` (%97.22)
- **Test Recall:** `0.9722` (%97.22)
- **Test F1-Score:** `0.9722` (%97.22)
- **Test ROC-AUC:** `0.9911` (%99.11)

### Karmaşıklık Matrisi (Confusion Matrix)
```text
[40  2]  -> Malignant (0): 40 Doğru, 2 Yanlış
[ 2 70]  -> Benign (1)   : 70 Doğru, 2 Yanlış
```
---
### Kurulum ve Çalıştırma

   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/REPO.git]
   cd REPO
   pip install -r requirements.txt
   python main.py
