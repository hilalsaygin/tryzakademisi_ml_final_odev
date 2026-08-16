# Makine Öğrenmesi Final Ödevi — Meme Kanseri Teşhis Tahmini

## Projenin Amacı
Bu proje, uçtan uca bir makine öğrenmesi sürecini (veri inceleme → veri
ön işleme → öznitelik mühendisliği → öznitelik seçimi → model eğitimi →
model karşılaştırma → hiperparametre ayarlama → test değerlendirmesi →
sonuç yorumu) tek bir Python scripti üzerinde uygulamayı amaçlamaktadır.

Çözülen problem **hastalık tahmini / ikili sınıflandırma (binary
classification)** problemidir: hücre çekirdeği görüntülerinden
hesaplanan sayısal ölçümlere bakarak bir tümörün **malignant** (kötü
huylu) mu yoksa **benign** (iyi huylu) mu olduğunu tahmin etmek.

## Veri Seti Açıklaması
- **Kaynak:** `sklearn.datasets.load_breast_cancer` (Wisconsin Breast
  Cancer Diagnostic Dataset)
- **Boyut:** 569 satır, 30 sayısal özellik + 1 hedef değişken
- **Özellikler:** Hücre çekirdeğinin yarıçapı, dokusu, çevresi, alanı,
  pürüzlülüğü, kompaktlığı, konkavlığı, simetrisi ve fraktal boyutu
  için hesaplanan `mean` (ortalama), `se` (standart hata) ve `worst`
  (en kötü/en büyük 3 değerin ortalaması) istatistikleri
- **Hedef değişken (`target`):** `0 = malignant` (kötü huylu),
  `1 = benign` (iyi huylu)
- Veri setinde eksik değer bulunmamaktadır ve tüm özellikler zaten
  sayısaldır (kategorik değişken yoktur).

## Uygulanan Adımlar (Özet)
1. **Veri inceleme:** `head()`, `shape`, `dtypes`, `describe()`, sınıf
   dağılımı incelemesi
2. **Eksik değer kontrolü:** Eksik değer bulunmadı; ileride oluşabilecek
   eksikler için medyan doldurma kodu hazır tutuldu
3. **Kategorik encoding:** Veri setinde kategorik değişken olmadığı
   tespit edildi; genel amaçlı `pd.get_dummies` mekanizması hazır
   tutuldu
4. **Aykırı değer analizi:** IQR yöntemiyle incelendi; tıbbi açıdan
   anlamlı oldukları için silinmedi, %1–%99 persentillerine
   **winsorize (capping)** edildi
5. **Ölçekleme:** `StandardScaler` — veri sızıntısını önlemek için
   yalnızca train verisiyle `fit` edildi
6. **Öznitelik mühendisliği (3 yeni özellik):**
   - `radius_worst_mean_ratio` = worst radius / mean radius
   - `shape_irregularity_index` = concavity × compactness × symmetry
   - `area_range` = worst area − mean area
7. **Öznitelik seçimi:** Korelasyon analizi + `SelectKBest` (ANOVA
   F-testi) ile en anlamlı 15 özellik seçildi
8. **Train / Validation / Test ayrımı:** %70 / %15 / %15, `stratify=y`
   ile sınıf dengesi korunarak
9. **Model eğitimi (4 model):** Logistic Regression, KNN, Decision
   Tree, Random Forest
10. **Validation karşılaştırması:** Accuracy, Precision, Recall,
    F1-Score, ROC-AUC (tek split) + 5 katlı Stratified Cross-Validation.
    **Model seçimi tek bir küçük validation split'ine (85 satır) değil,
    çok daha güvenilir olan 5 katlı CV ortalama F1 skoruna göre
    yapılmıştır** — küçük örneklemde modeller arası ince farklar
    rastlantısal olabileceği için.
11. **Hiperparametre ayarlama:** CV'ye göre en iyi/en kararlı model
    için `GridSearchCV`
12. **Test değerlendirmesi:** Confusion matrix, Accuracy, Precision,
    Recall, F1-Score (hem benign hem malignant sınıfı için ayrı ayrı)
13. **Açıklanabilirlik (bonus):** Model katsayıları / feature
    importance görselleştirmesi

## Nasıl Çalıştırılır
```bash
# (opsiyonel) sanal ortam
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# kütüphaneleri kur
pip install -r requirements.txt

# scripti çalıştır
python breast_cancer_ml.py
```
Script çalıştıktan sonra tüm analiz adımları konsola yazdırılır ve
aşağıdaki grafikler `outputs/` klasörüne kaydedilir:
- `outlier_boxplots.png`
- `correlation_heatmap.png`
- `model_comparison.png`
- `confusion_matrix.png`
- `feature_importance.png`

## Kısa Sonuç Yorumu
Tek seferlik validation split'inde 4 model birbirine çok yakın F1
skorları verdi (Logistic Regression 0.9907, Random Forest 0.9905, KNN
0.9815, Decision Tree 0.9808) — bu farklar 85 satırlık küçük bir
örneklemde rastlantısal olabileceği için, **model seçimi 5 katlı
Stratified Cross-Validation ortalama F1 skoruna göre yapıldı**:

| Model | Val F1 (tek split) | CV F1 (ortalama ± std) |
|---|---|---|
| **KNN** ✅ | 0.9815 | **0.9704 ± 0.0086** |
| Random Forest | 0.9905 | 0.9656 ± 0.0213 |
| Logistic Regression | 0.9907 | 0.9640 ± 0.0233 |
| Decision Tree | 0.9808 | 0.9598 ± 0.0193 |

**KNN**, tek split'te en yüksek skoru almasa da CV'de hem en yüksek
ortalama F1'i hem de en düşük standart sapmayı (en kararlı model)
verdiği için seçildi. `GridSearchCV` ile (`n_neighbors=3,
weights='uniform'`) ayarlandıktan sonra, model **test verisinde** şu
sonuçları elde etti:

| Metrik (benign sınıfı) | Değer |
|---|---|
| Accuracy | 0.930 |
| Precision | 0.914 |
| Recall | 0.981 |
| F1-Score | 0.946 |
| ROC-AUC | 0.931 |

Bu problemde asıl kritik metrik **malignant (kötü huylu) sınıfının
recall değeridir (0.844)** — confusion matrix'e göre 5 kötü huylu
vaka yanlışlıkla "iyi huylu" olarak sınıflandırılmıştır. Klinik bir
uygulamada bu tür yanlış negatifler, yanlış pozitiflerden çok daha
maliyetlidir; bu nedenle gerçek bir kullanımda karar eşiği recall'i
artıracak şekilde aşağı çekilebilir veya `class_weight='balanced'`
denenebilir.

KNN'in doğrudan bir katsayı/`feature_importances_` çıktısı olmadığı
için açıklanabilirlik, test seti üzerinde **permutation importance**
ile hesaplandı. En etkili özellikler türetilen `radius_worst_mean_ratio`
(açık ara en etkili özellik), `area error` ve `mean concavity`'dir;
türetilen `area_range` özelliği de ilk 5 içinde yer almaktadır — bu da
"worst" (en agresif bölge) ile "mean" (ortalama) ölçüleri arasındaki
oranın/farkının, ham özniteliklerin tek başına sağlayamadığı ek ayırt
edici bilgi taşıdığını ve yapılan öznitelik mühendisliğinin değerini
göstermektedir.

**Modelin sınırlılıkları:** Veri seti nispeten küçüktür (569 satır),
tek bir kaynaktan toplanmıştır ve yalnızca görüntüden türetilen
sayısal ölçümlere dayanır (klinik geçmiş bilgisi içermez). Bu nedenle
model bir **karar destek aracı** olarak değerlendirilebilir.

## Dosya Yapısı
```
.
├── breast_cancer_ml.py     # Ana ML pipeline scripti
├── requirements.txt        # Gerekli kütüphaneler
├── README.md                # Bu dosya
└── outputs/                 # Script çalıştırıldığında üretilen grafikler
```
