"""
MAKİNE ÖĞRENMESİ FİNAL ÖDEVİ PROJESİ
===============================================================================
Proje Amacı:
    Meme Kanseri Wisconsin Veri Seti (Breast Cancer Dataset) kullanılarak tümörün
    iyi huylu (Benign) veya kötü huylu (Malignant) olup olmadığını tahmin eden
    uçtan uca sınıflandırma modeli geliştirmek.

Kullanılan Kütüphaneler:
    - pandas, numpy (Veri işleme ve analizi)
    - matplotlib, seaborn (Görselleştirme)
    - scikit-learn (Veri ön işleme, modelleme, hiperparametre ayarlama, metrikler)

Çalıştırma Adımları:
    1. Gerekli kütüphaneleri yükleyin: pip install -r requirements.txt
    2. Python dosyasını çalıştırın: python main.py
Author : Hilal Saygın
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')


# ADIM 2 & 3: VERİ SETİNİ YÜKLEME VE PROBLEM TANIMI
print("--- ADIM 2 & 3: VERİ SETİ YÜKLEME VE PROBLEM TANIMI ---")
data = load_breast_cancer(as_frame=True)
df = data.frame

# Target Değişkeni: 0 = Malignant (Kötü Huylu), 1 = Benign (İyi Huylu)
# Analiz kolaylığı açısından 'target' adını koruyoruz.
print("Problem Türü: İkili Sınıflandırma (Binary Classification)")
print("Hedef Değişken (Target): 'target' (0: Kötü Huylu, 1: İyi Huylu)")
print(f"Sınıf Dağılımı:\n{df['target'].value_counts()}\n")


# ADIM 4: VERİ ANALİZİ (EDA)
print("--- ADIM 4: TEMEL VERİ İNCELEMESİ ---")
print(f"Veri Seti Boyutu (Satır, Sütun): {df.shape}")
print("\nİlk 5 Satır:")
print(df.head())

print("\nVeri Tipleri Özeti:")
print(df.dtypes.value_counts())

print("\nTemel İstatistiksel Özet:")
print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])


# ADIM 5: PRE-PROCESSING ADMILARI
print("\n--- ADIM 5: EKSİK DEĞER KONTROLÜ ---")
missing_count = df.isnull().sum().sum()
print(f"Toplam Eksik Değişken Sayısı: {missing_count}")

if missing_count > 0:
    # medyan ile doldur
    df = df.fillna(df.median())
    print("Eksik değerler medyan ile dolduruldu.")
else:
    print("Veri setinde eksik değer bulunmamaktadır.")


# ADIM 9: FEATURE ENGINEERING
print("\n--- ADIM 9: ÖZNİTELİK MÜHENDİSLİĞİ ---")
# 1. Yarıçap / Doku Oranı (Radius to Texture Ratio)
df['radius_to_texture_ratio'] = df['mean radius'] / df['mean texture']

# 2. En Kötü Değer / Ortalama Değer Oranı (Worst to Mean Radius Ratio)
df['worst_to_mean_radius_ratio'] = df['worst radius'] / df['mean radius']

# 3. Kategorik Değişken Üretimi : Alan Katmanlandırması
df['area_category'] = pd.qcut(df['mean area'], q=3, labels=['Small', 'Medium', 'Large'])

print("Yeni üretilen öznitelikler eklendi:")
print(" - radius_to_texture_ratio")
print(" - worst_to_mean_radius_ratio")
print(" - area_category (Kategorik)")


# ADIM 6: KATEGORİK DEĞİŞKEN ENCODING
print("\n--- ADIM 6: KATEGORİK DEĞİŞKEN ENCODING ---")
# pd.get_dummies / One-Hot Encoding uygulaması
df = pd.get_dummies(df, columns=['area_category'], drop_first=True, dtype=int)
print("Categorical 'area_category' değişkeni One-Hot Encoding ile sayısal forma dönüştürüldü.")


# ADIM 7: AYKIRI DEĞER İNCELEMESİ VE SINIRLANDIRMA (WINSORIZATION)
print("\n--- ADIM 7: AYKIRI DEĞER İNCELEMESİ VE BASKILAMA ---")
def cap_outliers_iqr(dataframe, feature):
    Q1 = dataframe[feature].quantile(0.25)
    Q3 = dataframe[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataframe[feature] = np.clip(dataframe[feature], lower_bound, upper_bound)
    return dataframe

# Sayısal sütunlarda uç değerleri sınırlandırma
numeric_cols = [c for c in df.columns if c != 'target']
for col in numeric_cols:
    df = cap_outliers_iqr(df, col)

print("IQR yöntemi kullanılarak uç değerler alt ve üst sınırlara baskılandı (Winsorization).")


# ADIM 10: FEATURE SELECTION
print("\n--- ADIM 10: ÖZNİTELİK SEÇİMİ ---")
X_raw = df.drop(columns=['target'])
y_raw = df['target']

# SelectKBest (ANOVA F-value tabanlı) ile en önemli 15 öznitelik seçimi
selector = SelectKBest(score_func=f_classif, k=15)
X_selected_np = selector.fit_transform(X_raw, y_raw)

selected_features = X_raw.columns[selector.get_support()].tolist()
X = X_raw[selected_features]
y = y_raw

print(f"Toplam öznitelik sayısı ({X_raw.shape[1]}) üzerinden en önemli {len(selected_features)} öznitelik seçildi.")
print("Seçilen Öznitelikler:", selected_features[:5], "... ve daha fazlası.")


# ADIM 11: VERİ KÜMESİNİ TRAIN, VALIDATION VE TEST OLARAK AYIRMA
print("\n--- ADIM 11: TRAIN - VALIDATION - TEST AYRIMI ---")
# %60 Train, %20 Validation, %20 Test olacak şekilde Stratified ayırım yapıyoruz.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Train Kümesi     : {X_train.shape[0]} örnek")
print(f"Validation Kümesi: {X_val.shape[0]} örnek")
print(f"Test Kümesi       : {X_test.shape[0]} örnek")


# ADIM 8: VERİ ÖLÇEKLEME (SCALING)
print("\n--- ADIM 8: ÖZEL SAYISAL ÖLÇEKLEME (STANDARD SCALER) ---")
scaler = StandardScaler()
# Data Leakage önlemek amacıyla scaler SADECE train verisinde fit edilir!
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("Standart ölçekleme (StandardScaler) başarıyla uygulandı.")


# ADIM 12 & 13: MODEL EĞİTİMİ VE VALIDATION KARŞILAŞTIRMASI
print("\n--- ADIM 12 & 13: MODELLERİN EĞİTİMİ VE VALIDATION DEĞERLENDİRMESİ ---")

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

val_results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    val_preds = model.predict(X_val_scaled)
    
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    
    val_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

val_df = pd.DataFrame(val_results)
print("\nValidation Kümesi Karşılaştırma Tablosu:")
print(val_df.to_string(index=False))

best_model_name = val_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
print(f"\nEn Yüksek Validation Performansı Gösteren Model: {best_model_name}")


# ADIM 14: HİPERPARAMETRE AYARLAMA (GRID SEARCH)
print("\n--- ADIM 14: EN İYİ MODEL İÇİN HİPERPARAMETRE AYARLAMA ---")

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"En İyi Hiperparametreler: {grid_search.best_params_}")
print(f"GridSearch CV En İyi F1 Skoru: {grid_search.best_score_:.4f}")


# ADIM 15: TEST VERİSİ ÜZERİNDE FİNAL DEĞERLENDİRME
print("\n--- ADIM 15: TEST KÜMESİ ÜZERİNDE FİNAL DEĞERLENDİRME ---")
y_test_pred = best_model.predict(X_test_scaled)

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])

print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Test Precision : {test_prec:.4f}")
print(f"Test Recall    : {test_rec:.4f}")
print(f"Test F1-Score  : {test_f1:.4f}")
print(f"Test ROC-AUC   : {test_auc:.4f}")

print("\nConfusion Matrix (Karmaşıklık Matrisi):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

print("\nAyrıntılı Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred, target_names=['Malignant', 'Benign']))


# ADIM 16 & 17: MODEL DEĞERLENDİRMESİ VE AÇIKLANABİLİRLİK (BONUS)
print("--- ADIM 16 & 17: SONUÇ YORUMU VE MODEL AÇIKLANABİLİRLİĞİ ---")

# Feature Importance Görselleştirme
importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nEn Önemli İlk 5 Öznitelik:")
print(importance_df.head())

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('En Önemli 10 Öznitelik (Random Forest Feature Importance)')
plt.xlabel('Önem Derecesi')
plt.ylabel('Öznitelikler')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance grafiği 'feature_importance.png' olarak kaydedildi.")

"""
YORUM VE SINIRLILIKLAR:
1. Model Performansı: Random Forest modeli Validation aşamasında ve Test kümesinde yüksek F1-Score
   başarısına ulaşmıştır.
2. Önemli Değişkenler: 'worst concave points', 'worst perimeter' ve mühendisliğini yaptığımız
   'worst_to_mean_radius_ratio' öznitelikleri tümörün teşhisinde kritik role sahiptir.
3. Sınırlılıklar: Veri seti gözlem sayısı (~569) medikal dünya için görece küçüktür.
   Gerçek hayat senaryolarında daha geniş ve çeşitli hasta veri gruplarıyla test edilmesi gerekir.
"""
