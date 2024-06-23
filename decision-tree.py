import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import os

# Baca data CSV
data = pd.read_csv("C:/SALES/sbo-sales.csv")

# Konversi 'TGL_INVC' menjadi format tanggal yang dapat diolah
data['TGL_INVC'] = pd.to_datetime(data['TGL_INVC'], format='%d-%m-%Y')

# Ekstraksi komponen tanggal dari kolom 'TGL_INVC'
data['TGL_DAY'] = data['TGL_INVC'].dt.day
data['TGL_MONTH'] = data['TGL_INVC'].dt.month
data['TGL_YEAR'] = data['TGL_INVC'].dt.year

# Hapus kolom 'TGL_INVC' yang tidak diperlukan lagi
data.drop(columns=['TGL_INVC'], inplace=True)

# Split data untuk melatih model Random Forest
X = data[['TGL_DAY', 'TGL_MONTH', 'TGL_YEAR', 'NAMABARANG', 'NAMAOUTLET']]
y = data['TRS_TYPE']

# Buat objek LabelEncoder untuk NAMABARANG dan NAMAOUTLET
label_encoder = LabelEncoder()
X['NAMABARANG'] = label_encoder.fit_transform(X['NAMABARANG'])
X['NAMAOUTLET'] = label_encoder.fit_transform(X['NAMAOUTLET'])

# Pisahkan data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = clf.predict(X_test)

# Mengukur akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Menghitung recall untuk SALES
recall_sales = recall_score(y_test, y_pred, pos_label="SALES", average="macro")

# Menghitung presisi untuk SALES
precision_sales = precision_score(y_test, y_pred, pos_label="SALES", average="macro")

# Menghitung F1 Score secara keseluruhan
f1_score_macro = f1_score(y_test, y_pred, average='macro')

# Mengambil salah satu estimator dari model (misalnya, estimator pertama)
tree_estimator = clf.estimators_[0]

# Export pohon keputusan dalam format DOT
dot_data = export_graphviz(tree_estimator, out_file=None, 
                           feature_names=X.columns.tolist(),  
                           class_names=['SALES', 'RETUR'],  # Ganti dengan kelas yang sesuai
                           filled=True, rounded=True, special_characters=True)

# Membuat objek Graphviz
graph = graphviz.Source(dot_data)

# Menyimpan gambar pohon keputusan dalam format PNG
graph.render("tree_decision")

# Menampilkan gambar pohon keputusan
graph.view("tree_decision")

# Menampilkan metrik evaluasi
print(f'Akurasi: {accuracy:.2f}')
print(f'Recall untuk SALES: {recall_sales:.2f}')
print(f'Presisi untuk SALES: {precision_sales:.2f}')
print(f'F1 Score: {f1_score_macro:.2f}')