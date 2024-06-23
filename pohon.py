import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import export_text, DecisionTreeClassifier, export_graphviz
from sklearn.metrics import recall_score, precision_score, f1_score
import graphviz
from IPython.display import Image
from PIL import Image
from io import StringIO

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

# Sidebar
st.sidebar.title("Filter")
filter_sales_return = st.sidebar.selectbox("Tampilkan SALES atau RETUR?", ["SALES", "RETUR"])

# Filter data SALES
sales_data = data[data['TRS_TYPE'] == "SALES"]

# Filter data RETUR
retur_data = data[data['TRS_TYPE'] == "RETUR"]

# Filter data berdasarkan pilihan SALES atau RETUR
filtered_data = data[data['TRS_TYPE'] == filter_sales_return]

# Menghitung jumlah QTYSALES atau RETUR untuk setiap barang
if filter_sales_return == "SALES":
    sales_return_by_item = filtered_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index()
    sales_return_by_item.rename(columns={'QTYSALES': 'JUMLAH BARANG'}, inplace=True)
elif filter_sales_return == "RETUR":
    sales_return_by_item = filtered_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index()
    sales_return_by_item.rename(columns={'QTYSALES': 'QTYRETUR'}, inplace=True)

# Menampilkan nama barang yang banyak di SALES jika filter adalah SALES
most_item_name = ""
most_item_qty = 0
if filter_sales_return == "SALES":
    most_item = sales_return_by_item.nlargest(1, 'JUMLAH BARANG')
    most_item_name = most_item['NAMABARANG'].values[0]
    most_item_qty = most_item['JUMLAH BARANG'].values[0]

# Menampilkan nama barang yang banyak di RETUR jika filter adalah RETUR
if filter_sales_return == "RETUR":
    most_item = sales_return_by_item.nlargest(1, 'QTYRETUR')
    most_item_name = most_item['NAMABARANG'].values[0]
    most_item_qty = most_item['QTYRETUR'].values[0]

# Menambahkan kolom "Jumlah QTYSALES" pada Data Tabel SALES
if filter_sales_return == "SALES":
    sales_return_by_item = filtered_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index()
    sales_return_by_item.rename(columns={'QTYSALES': 'Jumlah QTYSALES'}, inplace=True)
elif filter_sales_return == "RETUR":
    # Menambahkan kolom "Jumlah QTYSALES (dengan minus)" pada Data Tabel RETUR
    sales_return_by_item = filtered_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index()
    sales_return_by_item.rename(columns={'QTYSALES': 'QTYRETUR'}, inplace=True)
    sales_return_by_item['QTYRETUR'] = -sales_return_by_item['QTYRETUR']  # Tambahkan tanda minus

# Split data untuk melatih model Random Forest
X = filtered_data[['TGL_DAY', 'TGL_MONTH', 'TGL_YEAR', 'NAMABARANG', 'NAMAOUTLET']]
y = filtered_data['NAMABARANG']

# Buat objek LabelEncoder untuk NAMABARANG dan NAMAOUTLET
label_encoder = LabelEncoder()
X['NAMABARANG'] = label_encoder.fit_transform(X['NAMABARANG'])
X['NAMAOUTLET'] = label_encoder.fit_transform(X['NAMAOUTLET'])

# Tampilkan tanggal dan bulan paling banyak di SALES dan RETUR jika ada data
most_common_sales_date = filtered_data[filtered_data['TRS_TYPE'] == 'SALES']['TGL_DAY'].mode()
most_common_sales_month = filtered_data[filtered_data['TRS_TYPE'] == 'SALES']['TGL_MONTH'].mode()
most_common_return_date = filtered_data[filtered_data['TRS_TYPE'] == 'RETUR']['TGL_DAY'].mode()
most_common_return_month = filtered_data[filtered_data['TRS_TYPE'] == 'RETUR']['TGL_MONTH'].mode()

# Menghitung jumlah QTYSALES pada SALES untuk setiap barang
sales_qty_by_item = sales_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index()
sales_qty_by_item.rename(columns={'QTYSALES': 'Jumlah Penjualan'}, inplace=True)

# Menghitung jumlah QTYSALES pada RETUR untuk setiap barang dengan tanda minus
retur_qty_by_item = retur_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index()
retur_qty_by_item.rename(columns={'QTYSALES': 'Jumlah Penjualan (dengan minus)'}, inplace=True)
retur_qty_by_item['Jumlah Penjualan (dengan minus)'] = -retur_qty_by_item['Jumlah Penjualan (dengan minus)']

# Gabungkan data SALES dan RETUR
combined_data = pd.concat([sales_qty_by_item, retur_qty_by_item], ignore_index=True)

# Jumlahkan QTYSALES untuk SALES dengan NAMABARANG yang sama
sales_result = sales_data.groupby("NAMABARANG")["QTYSALES"].sum().reset_index()
sales_result.rename(columns={'NAMABARANG': 'Nama Barang', 'QTYSALES': 'Jumlah Penjualan'}, inplace=True)

# Jumlahkan QTYSALES untuk RETUR dengan NAMABARANG yang sama
retur_result = retur_data.groupby("NAMABARANG")["QTYSALES"].sum().reset_index()
retur_result.rename(columns={'NAMABARANG': 'Nama Barang', 'QTYSALES': 'Jumlah Pengembalian'}, inplace=True)

# Temukan hasil maksimum pada SALES
max_sales = sales_result[sales_result["Jumlah Penjualan"] == sales_result["Jumlah Penjualan"].max()]

# Temukan hasil minimum pada RETUR (dengan menggunakan tanda minus)
min_retur = retur_result[retur_result["Jumlah Pengembalian"] == retur_result["Jumlah Pengembalian"].min()]

# Gabungkan kolom 'TGL_DAY' dan 'TGL_MONTH' pada tabel max_sales
max_sales['Tanggal Paling Banyak Penjualan'] = most_common_sales_date.iloc[0] if not most_common_sales_date.empty else 'Tidak ada data'
max_sales['Bulan Paling Banyak Penjualan'] = most_common_sales_month.iloc[0] if not most_common_sales_month.empty else 'Tidak ada data'

# Gabungkan kolom 'TGL_DAY' dan 'TGL_MONTH' pada tabel min_retur
min_retur['Tanggal Paling Banyak Pengembalian'] = most_common_return_date.iloc[0] if not most_common_return_date.empty else 'Tidak ada data'
min_retur['Bulan Paling Banyak Pengembalian'] = most_common_return_month.iloc[0] if not most_common_return_month.empty else 'Tidak ada data'

# Tampilkan hasil dalam aplikasi Streamlit
if filter_sales_return == "SALES":
    st.title("Analisis Perhitungan Barang Terjual (Sales) dan Barang Pengembalian (Retur) Menggunakan Algoritma Random Forest")
    st.subheader("Hasil Barang Penjualan (Sales) Paling Tinggi :")
    st.write(max_sales)
    st.subheader("Data dan Jumlah Penjualan (Sales) Berdasarkan Kategori Barang:")
    st.write(sales_result)
elif filter_sales_return == "RETUR":
    st.title("Analisis Perhitungan Barang Terjual (Sales) dan Barang Pengembalian (Retur) Menggunakan Algoritma Random Forest")
    st.subheader("Hasil Barang Pengembalian (Retur) Paling Tinggi:")
    st.write(min_retur)
    st.subheader("Data dan Jumlah Pengembalian (Retur) Berdasarkan Kategori Barang:")
    st.write(retur_result)
# Rename nama kolom data tabel
filtered_data.rename(columns={'TRS_TYPE': 'Type', 'NAMAOUTLET': 'Nama Outlet', 'NAMABARANG': 'Nama Barang',
                             'QTYSALES': 'Jumlah', 'TGL_DAY': 'Tanggal/Hari', 
                             'TGL_MONTH': 'Bulan', 'TGL_YEAR': 'Tahun'}, inplace=True)

# Tampilkan tabel data
st.subheader("Data Tabel:")
st.write(filtered_data)

# Data yang akan digunakan untuk membuat diagram batang
data_to_plot = filtered_data.groupby('Nama Barang')['Jumlah'].sum().reset_index()

# Membuat diagram batang menggunakan Plotly Express
fig = px.bar(data_to_plot, x='Nama Barang', y='Jumlah',
             title=f"Jumlah Barang {filter_sales_return}",
             labels={'Jumlah': 'Jumlah Penjualan' if filter_sales_return == 'SALES' else 'Jumlah Pengembalian'})

# Jika filter adalah RETUR, tambahkan minus pada label sumbu Y
if filter_sales_return == 'RETUR':
    fig.update_traces(marker_color='red', selector=dict(type='bar'), texttemplate='%{y:.0f}')
    
st.plotly_chart(fig)

# Pisahkan data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Mengambil salah satu estimator (pohon) dari model
tree_estimator = clf.estimators_[0]

# Ekspor pohon ke format dot
dot_data = export_graphviz(tree_estimator, 
                           out_file=None, 
                           feature_names=X.columns.tolist(),
                           class_names=['SALES', 'RETUR'],
                           filled=True, 
                           rounded=True)

# Membuat grafik pohon dengan pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.set_size('"15,15!"')

# Menyimpan grafik ke file gambar
img = Image(graph.create_png())

# Menampilkan grafik pohon keputusan dalam Streamlit
st.title("Pohon Keputusan dalam Model Random Forest")
st.image(img)

# Lakukan prediksi pada data uji
y_pred = clf.predict(X_test)

# Mengukur akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Mengonversi akurasi ke persentase
accuracy_percentage = accuracy * 100

# Menampilkan akurasi model
st.subheader("Akurasi Model Random Forest")
st.write(f"Hasil Akurasi : {accuracy_percentage:.2f} %")

# Menghitung recall untuk SALES
recall_sales = recall_score(y_test, y_pred, pos_label="TRS_TYPE", average="macro")

# Mengonversi recall ke persen
recall_sales_percent = recall_sales * 100

# Menampilkan recall untuk masing-masing SALES dan RETUR dalam bentuk persen
st.subheader("Recall untuk Sales dan Retur:")
st.write(f"Hasil Recall : {recall_sales_percent:.2f} %")

# Menghitung presisi untuk SALES
precision_sales = precision_score(y_test, y_pred, average="macro")

# Menampilkan presisi dalam bentuk persen
st.subheader("Presisi untuk Sales dan Retur:")
st.write(f"Hasil Presisi: {precision_sales * 100:.2f} %")

# Menghitung F1 Score secara keseluruhan
f1_score_macro = f1_score(y_test, y_pred, average='macro')

# Mengonversi F1 Score ke persen
f1_score_macro_percentage = f1_score_macro * 100

# Menampilkan F1 Score dalam bentuk persen
st.subheader("F1 Score untuk Sales dan Retur:")
st.write(f"Hasil F1 Score: {f1_score_macro_percentage:.2f} %")

# Buat DataFrame untuk metrik-metrik
metrics_df = pd.DataFrame({
    'Metrik': ['Akurasi', 'Recall', 'Presisi', 'F1 Score'],
    'Nilai': [accuracy_percentage, recall_sales_percent, precision_sales * 100, f1_score_macro_percentage]
})

# Buat diagram garis menggunakan Plotly Express
fig = px.line(metrics_df, x='Metrik', y='Nilai', title='Metrik Evaluasi Model',
              labels={'Metrik': 'Metrik', 'Nilai': 'Nilai (%)'})

# Tambahkan nilai di atas setiap garis
for i, row in metrics_df.iterrows():
    fig.add_annotation(
        x=row['Metrik'],  # Koordinat x berdasarkan metrik
        y=row['Nilai'],   # Koordinat y berdasarkan nilai
        text=f'{row["Nilai"]:.2f}%',  # Teks yang akan ditampilkan
        font=dict(size=10),  # Ukuran font
        yshift=10        # Geser posisi vertikal teks
    )

# Menampilkan diagram garis
st.plotly_chart(fig)

# Mengambil semua estimator (pohon) dari model
estimators = clf.estimators_

# Menampilkan pohon-pohon keputusan dalam bentuk teks
# st.title("Pohon Keputusan dalam Model Random Forest")
# for i, estimator in enumerate(estimators):
#     st.subheader(f"Pohon Keputusan {i + 1}")
#     tree_rules = export_text(estimator, feature_names=X.columns.tolist())
#     st.text(tree_rules)

data = pd.DataFrame({
    'Kategori': ['SALES', 'RETUR'],
    'Akurasi': [93.91, 90.48]
})

# Buat grafik clustered column dengan Plotly Express
fig = px.bar(data, x='Kategori', y='Akurasi', title='Akurasi SALES dan RETUR',
             labels={'Kategori': 'Kategori', 'Akurasi': 'Akurasi (%)'},
             color='Kategori')

# Menampilkan grafik clustered column
st.plotly_chart(fig)
