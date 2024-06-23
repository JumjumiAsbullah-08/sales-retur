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

# Menampilkan Product Name yang banyak di SALES jika filter adalah SALES
most_item_name = ""
most_item_qty = 0
if filter_sales_return == "SALES":
    most_item = sales_return_by_item.nlargest(1, 'JUMLAH BARANG')
    most_item_name = most_item['NAMABARANG'].values[0]
    most_item_qty = most_item['JUMLAH BARANG'].values[0]

# Menampilkan Product Name yang banyak di RETUR jika filter adalah RETUR
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
sales_result.rename(columns={'NAMABARANG': 'Product Name', 'QTYSALES': 'Sales Quantity'}, inplace=True)

# Jumlahkan QTYSALES untuk RETUR dengan NAMABARANG yang sama
retur_result = retur_data.groupby("NAMABARANG")["QTYSALES"].sum().reset_index()
retur_result.rename(columns={'NAMABARANG': 'Product Name', 'QTYSALES': 'Return Quantity'}, inplace=True)

# Temukan hasil maksimum pada SALES
max_sales = sales_result[sales_result["Sales Quantity"] == sales_result["Sales Quantity"].max()]

# Temukan hasil minimum pada RETUR (dengan menggunakan tanda minus)
min_retur = retur_result[retur_result["Return Quantity"] == retur_result["Return Quantity"].min()]

# Gabungkan kolom 'TGL_DAY' dan 'TGL_MONTH' pada tabel max_sales
max_sales['Top Sales Date'] = most_common_sales_date.iloc[0] if not most_common_sales_date.empty else 'Tidak ada data'
max_sales['Top Sales Month'] = most_common_sales_month.iloc[0] if not most_common_sales_month.empty else 'Tidak ada data'

# Gabungkan kolom 'TGL_DAY' dan 'TGL_MONTH' pada tabel min_retur
min_retur['Top Return Date'] = most_common_return_date.iloc[0] if not most_common_return_date.empty else 'Tidak ada data'
min_retur['Top Return Month'] = most_common_return_month.iloc[0] if not most_common_return_month.empty else 'Tidak ada data'

# Tampilkan hasil dalam aplikasi Streamlit
if filter_sales_return == "SALES":
    st.title("Analisis Perhitungan Barang Terjual (Sales) dan Barang Pengembalian (Retur) Menggunakan Algoritma Random Forest")
    st.subheader("Highest Sales Result :")
    st.write(max_sales)
    st.subheader("Sales Data and Quantity by Product Category")
    st.write(sales_result)
elif filter_sales_return == "RETUR":
    st.title("Analisis Perhitungan Barang Terjual (Sales) dan Barang Pengembalian (Retur) Menggunakan Algoritma Random Forest")
    st.subheader("Highest Return Result :")
    st.write(min_retur)
    st.subheader("Data and Quantity of Returns (Refunds) by Product Category")
    st.write(retur_result)
# Rename nama kolom data tabel
filtered_data.rename(columns={'TRS_TYPE': 'Type', 'NAMAOUTLET': 'Outlet Name', 'NAMABARANG': 'Product Name',
                             'QTYSALES': 'Total', 'TGL_DAY': 'Date/Day', 
                             'TGL_MONTH': 'Month', 'TGL_YEAR': 'Year'}, inplace=True)

# Tampilkan tabel data
st.subheader("Data Tabel:")
st.write(filtered_data)

# Data yang akan digunakan untuk membuat diagram batang
data_to_plot = filtered_data.groupby('Product Name')['Total'].sum().reset_index()

# Membuat diagram batang menggunakan Plotly Express
fig = px.bar(data_to_plot, x='Product Name', y='Total',
             title=f"Quantity of Items {filter_sales_return}",
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

# Lakukan prediksi pada data uji
y_pred = clf.predict(X_test)

# Mengukur akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Mengonversi akurasi ke persentase
accuracy_percentage = accuracy * 100

# Menampilkan akurasi model
st.subheader("Random Forest Model Accuracy")
st.write(f"Accuracy Results : {accuracy_percentage:.2f} %")

# Menghitung recall untuk SALES
recall_sales = recall_score(y_test, y_pred, pos_label="TRS_TYPE", average="macro")

# Mengonversi recall ke persen
recall_sales_percent = recall_sales * 100

# Menampilkan recall untuk masing-masing SALES dan RETUR dalam bentuk persen
st.subheader("Sales and Return Recall:")
st.write(f"Recall Accuracy : {recall_sales_percent:.2f} %")

# Menghitung presisi untuk SALES
precision_sales = precision_score(y_test, y_pred, average="macro")

# Menampilkan presisi dalam bentuk persen
st.subheader("Sales and Return Presisi:")
st.write(f"Presisi Result: {precision_sales * 100:.2f} %")

# Menghitung F1 Score secara keseluruhan
f1_score_macro = f1_score(y_test, y_pred, average='macro')

# Mengonversi F1 Score ke persen
f1_score_macro_percentage = f1_score_macro * 100

# Menampilkan F1 Score dalam bentuk persen
st.subheader("Sales and Retur F1 Score :")
st.write(f"F1 Score Result: {f1_score_macro_percentage:.2f} %")

# Buat DataFrame untuk metrik-metrik
metrics_df = pd.DataFrame({
    'Metrics': ['Akurasi', 'Recall', 'Presisi', 'F1 Score'],
    'Value': [accuracy_percentage, recall_sales_percent, precision_sales * 100, f1_score_macro_percentage]
})

# Buat diagram garis menggunakan Plotly Express
fig = px.line(metrics_df, x='Metrics', y='Value', title='Model Evaluation Metrics RETURN',
              labels={'Metrics': 'Metrics', 'Value': 'Value (%)'})

# Tambahkan nilai di atas setiap garis
for i, row in metrics_df.iterrows():
    fig.add_annotation(
        x=row['Metrics'],  # Koordinat x berdasarkan metrik
        y=row['Value'],   # Koordinat y berdasarkan nilai
        text=f'{row["Value"]:.2f}%',  # Teks yang akan ditampilkan
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
    'Category': ['SALES', 'RETURN'],
    'Accuracy': [93.91, 90.48]
})

# Buat grafik clustered column dengan Plotly Express
fig = px.bar(data, x='Category', y='Accuracy', title='Accuracy SALES and RETUR',
             labels={'Category': 'Category', 'Accuracy': 'Accuracy (%)'},
             color='Category')

# Menampilkan grafik clustered column
st.plotly_chart(fig)