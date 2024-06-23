import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import export_text
from sklearn.metrics import precision_score, recall_score, mean_absolute_error


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

# Sebelum pemisahan data
print("Jumlah baris data awal:", data.shape)

# Pisahkan data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setelah pemisahan data
print("Jumlah baris data pelatihan (X_train, y_train):", X_train.shape[0])
print("Jumlah baris data uji (X_test, y_test):", X_test.shape[0])

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
st.subheader("Akurasi Model Random Forest")
st.write(f"Akurasi model Random Forest: {accuracy_percentage:.2f} %")

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

# Menghitung jumlah total Data Terjual (TerSales)
total_sales = sales_data['QTYSALES'].sum()

# Menghitung jumlah total Data Pengembalian (TerRetun)
total_retur = retur_data['QTYSALES'].sum()

# Menampilkan jumlah total Data Terjual (TerSales) dan Data Pengembalian (TerRetun)
st.subheader("Total Data Terjual (TerSales) dan Data Pengembalian (TerRetun)")
st.write(f"Total Data Terjual (TerSales): {total_sales}")
st.write(f"Total Data Pengembalian (TerRetun): {total_retur}")