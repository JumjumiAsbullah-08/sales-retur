import pandas as pd
import streamlit as st

# Baca file CSV
data = pd.read_csv("C:/buk ase/coba-lagi.csv")

# Buat DataFrame kosong untuk SALES dan RETUR
sales_data = data[data["TRS_TYPE"] == "SALES"]
retur_data = data[data["TRS_TYPE"] == "RETUR"]

# Jumlahkan QTYSALES untuk SALES dengan NAMABARANG yang sama
sales_result = sales_data.groupby("NAMABARANG")["QTYSALES"].sum().reset_index()

# Jumlahkan QTYSALES untuk RETUR dengan NAMABARANG yang sama
retur_result = retur_data.groupby("NAMABARANG")["QTYSALES"].sum().reset_index()

# Temukan hasil maksimum pada SALES
max_sales = sales_result[sales_result["QTYSALES"] == sales_result["QTYSALES"].max()]

# Temukan hasil minimum pada RETUR (dengan menggunakan tanda minus)
min_retur = retur_result[retur_result["QTYSALES"] == retur_result["QTYSALES"].min()]

# Tampilkan hasil dalam aplikasi Streamlit
st.title("Perhitungan SALES dan RETUR")
st.subheader("SALES:")
st.write(sales_result)
st.subheader("RETUR:")
st.write(retur_result)

# Tampilkan hasil maksimum pada SALES
st.subheader("Hasil SALES Paling Tinggi:")
st.write(max_sales)

# Tampilkan hasil minimum pada RETUR (dengan menggunakan tanda minus)
st.subheader("Hasil RETUR Paling Rendah (dengan tanda minus):")
st.write(min_retur)