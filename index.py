import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import datetime
import altair as alt
import matplotlib.pyplot as plt

# Function to preprocess data and train model
def preprocess_data_and_train_model(df):
    # Convert categorical variables to numerical using dummy encoding or label encoding
    df_encoded = pd.get_dummies(df, columns=['NAMABARANG', 'NAMAOUTLET', 'NAMASALESMAN'])
    
    # Ensure all columns exist in encoded dataset
    current_columns = df_encoded.columns
    for col in ['NAMABARANG_Obat A', 'NAMABARANG_Obat B', 'NAMABARANG_Obat C',
                'NAMAOUTLET_Apotek A', 'NAMAOUTLET_Apotek B', 'NAMAOUTLET_Apotek C',
                'NAMASALESMAN_John Doe', 'NAMASALESMAN_Jane Smith', 'NAMASALESMAN_Mike Johnson']:
        if col not in current_columns:
            df_encoded[col] = 0  # Add missing column with default value
    
    # Convert datetime columns to numerical representation (days since 2023-01-01)
    reference_date = pd.Timestamp('2023-01-01')
    df_encoded['TGL_INVC'] = (pd.to_datetime(df_encoded['TGL_INVC']) - reference_date).dt.days
    df_encoded['EXP_DATE'] = (pd.to_datetime(df_encoded['EXP_DATE']) - reference_date).dt.days
    
    # Split data into X (features) and y (target)
    X = df_encoded.drop(['TRS_TYPE'], axis=1)
    y = df_encoded['TRS_TYPE']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label='SALES')
    precision = precision_score(y_test, y_pred, pos_label='SALES')
    f1 = f1_score(y_test, y_pred, pos_label='SALES')
    
    metrics = {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    }
    
    return model, X_train.columns, metrics  # Return model, columns used for training, and metrics

# Function to predict sales and returns
def predict_sales_and_returns(model, input_data, columns_fit):
    # Ensure input_data has the same columns as during training
    current_columns = input_data.columns
    for col in columns_fit:
        if col not in current_columns:
            input_data[col] = 0  # Add missing column with default value
    
    # Remove any extra columns not seen during training
    input_data = input_data[columns_fit]
    
    # Convert datetime columns to numerical representation (days since 2023-01-01)
    reference_date = pd.Timestamp('2023-01-01')
    input_data['TGL_INVC'] = (pd.to_datetime(input_data['TGL_INVC']) - reference_date).dt.days
    input_data['EXP_DATE'] = (pd.to_datetime(input_data['EXP_DATE']) - reference_date).dt.days
    
    # Perform prediction using the trained model
    predictions = model.predict(input_data)
    
    return predictions

# Function to calculate total sales and returns
def calculate_total_sales_returns(df, nama_salesman=None, nama_barang=None):
    if nama_salesman:
        sales_data = df[(df['NAMASALESMAN'] == nama_salesman) & (df['TRS_TYPE'] == 'SALES')]
        retur_data = df[(df['NAMASALESMAN'] == nama_salesman) & (df['TRS_TYPE'] == 'RETUR')]
    elif nama_barang:
        sales_data = df[(df['NAMABARANG'] == nama_barang) & (df['TRS_TYPE'] == 'SALES')]
        retur_data = df[(df['NAMABARANG'] == nama_barang) & (df['TRS_TYPE'] == 'RETUR')]
    else:
        sales_data = df[df['TRS_TYPE'] == 'SALES']
        retur_data = df[df['TRS_TYPE'] == 'RETUR']
    
    total_sales = sales_data['QTYSALES'].sum() if not sales_data.empty else 0
    total_retur = retur_data['QTYSALES'].sum() if not retur_data.empty else 0
    total_sales_value = sales_data['VALUEHNA'].sum() if not sales_data.empty else 0
    total_retur_value = retur_data['VALUEHNA'].sum() if not retur_data.empty else 0
    
    return total_sales, total_retur, total_sales_value, total_retur_value

# Streamlit application
st.title('Prediksi Penjualan dan Retur di Apotek')

# Load data from Excel
@st.cache_data  # Cache data loading for better performance
def load_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

# Input form
st.sidebar.header('Masukkan Parameter Prediksi')
file_path = st.sidebar.file_uploader('Upload File Excel', type=['xlsx'])
if file_path is not None:
    df = load_data(file_path)
    
    # Display all loaded data
    st.subheader('Data yang Dimuat')
    st.write(df)  # Change here to display all data

    # Preprocess and train model using loaded data
    model, columns_fit, metrics = preprocess_data_and_train_model(df)
    
    # Define tanggal_prediksi here
    tanggal_prediksi = st.sidebar.date_input('Tanggal Prediksi', min_value=datetime.date(2023, 1, 1), max_value=datetime.date(2023, 6, 30), value=datetime.date(2023, 6, 30))
    nama_barang = st.sidebar.selectbox('Nama Barang', [''] + list(df['NAMABARANG'].unique()))
    nama_salesman = st.sidebar.selectbox('Nama Salesman', [''] + list(df['NAMASALESMAN'].unique()))
    
    # Filter data based on input
    filtered_data = df[
        (df['TGL_INVC'] <= pd.to_datetime(tanggal_prediksi)) &
        (df['NAMABARANG'] == nama_barang if nama_barang else True) &
        (df['NAMASALESMAN'] == nama_salesman if nama_salesman else True)
    ]

    if not filtered_data.empty:
        # Calculate total sales and returns
        total_sales, total_retur, total_sales_value, total_retur_value = calculate_total_sales_returns(filtered_data, nama_salesman, nama_barang)

        # Calculate net price after discount (SPEC_DISC)
        if nama_barang and nama_salesman:
            spec_disc = filtered_data[(filtered_data['NAMABARANG'] == nama_barang) & (filtered_data['NAMASALESMAN'] == nama_salesman)]['SPEC_DISC'].mean()
            value_hna = filtered_data[(filtered_data['NAMABARANG'] == nama_barang) & (filtered_data['NAMASALESMAN'] == nama_salesman)]['VALUEHNA'].mean()
            net_price_apotek = value_hna * (1 - spec_disc / 100)
        else:
            net_price_apotek = 0  # Default value if not specified

        # Display total sales and returns with badge style
        st.subheader('Total Penjualan dan Retur')
        if nama_salesman:
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #198754; color: #ffffff; padding: 8px; border-radius: 5px;">Total Sales oleh {nama_salesman}: {total_sales}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #dc3545; color: #ffffff; padding: 8px; border-radius: 5px;">Total Retur oleh {nama_salesman}: {total_retur}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #198754; color: #ffffff; padding: 8px; border-radius: 5px;">Nilai Total Sales oleh {nama_salesman}: Rp. {total_sales_value:,.2f}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #dc3545; color: #ffffff; padding: 8px; border-radius: 5px;">Nilai Total Retur oleh {nama_salesman}: Rp. {total_retur_value:,.2f}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #0d6efd; color: #ffffff; padding: 8px; border-radius: 5px;">Harga Net Apotek untuk Barang {nama_barang}: Rp. {net_price_apotek:,.2f}</span>', unsafe_allow_html=True)
        elif nama_barang:
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #198754; color: #ffffff; padding: 8px; border-radius: 5px;">Total Sales untuk {nama_barang}: {total_sales}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #dc3545; color: #ffffff; padding: 8px; border-radius: 5px;">Total Retur untuk {nama_barang}: {total_retur}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #198754; color: #ffffff; padding: 8px; border-radius: 5px;">Nilai Total Sales untuk {nama_barang}: Rp. {total_sales_value:,.2f}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #dc3545; color: #ffffff; padding: 8px; border-radius: 5px;">Nilai Total Retur untuk {nama_barang}: Rp. {total_retur_value:,.2f}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #0d6efd; color: #ffffff; padding: 8px; border-radius: 5px;">Harga Net Apotek untuk Barang {nama_barang}: Rp. {net_price_apotek:,.2f}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #198754; color: #ffffff; padding: 8px; border-radius: 5px;">Nilai Total Sales: {total_sales}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #dc3545; color: #ffffff; padding: 8px; border-radius: 5px;">Nilai Total Retur: {total_retur}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-size: 18px; font-weight: bold; background-color: #0d6efd; color: #ffffff; padding: 8px; border-radius: 5px;">Harga Net Apotek untuk Barang {nama_barang}: Rp. {net_price_apotek:,.2f}</span>', unsafe_allow_html=True)

        # Display top salesman and top outlet
        if nama_salesman:
            st.subheader(f'Top Barang untuk {nama_salesman}')
            top_barang = filtered_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index().sort_values(by='QTYSALES', ascending=False).head(5)
            top_barang = top_barang.rename(columns={'NAMABARANG': 'Nama Barang', 'QTYSALES': 'Jumlah Produk'})
            st.write(top_barang)

        if nama_barang:
            st.subheader(f'Top Salesman untuk Barang {nama_barang}')
            top_salesman_barang = filtered_data.groupby('NAMASALESMAN')['QTYSALES'].sum().reset_index().sort_values(by='QTYSALES', ascending=False)
            top_salesman_barang = top_salesman_barang.rename(columns={'NAMASALESMAN': 'Nama Salesman', 'QTYSALES': 'Jumlah Produk'})
            st.write(top_salesman_barang)

        st.subheader('Top Outlet untuk Sales dan Retur')
        top_outlet = filtered_data.groupby('NAMAOUTLET')['QTYSALES'].sum().reset_index().sort_values(by='QTYSALES', ascending=False).head(5)
        top_outlet = top_outlet.rename(columns={'NAMAOUTLET': 'Nama Outlet', 'QTYSALES': 'Jumlah Produk'})
        st.write(top_outlet)

        # Display trends
        st.subheader('Grafik Tren Sales dan Retur')
        sales_chart = alt.Chart(filtered_data[filtered_data['TRS_TYPE'] == 'SALES']).mark_line().encode(
            x='TGL_INVC',
            y='QTYSALES',
            color=alt.value('blue')
        ).properties(
            title='Tren Penjualan'
        )

        retur_chart = alt.Chart(filtered_data[filtered_data['TRS_TYPE'] == 'RETUR']).mark_line().encode(
            x='TGL_INVC',
            y='QTYSALES',
            color=alt.value('red')
        ).properties(
            title='Tren Retur'
        )

        # Display Altair charts using st.altair_chart
        st.altair_chart(sales_chart, use_container_width=True)
        st.altair_chart(retur_chart, use_container_width=True)

        # Display metrics as bar chart
        st.subheader('Metrik Model')
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
        metrics_df.reset_index(inplace=True)
        metrics_df = metrics_df.rename(columns={'index': 'Metric'})

        # Plot metrics
        bar_chart = alt.Chart(metrics_df).mark_bar().encode(
            x='Metric',
            y='Score',
            color=alt.Color('Metric', scale=alt.Scale(range=['green', 'blue', 'orange', 'red'])),
        ).properties(
            title='Metrik Model: Akurasi, Recall, Presisi, F1 Score',
            width=alt.Step(80),  # Adjust width between bars
            height=400,          # Set height of the chart
        )

        # Display chart using st.altair_chart
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.write('Tidak ada data yang cocok dengan kriteria yang dimasukkan.')

else:
    st.sidebar.text('Mohon upload file Excel untuk memulai prediksi.')

