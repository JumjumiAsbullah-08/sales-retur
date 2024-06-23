import streamlit as st
from datetime import datetime

# Title of the app
st.title("Formulir Pendaftaran")

# Introduction text
st.write("Silakan isi formulir di bawah ini:")

# Form
with st.form("my_form"):
    # Text input
    name = st.text_input("Nama Lengkap")
    
    # Number input
    age = st.number_input("Usia", min_value=1, max_value=100)
    
    # Date input
    birth_date = st.date_input("Tanggal Lahir", min_value=datetime(1900, 1, 1), max_value=datetime.now())
    
    # Select box
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan", "Lainnya"])
    
    # Multi select
    hobbies = st.multiselect("Hobi", ["Olahraga", "Membaca", "Menonton Film", "Memasak", "Berkebun", "Lainnya"])
    
    # Slider
    satisfaction = st.slider("Seberapa puas Anda dengan layanan kami?", 0, 100, 50)
    
    # Text area
    feedback = st.text_area("Masukan atau Saran")

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Terima kasih telah mengisi formulir berikut:")
        st.write(f"Nama Lengkap: {name}")
        st.write(f"Usia: {age}")
        st.write(f"Tanggal Lahir: {birth_date}")
        st.write(f"Jenis Kelamin: {gender}")
        st.write(f"Hobi: {', '.join(hobbies)}")
        st.write(f"Tingkat Kepuasan: {satisfaction}")
        st.write(f"Masukan atau Saran: {feedback}")

# Footer
st.write("Dibuat dengan Streamlit")
