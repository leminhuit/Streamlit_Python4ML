import streamlit as st
import cv2
import numpy as np
import PIL

st.markdown(
"""
# Đây là bài tutorial
## 1. Giới thiệu streamlit
## 2. Các thành phần cơ bản của giao diện
"""
)

a_value = st.text_input("Nhập a:")
b_value = st.text_input("Nhập b:")


# operator = st.selectbox("Chọn phép toán",
#     ['Cộng','Trừ','Nhân','Chia'])

operator = st.radio("Chọn phép toán",
    ['Cộng','Trừ','Nhân','Chia'])

button = st.button('Tính')
if button:
    if operator == 'Cộng':
        st.text_input("Kết quả:", float(a_value) + float(b_value))
    elif operator == 'Trừ':
        st.text_input("Kết quả:", float(a_value) - float(b_value))
    elif operator == 'Nhân':
        st.text_input("Kết quả:", float(a_value) * float(b_value))
    elif operator == 'Chia':
        st.text_input("Kết quả:", float(a_value) / float(b_value))

# Adding image into your website
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.header("A cat")
#     st.image("https://static.streamlit.io/examples/cat.jpg")

# with col2:
#     st.header("A dog")
#     st.image("https://static.streamlit.io/examples/dog.jpg")

# with col3:
#     st.header("An owl")
#     st.image("https://static.streamlit.io/examples/owl.jpg")

# Adding tabs instead of column
cat_tab, dog_tab, owl_tab = st.tabs(["Cat", "Dog", "Owl"])

with cat_tab:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with dog_tab:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with owl_tab:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")

# Upload your own file and write it down to local disk
uploaded_file = st.file_uploader("Chọn ảnh")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img_path = 'data/' + uploaded_file.name
    with open('data/' + uploaded_file.name, 'wb') as f:
        f.write(bytes_data)
    img = cv2.imread(img_path, 0)
    # Filter ảnh
    filter = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])/9.0
    
    result = cv2.filter2D(img, -1, filter);
    st.image(result);