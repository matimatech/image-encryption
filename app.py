import streamlit as st
from hill import Hill
from image import read_image
from PIL import Image
import imageio.v3 as iio

st.header("Enkripsi dan Dekripsi Citra")

# limited to 200MB image size
uploaded_file = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    real_img, modified_img = read_image(uploaded_file)
    # hill = Hill(img, )
    st.image(real_img, caption="Original Image", clamp=True, channels='RGB')
    print(uploaded_file)
    hill = Hill(modified_img)
    # st.write(filename)

    encrypted_img = hill.encrypt(modified_img.shape[0])
    
    st.write("Encrypted Image")
    st.image(encrypted_img, caption="Encrypted Image")

    decrypted_img = hill.decrypt(encrypted_img)
    st.write("Decrypted Image")
    st.image(decrypted_img, caption="Decrypted Image")



# def onEvent(args):
#     st.write(args)

# bt1 = st.button(
#     "test1",
#     key=None,
#     help="help info",
#     on_click=hill.encrypt(img.shape[0]),
#     kwargs=None,
#     disabled=False,
# )

# # Add a selectbox to the sidebar: