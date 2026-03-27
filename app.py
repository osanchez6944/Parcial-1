import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# ID de tu archivo de Google Drive
FILE_ID = '1UNik99esZBTe3O0ofsnh7BPq2zQaQAFL'
MODEL_PATH = 'modelo_flores_v2.h5'

@st.cache_resource
def load_model_from_drive():
    # Si el modelo no está en el servidor de Streamlit, lo descarga de Drive
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Descargando modelo desde Google Drive... Esto solo ocurre la primera vez.'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸")
st.title("🌸 Clasificador de Flores")
st.write("Sube una foto de una flor (Rosa, Girasol, Tulipán, Margarita o Diente de León) para identificarla.")

try:
    # Carga del modelo
    model = load_model_from_drive()
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', use_column_width=True)
        
        # Preprocesamiento idéntico al entrenamiento
        img = image.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Crear un batch

        # Predicción
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        resultado = class_names[np.argmax(score)]
        confianza = 100 * np.max(score)

        st.success(f"### Predicción: **{resultado.upper()}**")
        st.info(f"### Confianza: **{confianza:.2f}%**")
        
        # Gráfico de probabilidades
        st.write("#### Probabilidades por categoría:")
        st.bar_chart(score.numpy())

except Exception as e:
    st.error(f"Error en la aplicación: {e}")
    st.write("Asegúrate de que el archivo requirements.txt incluya 'gdown'.")
