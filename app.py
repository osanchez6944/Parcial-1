import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores por URL", layout="centered")

st.title("🌸 Sistema de Catalogación Botánica")
st.markdown("Pegue el **enlace directo** de la imagen de una flor para identificarla.")

# 1. Cargar el modelo entrenado
@st.cache_resource
def load_my_model():
    # Asegúrate de que el archivo .h5 esté en la misma carpeta que este script
    return tf.keras.models.load_model('modelo_flores_v2.h5')

model = load_my_model()
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# 2. Catálogo de especies
with st.expander("Ver catálogo de especies detectables"):
    st.write(", ".join([name.capitalize() for name in class_names]))

# 3. Módulo de entrada por URL
url = st.text_input("URL de la imagen (JPG, PNG):", placeholder="https://ejemplo.com/foto-flor.jpg")

if url:
    try:
        # Descargar la imagen desde la web
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        
        # Mostrar la imagen cargada
        st.image(image, caption='Imagen desde el enlace', use_container_width=True)
        
        # 4. Preprocesamiento (Ajustado a 180x180 como tu modelo original)
        img = image.convert('RGB') # Asegura que tenga 3 canales
        img = img.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Crear el batch (1, 180, 180, 3)

        # Predicción
        with st.spinner('Analizando imagen...'):
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
        
        # 5. Resaltado del resultado
        resultado = class_names[np.argmax(score)]
        confianza = 100 * np.max(score)
        
        st.success(f"### Predicción: {resultado.upper()} ({confianza:.2f}% de confianza)")

        # 6. Distribución de probabilidades
        st.write("#### Confianza por categoría:")
        for i, name in enumerate(class_names):
            prob = float(score[i])
            col1, col2 = st.columns([1, 4])
            col1.text(name.capitalize())
            col2.progress(prob)

    except Exception as e:
        st.error(f"Error: No se pudo cargar la imagen. Verifique que el link sea directo y público. (Detalle: {e})")

else:
    st.info("💡 Consejo: Copia el 'Vínculo de la imagen' (que termine en .jpg o .png) de Google Imágenes y pégalo arriba.")