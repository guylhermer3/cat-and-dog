import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Configuração da página
st.title("Classificação de Gatos e Cachorros")
st.write("Faça upload de uma imagem para classificar como 'Gato' ou 'Cachorro'.")
colab_link = "[Clique aqui para acessar o notebook de treinamento no Google Colab](https://colab.research.google.com/drive/1nAj95dCoF-V8zxCwb7hX3xgF3yxsqcIM?usp=sharing)"
st.markdown(f"### Treinamento do Modelo: {colab_link}")


# Carregar o modelo salvo
MODEL_PATH = 'modelo_gatos_cachorros.h5'  # Certifique-se de que o arquivo está na mesma pasta
model = load_model(MODEL_PATH)

# Função para pré-processar a imagem
def preprocess_image(image):
    IMG_SIZE = (128, 128)  # Tamanho usado no treinamento
    image = image.resize(IMG_SIZE)  # Redimensionar
    image = np.array(image) / 255.0  # Normalizar para [0, 1]
    image = np.expand_dims(image, axis=0)  # Adicionar dimensão para lote (1, 128, 128, 3)
    return image

# Carregar a imagem do usuário
uploaded_file = st.file_uploader("Faça upload de uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_column_width=True)

    # Pré-processar a imagem
    preprocessed_image = preprocess_image(image)

    # Fazer a predição
    prediction = model.predict(preprocessed_image)[0][0]

    # Determinar a classe
    if prediction >= 0.5:
        st.write("### Resultado: **Cachorro** 🐶")
    else:
        st.write("### Resultado: **Gato** 🐱")

    # Mostrar a probabilidade
    st.write(f"Probabilidade (Cachorro): {prediction}")
