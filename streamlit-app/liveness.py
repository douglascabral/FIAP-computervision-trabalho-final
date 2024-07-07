import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model

import cv2
import io
import numpy as np
import os

caminho_absoluto = 'notebook/_melhor_modelo_cnn/modelo.keras'
caminho_absoluto = os.path.join(os.getcwd(), caminho_absoluto)
model_cnn = load_model(caminho_absoluto)

"""
# Computer Vision
## Liveness Detection

O detector de Liveness (Vivacidade) tem por objetivo estabelecer um índice que atesta o quão
confiável é a imagem obtida pela câmera.
Imagens estáticas, provindas de fotos manipuladas, são os principais focos de fraude neste tipo de validação.
Um modelo de classificação deve ser capaz de ler uma imagem da webcam, classificá-la como (live ou não) e
exibir sua probabilidade da classe de predição.

"""


uploaded_file = st.file_uploader('Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(imagem, channels="BGR")
    camera = None
else:
    st.markdown('<div style="position: relative; height: 0px; max-height:0px;"><div style="border: solid 4px red; border-radius: 20px; height: 220px; left: 50%; margin-left: -110px; position: absolute; top: 100px; width: 220px; z-index: 1000;"></div></div>', unsafe_allow_html=True)
    camera = st.camera_input("Posione seu rosto na moldura", help="Lembre-se de permitir ao seu navegador o acesso a sua câmera.")

if camera is not None:
    bytes_data = camera.getvalue()
    imagem = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


PRECISAO_DETECCAO_MAOS = 0.3
TAMANHO_MINIMO_DETECCAO_FACE = (30, 30)
TAMANHO_IMAGEM_MOLDURA = (306, 306) #(256 x 256) + 25px para cada lado de margem para pegar o contexto
TAMANHO_IMAGEM_CNN = (64, 64)

if camera or uploaded_file:

    classe = 'Fake'
    porcentagem = 0
    with st.spinner(''):
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            min_detection_confidence=PRECISAO_DETECCAO_MAOS,
            min_tracking_confidence=PRECISAO_DETECCAO_MAOS
        )
        resultados_maos  = hands.process(imagem_rgb)
        maos_detectadas = resultados_maos.multi_hand_landmarks is not None

        if maos_detectadas:
            st.error(f'Mãos detectadas!')
            st.stop()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            imagem_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=TAMANHO_MINIMO_DETECCAO_FACE
        )
        num_faces = len(faces)

        if num_faces == 0:
            st.error(f'Nenhuma face detectada!')
            st.stop()

        if num_faces > 1:
            st.error(f'Mais de uma face detectada!')
            st.stop()

        # tamanho da imagem:
        altura, largura = imagem_gray.shape

        imagem_anot = imagem.copy()

        # se for pela câmera, já sabemos onde está o rosto
        # cortamos a imagem para levar para a CNN já redimensionada
        if camera:
            recortar_x = int(largura/2 - TAMANHO_IMAGEM_MOLDURA[0]/2)
            recortar_y = 50
            imagem_anot = imagem_anot[
                recortar_y:recortar_y+TAMANHO_IMAGEM_MOLDURA[1],
                recortar_x:recortar_x+TAMANHO_IMAGEM_MOLDURA[0]
            ]

        imagem_anot = cv2.resize(imagem_anot, TAMANHO_IMAGEM_CNN)

        img_array = cv2.cvtColor(imagem_anot, cv2.COLOR_RGB2BGR)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalizar a imagem

        prediction = model_cnn.predict(img_array)

        # Porcetagem para cada classe prevista
        porcentagem_real = prediction[0][0] * 100
        porcentagem_fake = (1 - prediction[0][0]) * 100

        classe_prevista = 'Real' if prediction[0] > 0.5 else 'Fake'
        porcentagem = porcentagem_real if classe_prevista == 'Real' else porcentagem_fake

        motivo = f'Imagem {classe_prevista}, probabilidade de {porcentagem}%!'

        if classe_prevista == 'Real':
            st.success(motivo)
        else:
            st.error(motivo)
