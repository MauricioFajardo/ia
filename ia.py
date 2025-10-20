import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import zipfile
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
DB_PATH = 'predictions.db'
MODEL_FILENAME_DEFAULT = 'Keras_model.h5'
LABELS_FILENAME_DEFAULT = 'labels.txt'
EXPORTS_DIR = 'exports'
os.makedirs(EXPORTS_DIR, exist_ok=True)

# ---------------------------
# Database
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE,
            name TEXT,
            email TEXT,
            role TEXT,
            threshold REAL DEFAULT 0.5,
            notes TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(timestamp, source, label, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (timestamp, source, label, confidence) VALUES (?, ?, ?, ?)',
              (timestamp, source, label, float(confidence)))
    conn.commit()
    conn.close()

def get_persons():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM persons', conn)
    conn.close()
    return df

def get_person_by_label(label):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name FROM persons WHERE label=?', (label,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else label  # Si no está en DB, usa el label

def upsert_person(label, name, email, role, threshold, notes):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO persons (label, name, email, role, threshold, notes) VALUES (?, ?, ?, ?, ?, ?)',
              (label, name, email, role, threshold, notes))
    conn.commit()
    conn.close()

# ---------------------------
# Model helpers
# ---------------------------
@st.cache_resource
def load_keras_model(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
    try:
        input_shape = model.input_shape[1:3]
        img_size = input_shape[0]
    except:
        img_size = 224
    return model, img_size

def load_labels_from_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    return labels

def preprocess_image_pil(pil_img, img_size):
    pil = pil_img.convert('RGB').resize((img_size, img_size))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(model, img_array):
    preds = model.predict(img_array)
    if preds.ndim == 2:
        preds = preds[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    return class_idx, confidence, preds

# ---------------------------
# App UI
# ---------------------------
init_db()
st.set_page_config(layout='wide', page_title='Reconocimiento de Personas')
st.title('👁️‍🗨️ Reconocimiento de Personas (Teachable Machine)')

# Sidebar
st.sidebar.header('Configuración del modelo')
uploaded_model = st.sidebar.file_uploader('Sube tu modelo Keras (.h5)', type=['h5'])
model_path_input = st.sidebar.text_input('Ruta local del modelo', MODEL_FILENAME_DEFAULT)
uploaded_labels = st.sidebar.file_uploader('Sube labels.txt (opcional)', type=['txt', 'csv'])

model = None
labels = None
IMG_SIZE = 224

try:
    if uploaded_model is not None:
        model_tmp_path = os.path.join(EXPORTS_DIR, uploaded_model.name)
        with open(model_tmp_path, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        model, IMG_SIZE = load_keras_model(model_tmp_path)
    elif os.path.exists(model_path_input):
        model, IMG_SIZE = load_keras_model(model_path_input)
    else:
        st.sidebar.warning('⚠️ No se encontró modelo. Sube o coloca el archivo .h5 en la carpeta.')

    if uploaded_labels is not None:
        labels_tmp_path = os.path.join(EXPORTS_DIR, uploaded_labels.name)
        with open(labels_tmp_path, 'wb') as f:
            f.write(uploaded_labels.getbuffer())
        labels = load_labels_from_file(labels_tmp_path)
    elif os.path.exists(LABELS_FILENAME_DEFAULT):
        labels = load_labels_from_file(LABELS_FILENAME_DEFAULT)
except Exception as e:
    st.sidebar.error(f'Error cargando modelo: {e}')

# Tabs
tab = st.tabs(['🎥 En vivo', '🧍 Administración', '📊 Analítica', '📤 Exportar'])

# ---------------------------
# EN VIVO
# ---------------------------
with tab[0]:
    st.header('🎥 Detección en tiempo real o por imagen')

    mode = st.radio('Fuente', ['Cámara', 'Subir imagen'])
    FRAME_WINDOW = st.image([])

    if mode == 'Cámara':
        run = st.checkbox('Iniciar cámara')
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error('No se pudo acceder a la cámara.')
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error('Error leyendo cámara.')
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    arr = preprocess_image_pil(pil, IMG_SIZE)

                    if model:
                        idx, conf, raw = predict(model, arr)
                        label = labels[idx] if labels and idx < len(labels) else str(idx)
                        name = get_person_by_label(label)
                        timestamp = datetime.utcnow().isoformat()
                        insert_prediction(timestamp, 'camera', label, conf)

                        cv2.putText(rgb, f"{name} ({conf*100:.1f}%)",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    FRAME_WINDOW.image(rgb)
                    if not run:
                        break
            cap.release()

    else:
        uploaded_img = st.file_uploader('Sube una imagen', type=['png', 'jpg', 'jpeg'])
        if uploaded_img is not None and model:
            pil = Image.open(uploaded_img)
            st.image(pil, caption='Imagen cargada', use_column_width=True)
            arr = preprocess_image_pil(pil, IMG_SIZE)
            idx, conf, raw = predict(model, arr)
            label = labels[idx] if labels and idx < len(labels) else str(idx)
            name = get_person_by_label(label)
            timestamp = datetime.utcnow().isoformat()
            insert_prediction(timestamp, 'upload', label, conf)
            st.success(f'Persona reconocida: **{name}** (Confianza: {conf*100:.1f}%)')

# ---------------------------
# ADMINISTRACIÓN
# ---------------------------
with tab[1]:
    st.header('🧍 Administración de personas')
    with st.form('add_person'):
        label = st.text_input('Etiqueta del modelo (label)')
        name = st.text_input('Nombre')
        email = st.text_input('Correo')
        role = st.text_input('Rol')
        threshold = st.number_input('Umbral de confianza', 0.0, 1.0, 0.5)
        notes = st.text_area('Notas')
        submitted = st.form_submit_button('Guardar')
        if submitted and label:
            upsert_person(label, name, email, role, threshold, notes)
            st.success('Persona guardada/actualizada')

    df_persons = get_persons()
    st.dataframe(df_persons)

# ---------------------------
# ANALÍTICA
# ---------------------------
with tab[2]:
    st.header('📊 Analítica de predicciones')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM predictions', conn)
    conn.close()

    if df.empty:
        st.info('No hay predicciones registradas aún.')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.subheader('1️⃣ Distribución por etiqueta')
        fig1, ax1 = plt.subplots()
        df['label'].value_counts().plot(kind='bar', ax=ax1)
        st.pyplot(fig1)

        st.subheader('2️⃣ Histograma de confianza')
        fig2, ax2 = plt.subplots()
        ax2.hist(df['confidence'], bins=20)
        st.pyplot(fig2)

        st.subheader('3️⃣ Fuente de predicción')
        fig3, ax3 = plt.subplots()
        df['source'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax3)
        ax3.set_ylabel('')
        st.pyplot(fig3)

        st.subheader('4️⃣ Evolución temporal')
        df['day'] = df['timestamp'].dt.date
        fig4, ax4 = plt.subplots()
        df.groupby('day').size().plot(ax=ax4)
        st.pyplot(fig4)

        st.subheader('5️⃣ Confianza promedio por etiqueta')
        fig5, ax5 = plt.subplots()
        df.groupby('label')['confidence'].mean().plot(kind='bar', ax=ax5)
        st.pyplot(fig5)

# ---------------------------
# EXPORTAR
# ---------------------------
with tab[3]:
    st.header('📤 Exportar datos')
    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC', conn)
    conn.close()

    st.dataframe(df_all)
    csv = df_all.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV', data=csv, file_name='predicciones.csv', mime='text/csv')
