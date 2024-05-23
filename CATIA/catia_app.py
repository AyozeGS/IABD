import gradio as gr
import numpy as np
import re
import json
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el tokenizer
with open('models/tokenizer.json', 'r') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Cargar el LabelEncoder
with open('models/label_encoder.json', 'r') as f:
    label_encoder_data = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_data['classes']

# Cargar el modelo
model = load_model('models/model_mix_dataset_expanded_bidir.h5')

# Descargas necesarias
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw')

def clear_text(text):

    # Convertir todo a minúsculas
    text = text.lower()

    # Eliminamos signos de puntuación y números
    punctuation = string.punctuation + "¡¿«»©"
    text = text.translate(str.maketrans(punctuation, " " * len(punctuation)))
    
    # Eliminar palabras que contienen caracteres diferentes al español
    pattern = re.compile(r'^[a-záéíóúüñ]+$')
    text = ' '.join([word for word in text.split() if pattern.match(word)])

    # Eliminar stopwords
    stop_words = set(stopwords.words(['spanish', 'english']))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Obtener lemas de las palabras
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

def classify_text(text):
    # Tokenizar el texto
    input_ids = clear_text(text)
    input_ids = tokenizer.texts_to_sequences([input_ids])
    input_ids = pad_sequences(input_ids, maxlen=500)

    # Realizar la clasificación
    output = model.predict(input_ids)[0]
    predicted_class = output.argmax()

    # Mapear la clase predicha a una etiqueta
    labels = label_encoder.classes_
    predicted_label = labels[predicted_class]

    return predicted_label

# Crear la interfaz de Gradio
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=5, placeholder="Escribe tu texto aquí...", label="Texto de entrada"),
    outputs=gr.Textbox(label="Resultado"),
    title="CATIA",
    description="Clasificación de texto utilizando un modelo de aprendizaje profundo.",
    css="""
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }

    .gradio-container {
        text-align: center;
        background-color: #fff;
        padding: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        max-width: 600px;
        width: 100%;
        margin: 20px;
    }

    h1 {
        font-size: 2em;
        margin-bottom: 20px;
    }

    input[type="text"], textarea {
        width: 100%;
        padding: 10px;
        margin-bottom: 20px;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
    }

    #component-14 {
        display: none !important;
    }
    """
)

# Lanzar la aplicación
demo.launch()
# demo.launch(share=True)