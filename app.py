from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


def analyze_sentiment(text):
    # Tokenizar el texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Obtener las predicciones del modelo
    with torch.no_grad():
        outputs = model(**inputs)

    # Obtener las probabilidades y etiquetas
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    sentiment_score = probabilities[0, predicted_class].item()

    # Mapear la etiqueta predicha a una etiqueta legible
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    sentiment_label = labels[predicted_class]

    return sentiment_label, sentiment_score


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/example', methods=['GET', 'POST'])
def example():
    user_input = None
    processed_output = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment_label, sentiment_score = analyze_sentiment(user_input)

        # Mapeo de etiquetas de sentimiento
        sentiment_mapping = {
            'very positive': 'Positivo',
            'positive': 'Positivo',
            'neutral': 'Neutral',
            'negative': 'Negativo',
            'very negative': 'Negativo'
        }
        sentiment_result = sentiment_mapping.get(sentiment_label, 'Desconocido')

        processed_output = f"Análisis de sentimiento: {sentiment_result} (Confianza: {sentiment_score:.2f})"

    return render_template('example.html', user_input=user_input, processed_output=processed_output)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
