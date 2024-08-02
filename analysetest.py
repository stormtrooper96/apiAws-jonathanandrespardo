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
