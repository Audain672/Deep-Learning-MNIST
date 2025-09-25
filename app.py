import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify

app = Flask(__name__)

# Chargement du modèle Keras sauvegardé
try:
    model = keras.models.load_model('mnist_model.h5')
    print(f"Modèle mnist_model.h5 chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle mnist_model.h5: {e}")
    model = None


@app.route('/')
def home():
    return "API de prédiction MNIST. Utilisez /predict pour faire des prédictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé."}), 500

    data = request.get_json(force=True)
    #verification des donnees
    if 'image' not in data:
        return jsonify({"error": "Clé 'image' manquante dans la requête."}), 400

    try:
        image_data = np.array(data['image'])

        if image_data.shape != (784,):
            return jsonify({"error": f"La forme de l'image est incorrecte. Attend (784,), reçu {image_data.shape}"}), 400

        image_data = image_data.reshape(1, 784)
        image_data = image_data.astype("float32") / 255.0

        # Faire la prédiction
        prediction = model.predict(image_data)

        # Obtenir la classe prédite
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        return jsonify({
            "prediction": int(predicted_class), 
            "probabilities": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)