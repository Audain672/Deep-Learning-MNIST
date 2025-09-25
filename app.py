import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
import json # Pour l'affichage des logs et pour une meilleure manipulation

app = Flask(__name__)

# --- Chargement du modèle Keras ---
# Nous allons charger le modèle que MLflow a enregistré.
# Par défaut, MLflow stocke les runs dans un dossier 'mlruns'.
# Pour ce TP, simplifions en chargeant directement le fichier .h5 si vous ne voulez pas passer par MLflow Model Registry (plus complexe pour un TP introductif)
# Si vous avez un fichier "mnist_model.h5" généré par la sauvegarde précédente, utilisez-le.
# Sinon, vous pouvez le récupérer depuis un run MLflow:
# Trouvez le chemin vers votre modèle dans le dossier mlruns. Ex: './mlruns/0/abcdef1234567890/artifacts/mnist_model_keras/model.h5'
# Pour simplifier et être autonome : je vais considérer qu'on a un `mnist_model.h5` sauvegardé.
# Pour une vraie intégration MLflow, on utiliserait `mlflow.keras.load_model('runs:/<run_id>/mnist_model_keras')`
# Pour le TP, chargeons le fichier H5 créé précédemment.
MODEL_PATH = "mnist_model.h5"
model = None # Initialiser le modèle à None

def load_keras_model():
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"Modèle '{MODEL_PATH}' chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {MODEL_PATH}: {e}")
        # Si le modèle n'est pas trouvé, le conteneur ne démarrera pas correctement.
        # Vous devrez vous assurer que 'mnist_model.h5' est dans le même répertoire que le Dockerfile.
        # Ou utiliser MLflow Model Registry pour charger depuis un registre distant.
        # Pour ce TP, on suppose qu'il est présent localement.

# Charger le modèle au démarrage de l'application Flask
# Cela garantit que le modèle est prêt avant de recevoir des requêtes.
with app.app_context():
    load_keras_model()


@app.route('/')
def home():
    return "API de prédiction MNIST. Utilisez /predict pour faire des prédictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé."}), 500

    data = request.get_json(force=True)

    if 'image' not in data:
        return jsonify({"error": "Clé 'image' manquante dans la requête."}), 400

    try:
        # Les données d'image attendues sont une liste de 784 nombres (pixels aplatis)
        # Convertir la liste Python en tableau NumPy
        input_image = np.array(data['image'], dtype=np.float32)

        # Assurez-vous que l'image est de la bonne forme pour le modèle (1, 784)
        # Le modèle attend un lot d'images, même s'il n'y en a qu'une
        if input_image.shape != (784,):
            return jsonify({"error": f"La forme de l'image est incorrecte. Attend (784,), reçu {input_image.shape}"}), 400

        input_image = input_image.reshape(1, 784)

        # Normalisation (si les données d'entrée ne sont pas déjà normalisées entre 0 et 1)
        # Assurez-vous que le client envoie des pixels entre 0 et 1 ou normalisez ici si le client envoie 0-255
        # Le modèle a été entraîné avec des images normalisées (0-1), donc c'est crucial.
        # Supposons que le client envoie déjà des valeurs entre 0 et 1.
        # Si le client envoie 0-255, il faudrait: input_image = input_image / 255.0

        # Faire la prédiction
        predictions = model.predict(input_image)

        # Obtenir la classe prédite (celle avec la plus haute probabilité)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) # Convertir en float standard pour jsonify

        return jsonify({
            "prediction": int(predicted_class), # Convertir en int standard
            "confidence": confidence,
            "probabilities": predictions[0].tolist() # Convertir en liste Python pour jsonify
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

if __name__ == '__main__':
    # Flask s'exécute sur tous les interfaces par défaut pour le conteneur Docker.
    # debug=True n'est pas recommandé en production.
    app.run(host='0.0.0.0', port=5000, debug=False)