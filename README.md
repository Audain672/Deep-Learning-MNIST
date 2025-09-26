# TP1 – Deep Learning : De la conception au déploiement

## Présentation

Ce projet propose un cycle complet pour concevoir, entraîner, suivre, exposer et déployer un modèle de Deep Learning appliqué à la classification d’images manuscrites (MNIST).  
**Encadrants :** Louis Fippo Fitime, Claude Tinku, Kerolle Sonfack  
**Étudiant :** NGEUKEU MELI Audain  
**École :** National Advanced School of Engineering, Université de Yaoundé I

---

## Objectifs pédagogiques

- Compréhension et mise en œuvre d’un réseau dense avec Dropout et softmax, sur MNIST.
- Utilisation de Git/GitHub pour le versionnement et la collaboration.
- Intégration de MLflow pour le suivi des entraînements et modèles.
- Packaging du modèle dans une API Flask.
- Conteneurisation avec Docker.
- Introduction au CI/CD sur cloud.

---

## Structure du dépôt

├── train_model.py # Script principal d’entraînement

├── app.py # API Flask pour servir le modèle

├── requirements.txt # Dépendances Python

├── Dockerfile # Conteneurisation

├── README.md # Ce fichier


---

## Installation

### 1. Cloner le dépôt

git clone https://github.com/Audain672/Deep-Learning-MNIST.git

cd Deep-Learning-MNIST


### 2. Créer et activer un environnement virtuel
python3 -m venv venv

source venv/bin/activate # Linux/Mac

venv\Scripts\activate # Windows

### 3. Installer les dépendances
pip install -r requirements.txt

---

## Utilisation

### 1. Lancer MLflow

Avant l’entraînement, démarrer le serveur MLflow :

mlflow server --host 127.0.0.1 --port 5000

### 2. Entraîner le modèle

python train_model.py

Le script loggue tous les paramètres et métriques dans MLflow et sauvegarde `mnistmodel.h5`.

### 3. Déployer l’API Flask

python app.py

Envoyez une requête POST sur `/predict` avec l’image normalisée (784 valeurs).  
Réponse :
{
"prediction": <classe>,
"probabilities": [ ... ]
}

---

## Docker
Pour lancer toute l’application conteneurisée :

docker build -t deep-mnist-app .

docker run -p 5000:5000 deep-mnist-app

---

## CI/CD

Le projet peut inclure une action GitHub pour :
- Tests unitaires + build Docker à chaque push
- Déploiement cloud (ex : Google Cloud Run)

---

## Monitoring (production)

Indicateurs clés :
- **Performance modèle** (précision, rappel, F1-score)
- **Drift des données** (variabilité des inputs)
- **Santé du service** (latence, erreurs, ressources)

---

## Scripts

- **train_model.py** : entraînement, tracking MLflow, export modèle.
- **app.py** : API Flask, endpoint `/predict`.
- **requirements.txt** : tensorflow, numpy, flask, mlflow.
- **Dockerfile** :

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]

---

## Liens utiles

- [Dépôt GitHub principal](https://github.com/Audain672/Deep-Learning-MNIST)
- [Exercices TP](https://github.com/Audain672)

---

## Licence

Projet académique ENSPY. Toute réutilisation doit citer l’auteur.

---

## Auteur

**NGEUKEU MELI Audain**

---

Pour toute question, ouvrez une Issue GitHub ou contactez l’encadrant.