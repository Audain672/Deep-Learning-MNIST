# Utilise une image de base Python officielle
FROM python:3.9-slim-buster

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie le fichier requirements.txt et installe les dépendances
# Ceci est fait en premier pour tirer parti du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie l'application Flask et le modèle entraîné
# Assurez-vous que mnist_model.h5 est dans le même dossier que le Dockerfile et app.py
COPY app.py .
COPY mnist_model.h5 .

# Expose le port sur lequel l'API va écouter
EXPOSE 5000

# Commande pour démarrer l'application Flask
# Utilise Gunicorn ou uWSGI pour la production, mais pour ce TP Flask suffit.
# Pour un environnement de production, utiliser `gunicorn --bind 0.0.0.0:5000 app:app`
CMD ["python", "app.py"]