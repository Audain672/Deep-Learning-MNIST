# TP1:
## Partie 1: Fondations du Deep Learning 
### 1.1 concepts Theoriques
1. Rappel des modeles lineaires et de l'optimisation stochastique.

Différences et Préférence pour la SGD en Deep Learning:
- Taille de l'échantillon pour le gradient :
    - BGD : Utilise l'intégralité du jeu de données.
    - SGD : Utilise un seul exemple.
- Fréquence des mises à jour :
    - BGD : Une seule mise à jour par époque (passage complet sur le jeu de données).
    - SGD : Une mise à jour par exemple d'entraînement, donc m mises à jour par époque si le jeu de données contient m exemples.
- Stabilité du gradient :
    - BGD : Gradient stable et précis.
    - SGD : Gradient bruyant et moins précis.
- Chemin de convergence :
    - BGD : Lisse, direct.
    - SGD : Ondulant, bruyant.
- Exigences en mémoire :
    - BGD : Élevées.
    - SGD : Faibles.
Décrivez les rôles des
couches d’entrée, cachées et de sortie. Expliquez le processus de rétropropagation du
gradient (backpropagation) en termes simples.
Un réseau de neurones typique est structuré en plusieurs couches, chacune ayant un rôle spécifique :
- Couche d'Entrée (Input Layer) :
    - Rôle : Cette couche est le point d'entrée des données dans le réseau. Elle reçoit les caractéristiques (features) de l'entrée brute.
    - Fonctionnement : Elle ne réalise aucun calcul sur les données ; elle se contente de les distribuer aux neurones de la première couche cachée. Le nombre de neurones dans cette couche correspond généralement au nombre de caractéristiques (dimensions) des données d'entrée.
    - Exemple : Pour une image de 28x28 pixels, si elle est "aplatie" en un vecteur, la couche d'entrée aurait 784 neurones. Pour des données tabulaires avec 10 caractéristiques, elle aurait 10 neurones.
- Couches Cachées (Hidden Layers) :
    - Rôle : Ce sont les couches intermédiaires entre la couche d'entrée et la couche de sortie. Elles sont "cachées" car leurs activations ne sont pas directement visibles comme entrée ou sortie du réseau. Leur rôle principal est d'extraire des caractéristiques de plus en plus complexes et abstraites des données d'entrée.
    - Fonctionnement : Chaque neurone d'une couche cachée reçoit les sorties des neurones de la couche précédente, calcule une somme pondérée de ces entrées (chaque connexion ayant un poids associé), ajoute un biais, puis applique une fonction d'activation non linéaire (comme ReLU, Sigmoïde, Tanh) à ce résultat. C'est cette non-linéarité qui permet au réseau d'apprendre des relations complexes et non linéaires dans les données.
    - Profondeur : Un réseau avec plusieurs couches cachées est appelé un "réseau de neurones profond" (d'où le terme "Deep Learning"). Plus il y a de couches, plus le réseau peut apprendre des représentations hiérarchiques et sophistiquées.
- Couche de Sortie (Output Layer) :
    - Rôle : Cette couche produit le résultat final du réseau. Elle est chargée de fournir la prédiction ou la classification en fonction de la tâche.
    - Fonctionnement : Les neurones de cette couche reçoivent les activations de la dernière couche cachée. La fonction d'activation utilisée dans cette couche dépend directement du type de problème : Régression(Généralement une activation linéaire (ou pas de fonction d'activation) si la sortie est une valeur continue), Classification binaire(Fonction sigmoïde (pour produire une probabilité entre 0 et 1)), Classification multi-classes(Fonction softmax (pour produire une distribution de probabilité sur plusieurs classes, la somme des sorties étant égale à 1)).
    - Nombre de neurones : Le nombre de neurones dans la couche de sortie correspond au nombre de valeurs que le réseau doit prédire (par exemple, 1 pour la régression ou la classification binaire, ou le nombre de classes pour la classification multi-classes).
--- 
La rétropropagation est l’algorithme clé qui permet aux réseaux de neurones d’apprendre. Elle calcule efficacement le gradient de la fonction de coût par rapport à chaque poids et biais. Ces gradients servent ensuite à un optimiseur (comme la descente de gradient) pour ajuster les poids et réduire l’erreur.

Processus simplifié :

- Propagation avant (Forward Pass)
    - Les données entrent dans le réseau, passent de couche en couche jusqu’à la sortie.
    - Chaque neurone calcule une somme pondérée puis applique une fonction d’activation.
    - La sortie finale est une prédiction.
- Calcul de l’erreur
    - La prédiction est comparée à la vérité terrain.
    - Une fonction de coût mesure l’écart : plus il est grand, plus le coût est élevé.
- Rétropropagation (Backward Pass)
    - L’erreur est propagée en sens inverse, de la sortie vers l’entrée.
    - À chaque couche, on calcule l’impact de chaque poids et biais sur l’erreur (le gradient).
    - Cela utilise la règle de la chaîne pour évaluer comment une petite variation de chaque paramètre affecte l’erreur finale.
- Mise à jour des poids
    - L’optimiseur (SGD, Adam, etc.) ajuste les poids selon les gradients pour minimiser la fonction de coût.
    - Le cycle complet (avant, erreur, arrière, mise à jour) est répété des milliers ou millions de fois jusqu’à ce que le réseau apprenne à prédire correctement.

### 1.2 EXERCICE 1: construction d'un reseau de neurones avec Keras
Instructions:
1. Créez un nouveau répertoire de projet et un environnement virtuel Python.(commande pour creation d'un venv en python: python3 -m venv nom_de_votre_environnement, commande pour activer celui-ci: .venv\Scripts\Activate.ps1)
2. Installez les bibliothèques nécessaires : tensorflow et numpy.
3. Écrivez le code Python suivant dans un ﬁchier nommé train_model.py.

---
#### Questions:
1. Expliquez l’utilité des couches Dense et Dropout. Pourquoi la fonction
d’activation softmax est-elle utilisée dans la couche de sortie pour ce problème de
classiﬁcation?
    - Couches Dense
    Les couches Dense (ou fully connected) sont essentielles pour apprendre des représentations complexes.
        - Chaque neurone reçoit toutes les entrées de la couche précédente, applique une somme pondérée, un biais, puis une fonction d’activation.
        - En empilant ces couches, le réseau apprend des représentations de plus en plus abstraites : d’abord des caractéristiques simples (bords, coins), puis des motifs complexes.
        - Avec au moins une couche Dense et une activation non linéaire, un réseau peut théoriquement approximer n’importe quelle fonction continue (approximateur universel).
        - Elles transforment l’espace de caractéristiques d’entrée en un espace plus adapté à la tâche.
    - Couche Dropout: Le Dropout est une régularisation pour éviter le surapprentissage.
        - Pendant l’entraînement, il désactive aléatoirement un pourcentage de neurones (ex. 20% avec Dropout(0.2)), forçant le réseau à être plus robuste et à apprendre des caractéristiques indépendantes.
        - Cela agit comme si on entraînait plusieurs sous-réseaux, ce qui rend le modèle final plus généralisable.
        - En phase de test, Dropout est désactivé et Keras ajuste automatiquement les poids restants.
    - Fonction softmax en sortie. Dans MNIST (10 classes), softmax est idéale car :
        - Elle transforme les scores (logits) en probabilités dont la somme vaut 1.
        - Chaque sortie représente la probabilité d’appartenir à une classe.
        - Elle facilite l’interprétation (classe = probabilité max).
        - Elle est parfaitement compatible avec la perte categorical_crossentropy, qui maximise la proba de la bonne classe.
---
2. L’optimiseur adam est utilisé. Faites une recherche et expliquez briève-ment en quoi il s’agit d’une amélioration par rapport à la SGD simple.
- Optimiseur Adam vs SGD
    - SGD simple : utilise un seul taux d’apprentissage fixe pour tous les paramètres, ce qui peut ralentir la convergence et provoquer des oscillations.
    - Adam : adapte automatiquement le taux d’apprentissage de chaque paramètre et combine deux idées :
        - Momentum (moyenne des gradients passés) → accélère dans la bonne direction et réduit les oscillations.
        - Adaptation (moyenne des gradients au carré) → ajuste le pas selon la fréquence et la magnitude des gradients.
- Avantages d’Adam :
    - Converge plus vite et plus efficacement dans des réseaux profonds.
    - Pas besoin d’un réglage fin du taux d’apprentissage, ses valeurs par défaut marchent souvent très bien.
    - Plus stable et robuste que la SGD simple.

    En résumé : Adam = SGD + taux d’apprentissage adaptatif + momentum → meilleures performances et plus grande facilité d’utilisation.

---
3. Comment les concepts de "vectorisation" et de "calculs par lots" sont-ils
appliqués dans le code ci-dessus ?
- Vectorisation et calculs par lots dans train_model.py
    - Vectorisation
        - NumPy et TensorFlow/Keras exécutent toutes les opérations (normalisation, reshape, multiplications matricielles, activations) de manière vectorisée, donc sans boucles élément par élément.
        - Exemple : x_train.astype("float32") / 255.0 normalise tout le tableau d’un coup.
    - Calculs par lots (Batch Processing)
        - Avec batch_size=128, le réseau traite 128 exemples en parallèle : propagation avant, calcul de la perte, rétropropagation, puis mise à jour unique des poids.
        - Cela exploite l’efficacité GPU et rend les gradients plus stables que la SGD pure, tout en convergeant plus vite que la BGD complète.

    En résumé :
    La vectorisation accélère les calculs en traitant des tableaux entiers, et l’apprentissage par lots combine cette efficacité avec plus de stabilité et de rapidité d’entraînement.
---

## Partie 2: Ingenierie du Deep Learning
voire github pour les exercices 2, 3 et 4
### Exercice 2 : Versionnement et collaboration avec Git, GitHub et Gitlab
### Exercice 3 : Suivi des expérimentations avec MLﬂow
### Exercice 4 : Conteneurisation avec Docker et création d’une API

### Exercice 5 : Déploiement et CI/CD
### Question:
1. Expliquez comment un pipeline de CI/CD (par exemple, avec GitHub Actions) pourrait automatiser la construction et le déploiement de votre image Docker sur un service comme Google Cloud Run ou Amazon Elastic Container Service
- Un pipeline CI/CD MLOps automatise tout le cycle de vie d’un modèle ML : code → entraînement → validation → déploiement → surveillance.
    - Étapes clés :
        - CI : déclenchement (git push), tests unitaires, analyse qualité, build environnement (Docker).
        - Entraînement/validation : prétraitement, entraînement auto, suivi (ex. MLflow), comparaison au modèle prod, détection de drift.
        - Versioning : enregistrement dans un registry avec métadonnées.
        - CD : build API Docker, déploiement staging puis prod (K8s, A/B, canary).
        - Monitoring : suivi en production.
    - Avantages :
        - Reproductibilité (mêmes versions code/données).
        - Déploiement rapide, fiable.
        - Qualité et robustesse via tests auto.
        - Collaboration facilitée DS/ML/DevOps.
        - Traçabilité et gestion de versions.
        - Détection précoce du data/model drift.
        - Gains d’efficacité par automatisation. 
---
2. Une fois le modèle déployé, quels sont les indicateurs clés que vous mettriez en place pour le monitoring et le débogage en production ? Citez au moins trois types d’indicateurs.
- Trois types d’indicateurs clés à mettre en place pour le monitoring et le débogage en production :

    - Performance du modèle : précision, rappel, F1-score, RMSE, confiance des prédictions.

    - Dérive des données / du modèle : changement de distribution des données d’entrée, dégradation des performances dans le temps.

    - Santé opérationnelle : latence, taux d’erreur, utilisation des ressources, disponibilité.
 
