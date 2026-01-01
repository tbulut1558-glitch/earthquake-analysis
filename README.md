# Earthquake Prediction Project

Ce projet vise à explorer l’utilisation de techniques de **machine learning** pour analyser des données sismiques historiques et produire des estimations concernant la **magnitude** et la **profondeur** des séismes en fonction du temps et de la localisation géographique.

L’objectif principal est pédagogique : comprendre le cycle complet d’un projet de data science, depuis la préparation des données jusqu’à l’entraînement d’un modèle et son utilisation pour effectuer des prédictions.


## Structure du projet

`
├── data/
│ └── database.csv
│
├── models/
│ ├── earthquake_model.h5
│ └── scaler.pkl
│
├── outputs/
│ ├── eda_plots.png
│ ├── training_history.png
│ └── performance_comparison.png
│
├── scripts/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ └── predict.py
│
├── README.md
└── requirements.txt
`

## Préparation et analyse des données

Le script `data_preprocessing.py` est responsable de :

- Charger le jeu de données sismiques
- Sélectionner les variables pertinentes (date, heure, latitude, longitude, profondeur, magnitude)
- Convertir la date et l’heure en **timestamp Unix**
- Nettoyer les données invalides
- Générer des graphiques exploratoires (EDA)
- Diviser les données en ensembles d’entraînement et de test
- Normaliser les variables d’entrée à l’aide de `StandardScaler`

Les graphiques générés permettent de visualiser :
- La distribution des magnitudes
- La relation entre la magnitude et la profondeur


## Entraînement du modèle

Le fichier `model_training.py` permet d’entraîner un réseau de neurones artificiels simple à l’aide de **TensorFlow / Keras**.

Caractéristiques principales :
- Modèle `Sequential`
- Couches entièrement connectées (Dense)
- Fonction d’activation ReLU
- Fonction de perte : Mean Squared Error (MSE)
- Métrique : Mean Absolute Error (MAE)

À la fin de l’entraînement :
- Les courbes de perte sont sauvegardées
- Une comparaison entre valeurs réelles et prédites est générée
- Le modèle entraîné est sauvegardé dans le dossier `models/`



## 🔮 Prédiction

Le script `predict.py` permet d’utiliser le modèle entraîné pour effectuer des prédictions à partir de nouvelles entrées utilisateur.

L’utilisateur fournit :
- Latitude
- Longitude
- Date (format YYYY-MM-DD)

Le programme retourne :
- Une estimation de la magnitude
- Une estimation de la profondeur du séisme



## Remarques importantes

Ce projet ne constitue **pas un système réel de prévision des séismes**.  
Les résultats produits sont uniquement des **estimations expérimentales** basées sur des données historiques et un modèle simplifié.

Les séismes étant des phénomènes extrêmement complexes, ce projet doit être considéré comme une **expérience académique** et non comme un outil de prédiction fiable.



## Dépendances principales

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- joblib

