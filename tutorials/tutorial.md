# Tutoriel : Lancer une analyse EyeTracker

Ce guide vous explique comment utiliser le script pour analyser vos premières données.

## Pré-requis
Avant de commencer, assurez-vous d'avoir installé les dépendances :
```bash
poetry install
```

## 1. Gestion des données
Maintenant que les dépendances sont installées, il faut respecter l'architecure de dossier suivante :
```text
Project-EyeTracker/
├── Codes/
│   ├── Code_V4.0.py       # Le script principal
│   └── ...				   # Reste des codes
├── Datas/
│   └── ...				   # Mets ici tes fichiers à analyser
├── docs/
│   ├── tutorials/
│   │   └── tutorial.md    # Le tutoriel que tu es en train de lire
│   └── ...                # Les fichiers de documentations
├── Outputs/
│   ├── Graphs/
│   │   └── ...			   # Ici seront enregistrés tous les graphs générés
│   └── ... 			   # Ici seront enregistrés les différents CSV et les timelines
├── LICENSE
├── README.md
└── ...					   # Reste des fichiers de gestion
```
Si jamais tu n'as pas ces dossier, créé les avant la suite (normalement tous est là si tu as cloner le dépôt GitHub)

Place ensuite dans le dossier `Datas/` les vidéos MP4 et les fichiers CSV associés aux données que tu souhaites analyser.

## 2. Lancement de l'analyse
Maintenant que l'architecture de dossier est claire, tu peux executer le programme.
Pour cela va dans le dossier `Codes/` et execute le code `Code_V4_0.py` dans ton terminal :
```bash
poetry run python Codes/Code_V4_0.py
```
Adapte juste `python` avec `python3` en fonction de ton environnement.

Maintenant remplit les différents champ avec le chemin des fichiers correspondants.
Plus qu'à attendre que le programme tourne (il te dit lorqu'un fichier est enregistré).

## 3. Visualisation des résultats
Si tu es là c'est que le code à compiler. Youpi !!

Tu peux maintenant aller dans le dossier `Outputs/` et voir les fabuleux graphs que tu viens de générer.

Bravo champion 🗿, tu viens de faire une analyse d'EyeTracking !
