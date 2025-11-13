# Projet d'Analyse Eye-Tracking

Ce projet utilise OpenCV, Scikit-learn et Matplotlib pour analyser des flux
vidéo d'eye-tracking et générer des timelines d'attention.

## 🚀 Prérequis

* Python 3.10+
* [Poetry](https://python-poetry.org/) (pour la gestion des dépendances)

## 🛠️ Installation 

1.  Clonez ce dépôt :
    ```bash
    git clone [URL_DE_VOTRE_DEPOT]
    cd mon_projet_eye_tracking
    ```

2.  Installez les dépendances avec Poetry :
    *(Cette commande lit le fichier `poetry.lock` et installe 
    exactement les bonnes versions de `numpy`, `matplotlib`, etc.)*
    ```bash
    poetry install
    ```

3.  **Configuration des données :**
    Ce dépôt n'inclut pas les données vidéo. Vous devez placer vos propres
    fichiers `.mp4` dans le dossier `data/`.

## 🏃 Lancement de l'analyse

Pour lancer le script d'analyse principal :

```bash
poetry run codes/Code_principal_2024-Modified.py
