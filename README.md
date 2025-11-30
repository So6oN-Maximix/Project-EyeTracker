# Projet d'Analyse Eye-Tracking

Ce projet utilise OpenCV, Scikit-learn et Matplotlib pour analyser des flux
vidéo d'eye-tracking et générer des timelines d'attention.

## Prérequis

* Python 3.10+
* [Poetry](https://python-poetry.org/) (pour la gestion des dépendances)

## Installation 

1.  Clonez ce dépôt :
    ```bash
    git clone [URL_DE_VOTRE_DEPOT]
    cd Project-EyeTracker
    ```

2.  Installez les dépendances avec Poetry :
    *(Cette commande lit le fichier `poetry.lock` et installe exactement les bonnes versions de `numpy`, `matplotlib`, etc.)*
    ```bash
    poetry install
    ```

3.  **Configuration des données :**
    Ce dépôt n'inclut pas les données vidéo. Vous devez placer les fichiers `.mp4` et `.csv` dans un dossier `datas/`.

4. **Gestion des sorties :**
	Ce dépôt n'inclut pas non plus les sorties que peuvent produire les différents programmes. Vous devez créer un dossier `outputs/` (Nom donné dans les codes) afin d'y acceuillir les données de sorties.

## Lancement de l'analyse

Pour lancer le script d'analyse principal :

```bash
poetry run python3 codes/Code_V3.0.py
```

Vous pouvez adapter le `python` en `python3` en fonction de votre configuration.
