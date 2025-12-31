FROM python:3.10-slim

# 1. Installation des dépendances système (toujours nécessaires pour OpenCV/MoviePy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Installer Poetry via pip
RUN pip install poetry

# 3. Configurer Poetry : Ne pas créer de virtualenv
RUN poetry config virtualenvs.create false

# Dossier de travail
WORKDIR /app

# 4. Copier UNIQUEMENT les fichiers de définition de dépendances
COPY pyproject.toml poetry.lock ./

# 5. Installer les dépendances
RUN poetry install --no-root --no-interaction --no-ansi

# Commande par défaut pour garder le conteneur en vie
CMD ["tail", "-f", "/dev/null"]
