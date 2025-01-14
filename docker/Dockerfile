# Utiliser une image de base Python
FROM python:3.9-slim

# Mettre à jour les paquets et installer les dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install tiktoken  # Installer tiktoken ici

# Copier l'application et les règles
COPY app.py /app/app.py
COPY rules /app/rules

# Exposer le port 5000
EXPOSE 5000

# Démarrer l'application
CMD ["python", "app.py"]
