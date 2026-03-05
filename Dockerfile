FROM mambaorg/micromamba:1.5.8

WORKDIR /app

# Copy environment first for caching
COPY environment.yml /app/environment.yml

# Create env
RUN micromamba create -y -f /app/environment.yml && micromamba clean -a -y
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Copy app
COPY . /app

EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["micromamba", "run", "-n", "sdgnn-smiles-app", "streamlit", "run", "app.py"]
