FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /pandore_app

# Create user and directories
RUN useradd -m pandore
RUN mkdir -p /pandore_app/uploads \
   && mkdir -p /pandore_app/static/models/tessdata \
   && mkdir -p /pandore_app/temp \
   && mkdir -p /pandore_app/certificates

# Install system dependencies
RUN apt-get update && apt-get install -y \
   wget \
   net-tools \
   gcc \
   g++ \
   poppler-utils \
   tesseract-ocr \
   tesseract-ocr-fra \
   tesseract-ocr-eng \
   tesseract-ocr-chi-sim \
   tesseract-ocr-chi-tra \
   tesseract-ocr-spa \
   ffmpeg \
   locales \
   && rm -rf /var/lib/apt/lists/*

# Copy Tesseract data
RUN cp -r /usr/share/tesseract-ocr/4.00/tessdata/* /pandore_app/static/models/tessdata/ \
   && chmod -R 755 /pandore_app/static/models/tessdata

# Set Tesseract path
ENV TESSDATA_PREFIX=/pandore_app/static/models/tessdata


# Set up locale
RUN locale-gen fr_FR.UTF-8 && update-locale LANG=fr_FR.UTF-8
ENV LANG=fr_FR.UTF-8
ENV LANGUAGE=fr_FR.UTF-8
ENV LC_ALL=fr_FR.UTF-8

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
   /bin/bash Miniconda.sh -b -p /opt/conda && \
   rm Miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
ENV PIP_DEFAULT_TIMEOUT=300

# Accept the Terms of Service for the required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy the environment.yml file
COPY environment.yml ./

# Create conda environment from environment.yml
# The --yes flag ensures non-interactive installation
RUN conda env create -f environment.yml && \
    conda clean -afy
    
RUN conda run -n toolbox_env pip install gunicorn kraken

# Add toolbox_env binaries to PATH globally
ENV PATH=/opt/conda/envs/toolbox_env/bin:$PATH

# Install spaCy French model inside toolbox_env
RUN conda run -n toolbox_env python -m spacy download fr_core_news_lg

# Activate the environment for subsequent commands
SHELL ["conda", "run", "-n", "toolbox_env", "/bin/bash", "-c"]

# Set up NLTK data
ENV NLTK_DATA=/home/pandore/nltk_data
RUN mkdir -p ${NLTK_DATA} && \
    chown -R pandore:pandore /home/pandore && \
    python -m nltk.downloader -d ${NLTK_DATA} punkt stopwords averaged_perceptron_tagger maxent_ne_chunker words punkt_tab wordnet omw-1.4

# Copy the rest of your application code
COPY . /pandore_app

# Set permissions
RUN chown -R pandore:pandore /pandore_app /home/pandore

# Switch to the non-root user
USER pandore

EXPOSE 5000

# Replacing local host from toolbox_app
# DISABLED as NOT NECESSARY BEHIND FRONTEND
#RUN sed -i 's|if __name__ == "__main__":.*|if __name__ == "__main__":\n\
#    import os\n\
#    cert_file = "/pandore_app/certificates/cert.pem"\n\
#    key_file = "/pandore_app/certificates/server.key"\n\
#    if not os.path.isfile(cert_file):\n\
#        raise FileNotFoundError(f"Certificate file not found: {cert_file}")\n\
#    if not os.path.isfile(key_file):\n\
#        raise FileNotFoundError(f"Key file not found: {key_file}")\n\
#    print(f"Cert file permissions: {oct(os.stat(cert_file).st_mode)}")\n\
#    print(f"Key file permissions: {oct(os.stat(key_file).st_mode)}")\n\
#    ssl_context = (cert_file, key_file)\n\
#    print("Starting Pandore Toolbox with HTTPS...")\n\
#    app.run(host=\"0.0.0.0\", port=5000, debug=False, use_reloader=False, ssl_context=ssl_context)|g' toolbox_app.py


# === CMD syntax ===

# To run locally for debugging (HTTP)
# Comment out the production CMD below and uncomment this one.
#CMD ["/opt/conda/envs/toolbox-env/bin/gunicorn", "-c", "/dev/null", "--preload", "--workers", "1", "--timeout", "1000", "--bind", "0.0.0.0:5000", "--log-level", "debug", "toolbox_app:app"]

# For PRODUCTION server (HTTPS)
# This command first activates the conda environment, then executes gunicorn.
#CMD ["/opt/conda/envs/toolbox_env/bin/gunicorn", "-c", "/dev/null", "--workers", "7", "--timeout", "300", "--worker-connections","1000", "--threads", "4", "--bind", "0.0.0.0:5000", "--certfile", "/pandore_app/certificates/cert.pem", "--keyfile", "/pandore_app/certificates/server.key", "--timeout", "120", "--log-level", "info", "toolbox_app:app"]

# For PRODUCTION server behind NGINX frontend
# initial setup before my changes
#CMD ["/opt/conda/envs/toolbox_env/bin/gunicorn", "-c", "/dev/null", "--workers", "7", "--timeout", "300", "--worker-connections","1000", "--threads", "4", "--bind", "0.0.0.0:5000", "-e", "SCRIPT_NAME=/pandore", "--timeout", "120", "--log-level", "info", "toolbox_app:app"]

CMD ["/opt/conda/envs/toolbox_env/bin/gunicorn", "-c", "/dev/null", "--workers", "7", "--timeout", "600", "--worker-connections","1000", "--threads", "4", "--bind", "0.0.0.0:5000", "-e", "SCRIPT_NAME=/pandore", "--log-level", "info", "toolbox_app:app"]

#CMD ["/opt/conda/envs/toolbox_env/bin/gunicorn", "-c", "/dev/null", "--workers", "4","--worker-connections","1000", "--threads", "4", "--bind", "0.0.0.0:5000", "--certfile", "/pandore_app/certificates/cert.pem", "--keyfile", "/pandore_app/certificates/server.key", "--timeout", "120", "--log-level", "info", "toolbox_app:app"]
