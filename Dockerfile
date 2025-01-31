FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /pandore_app
ADD . /pandore_app

# Create necessary directories first, including certificates
RUN mkdir -p /pandore_app/uploads \
    && mkdir -p /pandore_app/static/models/tessdata \
    && mkdir -p /pandore_app/temp \
    && mkdir -p /pandore_app/certificates

# Create a non-root user
RUN useradd -m pandore

# Install system dependencies including ALL Tesseract language packs
RUN apt-get update && apt-get install -y \
    wget \
    net-tools \
    gcc \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-spa \
    tesseract-ocr-spa-old \
    tesseract-ocr-frk \
    tesseract-ocr-grc \
    tesseract-ocr-lat \
    tesseract-ocr-por \
    locales \
    libpcre3 \
    libpcre3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Tesseract trained data to our custom tessdata directory
RUN cp -r /usr/share/tesseract-ocr/4.00/tessdata/* /pandore_app/static/models/tessdata/ \
    && chmod -R 777 /pandore_app/static/models/tessdata

# Configure SWIG
RUN cd swig && ./swig-3.0.12/configure && make && make install

# Setup locale
RUN locale-gen fr_FR.UTF-8 && update-locale LANG=fr_FR.UTF-8

ENV LANG fr_FR.UTF-8
ENV LANGUAGE fr_FR.UTF-8
ENV LC_ALL fr_FR.UTF-8
ENV TESSDATA_PREFIX=/pandore_app/static/models/tessdata

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=toolbox_app.py
ENV FLASK_ENV=production

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
    /bin/bash Miniconda.sh -b -p /opt/conda && \
    rm Miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Create and configure conda environment
RUN conda create -n toolbox-env python=3.9.13 && \
    conda clean -afy

# Activate environment and install packages
SHELL ["conda", "run", "-n", "toolbox-env", "/bin/bash", "-c"]

# Install pip packages and requirements first
RUN pip install -U pip setuptools wheel
RUN pip install docopt==0.6.2

RUN pip install --no-cache-dir pip==23.0.1

# Core requirements
RUN pip install -r requirements.txt

# Install bert and keybert
RUN pip install protobuf==3.20.3
RUN pip install keybert==0.7.0

# Install NLTK and its data
RUN pip install nltk
RUN python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words punkt_tab stopwords

# Install wordcloud
RUN pip install wordcloud

RUN pip install --upgrade "numexpr>=2.8.4" "bottleneck>=1.3.6"

# Install spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download fr_core_news_md
RUN python -m spacy download fr_core_news_lg

# Copy certificates into the container and set correct ownership and permissions
# COPY --chown=pandore:pandore certificates/cert-fullchain.pem /pandore_app/certificates/cert.pem
# COPY --chown=pandore:pandore certificates/server.key /pandore_app/certificates/server.key

# RUN chmod 644 /pandore_app/certificates/cert.pem && \
#     chmod 600 /pandore_app/certificates/server.key

# USER pandore
# Copy certificates and set ownership/permissions
#COPY certificates/cert-fullchain.pem /pandore_app/certificates/cert.pem
#COPY certificates/server.key /pandore_app/certificates/server.key

# Verification and debugging steps
RUN echo "For certificate:" && \
    ls -l /pandore_app/certificates/ && \
    echo "For fullchain:" && \
    openssl x509 -noout -modulus -in /pandore_app/certificates/cert.pem | openssl md5 && \
    echo "Certificate modulus:" && \
    openssl x509 -noout -modulus -in /pandore_app/certificates/fullchain.pem | openssl md5 && \
    echo "For private key:" && \
    openssl rsa -noout -modulus -in /pandore_app/certificates/server.key | openssl md5

RUN chmod 644 /pandore_app/certificates/fullchain.pem && \
    chmod 600 /pandore_app/certificates/server.key && \
    chown -R pandore:pandore /pandore_app/certificates

USER pandore

# Add local bin to PATH for the pandore user
ENV PATH=/home/pandore/.local/bin:$PATH

EXPOSE 5000

# Add these volume configurations
VOLUME ["/pandore_app/uploads", "/pandore_app/static/models"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "toolbox-env", "python", "toolbox_app.py"]