FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /pandore_app
ADD . /pandore_app

# Create necessary directories first
RUN mkdir -p /pandore_app/uploads \
   && mkdir -p /pandore_app/static/models/tessdata \
   && mkdir -p /pandore_app/temp \
   && mkdir -p /pandore_app/certificates

# Install system dependencies
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
   nginx \
   && rm -rf /var/lib/apt/lists/*

# Copy Tesseract trained data
RUN cp -r /usr/share/tesseract-ocr/4.00/tessdata/* /pandore_app/static/models/tessdata/ \
   && chmod -R 755 /pandore_app/static/models/tessdata

# Setup locale
RUN locale-gen fr_FR.UTF-8 && update-locale LANG=fr_FR.UTF-8

ENV LANG fr_FR.UTF-8
ENV LANGUAGE fr_FR.UTF-8
ENV LC_ALL fr_FR.UTF-8
ENV TESSDATA_PREFIX=/pandore_app/static/models/tessdata
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=toolbox_app.py
ENV FLASK_DEBUG=0 
ENV FLASK_ENV=production
ENV PYTHONPATH=/pandore_app
ENV PYTHONWARNINGS="ignore::DeprecationWarning"

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
   /bin/bash Miniconda.sh -b -p /opt/conda && \
   rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Create and activate conda environment
RUN conda create -n toolbox-env python=3.9.13 && \
   conda clean -afy

SHELL ["conda", "run", "-n", "toolbox-env", "/bin/bash", "-c"]

# Install dependencies
RUN pip install -U pip setuptools wheel && \
   pip install docopt==0.6.2 && \
   pip install --no-cache-dir pip==23.0.1 && \
   pip install -r requirements.txt && \
   pip install protobuf==3.20.3 keybert==0.7.0 nltk wordcloud gunicorn && \
   pip install --upgrade numexpr>=2.8.4 bottleneck>=1.3.6 && \
   pip install contextualSpellCheck textdistance textstat plotly

# Install spacy and its models first
RUN pip install spacy && \
   python -m spacy download en_core_web_sm && \
   python -m spacy download en_core_web_lg && \
   python -m spacy download fr_core_news_sm && \
   python -m spacy download fr_core_news_md && \
   python -m spacy download fr_core_news_lg && \
   python -m spacy download es_core_news_sm && \
   python -m spacy download es_core_news_lg

# Install Flair with specific versions and create necessary directories
RUN pip install torch==1.13.1 && \
   pip install flair==0.11.3 && \
   pip install transformers==4.20.1 && \
   mkdir -p /home/pandore/.flair && \
   mkdir -p /home/pandore/.cache/huggingface

# Set cache directory environment variables  
ENV FLAIR_CACHE_ROOT=/home/pandore/.flair
ENV TRANSFORMERS_CACHE=/home/pandore/.cache/huggingface
ENV HF_HOME=/home/pandore/.cache/huggingface

# Download NLTK data
RUN python -m nltk.downloader punkt && \
   python -m nltk.downloader averaged_perceptron_tagger && \
   python -m nltk.downloader maxent_ne_chunker && \
   python -m nltk.downloader words && \
   python -m nltk.downloader punkt_tab

# Add nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Set up user and permissions (do this after all installations)
RUN useradd -m pandore && \
   chown -R pandore:pandore /pandore_app && \
   chown -R pandore:pandore /opt/conda/envs/toolbox-env/lib/python3.9/site-packages/spacy && \
   chown -R pandore:pandore /home/pandore/.flair && \
   chown -R pandore:pandore /home/pandore/.cache && \
   chmod 755 /pandore_app

RUN chown -R pandore:pandore /opt/conda/envs/toolbox-env/lib/python3.9/site-packages/en_core_web_* && \
    chown -R pandore:pandore /opt/conda/envs/toolbox-env/lib/python3.9/site-packages/es_core_news_* && \
    chown -R pandore:pandore /opt/conda/envs/toolbox-env/lib/python3.9/site-packages/fr_core_news_*   
    
# Switch to pandore user
USER pandore

ENV PATH=/home/pandore/.local/bin:$PATH

EXPOSE 5000

# Use gunicorn for production
CMD ["conda", "run", "--no-capture-output", "-n", "toolbox-env", \
    "gunicorn", "--workers", "4", "--threads", "2", \
    "--bind", "0.0.0.0:5000", \
    "--certfile", "/pandore_app/certificates/fullchain.pem", \
    "--keyfile", "/pandore_app/certificates/server.key", \
    "--timeout", "120", "--keep-alive", "5", \
    "toolbox_app:app"]