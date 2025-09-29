# Start with a generic Ubuntu base image. Docker will automatically pull the correct
# architecture (amd64 or arm64) for your machine.
FROM ubuntu:22.04

# Avoid interactive prompts during the build process
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies. These are available for both architectures.
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    swig \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/*

# === ARCHITECTURE-AWARE MINICONDA INSTALLATION ===
ARG TARGETARCH
RUN case ${TARGETARCH} in \
        "amd64") INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;; \
        "arm64") INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;; \
    esac && \
    wget "https://repo.anaconda.com/miniconda/${INSTALLER}" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to the system's PATH
ENV PATH="/opt/conda/bin:${PATH}"

# === ACCEPT ANACONDA TERMS OF SERVICE ===
RUN conda config --set always_yes yes --set changeps1 no && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# =============================================

# Copy the environment definition file into the container
COPY environment.yml .

# Create the Conda environment from the .yml file.
RUN conda env create -f environment.yml && conda clean -afy

# Activate the Conda environment for all subsequent commands in this Dockerfile.
SHELL ["conda", "run", "-n", "toolbox_env", "/bin/bash", "-c"]

# Download the necessary spaCy models inside the activated environment
RUN python -m spacy download fr_core_news_lg && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm && \
    python -m spacy download fr_core_news_md && \
    python -m spacy download es_core_news_sm && \
    python -m spacy download de_core_news_sm && \
    python -m spacy download da_core_news_sm && \
    python -m spacy download nl_core_news_sm && \
    python -m spacy download fi_core_news_sm && \
    python -m spacy download it_core_news_sm && \
    python -m spacy download pt_core_news_sm && \
    python -m spacy download el_core_news_sm && \
    python -m spacy download ru_core_news_sm

# Copy the rest of your application's source code into the container
COPY . .

# Expose port 5000 to access the application from your local machine
EXPOSE 5000

# Set the command to run when the container starts.
CMD ["conda", "run", "--no-capture-output", "-n", "toolbox_env", "python", "toolbox_app.py"]
