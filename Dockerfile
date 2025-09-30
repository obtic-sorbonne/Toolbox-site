# Start with a generic Ubuntu base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
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

# === Conda installation ===
ARG TARGETARCH
RUN case ${TARGETARCH} in \
        "amd64") INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;; \
        "arm64") INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;; \
    esac && \
    wget "https://repo.anaconda.com/miniconda/${INSTALLER}" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda config --set always_yes yes --set changeps1 no && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy env
COPY environment.yml .

# Create environment
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "toolbox_env", "/bin/bash", "-c"]

# Copy code
COPY . .

EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "toolbox_env", "python", "toolbox_app.py"]
