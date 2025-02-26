# Toolbox-site

## Installation locale (Linux / Mac)

- Créer un dossier ObTIC-Toolbox et ouvrir un terminal dans celui-ci.

- Cloner le répertoire Toobox-site :

`git clone https://github.com/obtic-scai/Toolbox-site.git`

- Créer et activer un environnement virtuel (Python 3.6 et au-dessus) :

`python3 -m venv toolbox-env`

`source toolbox-env/bin/activate`

- Se placer dans le répertoire Toolbox-site :

`cd Toolbox-site`

- Il est nécessaire de [télécharger le paquet Swig](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz/download?use_mirror=netix). Pour l'installer, lancer :

`./swig-3.0.12/configure && make && sudo make install`

- Installer les paquets nécessaires à l'exécution de l'application :

`pip install -r requirements.txt`

- Finalement, il faut lancer :

`chmod +x install.sh` puis `bash install.sh`

Il se peut qu'il faille lancer les commandes suivantes : 

```bash
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download fr_core_news_md
python -m spacy download fr_core_news_lg
python -m spacy download es_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download da_core_news_sm
python -m spacy download nl_core_news_sm
python -m spacy download fi_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download el_core_news_sm
python -m spacy download ru_core_news_sm
```

### Lancer l'application

Placé dans le dossier Toolbox-site, lancer la commande :

```bash
python3 toolbox_app.py
```

Ouvrir le lien http://127.0.0.1:5000 dans un navigateur pour accéder à l'interface de la Toolbox ObTIC.

## Version en ligne

Une [version de démonstration](https://obtic-gpu1.mesu.sorbonne-universite.fr:8550/) est disponible en ligne.
Une nouvelle version pour diffusion plus large est en cours de conception.

____



# Bibliographie

Motasem ALRAHABI, Valentina FEDCHENKO, Ljudmila PETKOVIC, Glenn ROE. Pandore: a toolbox for digital humanities text-based workflows. [soumission acceptée [DH2023](https://dh2023.adho.org/?page_id=390)]

Johanna Cordova, Yoann Dupont, Ljudmila Petkovic, James Gawley, Motasem Alrahabi, et al.. Toolbox : une chaîne de traitement de corpus pour les humanités numériques. *Traitement Automatique des Langues Naturelles*, 2022, Avignon, France. pp.11-13. ⟨[hal-03701464](https://hal.archives-ouvertes.fr/TALN-RECITAL2022/hal-03701464)⟩


# Mentions légales

Le code est distribué sous licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) par l'[équipe ObTIC](https://obtic.sorbonne-universite.fr/) (Sorbonne Université).

# 
