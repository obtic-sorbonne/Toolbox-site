# Pandore : une boîte à outil pour les humanités numériques

Version anglaise de ce README : https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md

-----
## Sommaire de ce README
* [Présentation du projet](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#présentation-du-projet)
* [Contenu du dépôt GitHub](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#contenu-du-dépôt-github)
* [Installation locale (Linux / Mac)](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#installation-locale-linux--mac)
* [Pandore en ligne](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#pandore-en-ligne)
* [Bibliographie](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#bibliographie)
* [Mentions légales](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#mentions-légales)

-----

## Présentation du projet
Pandore est une boîte à outils conçue pour les chercheurs, les enseignants et les étudiants en sciences humaines et sociales, en particulier ceux qui n'ont pas de compétences techniques avancées en programmation. Elle permet aux utilisateurs de gérer efficacement leurs données grâce à des interfaces graphiques intuitives, leur permettant de collecter, préparer, convertir et analyser leurs données, d'extraire des informations pertinentes et de créer des visualisations interprétatives. En s'appuyant sur les nombreux retours de la version bêta précédente, plusieurs améliorations ont été mises en œuvre, notamment la résolution de bugs identifiés et les améliorations de l'interface graphique pour améliorer l'expérience utilisateur. Des scripts Python interopérables et modulaires ont également été intégrés pour étendre les fonctionnalités de la plateforme. Des tutoriels ont été finalisés pour guider les utilisateurs dans la maîtrise des outils, et l'application a été déployée sur un serveur équipé d'un GPU pour optimiser les performances des tâches à forte intensité de calcul. Ces avancées positionnent Pandore comme un outil polyvalent et efficace pour les chercheurs dans un large éventail de disciplines des sciences humaines et sociales. 

------

## Contenu du dépôt GitHub
- **static** : Fichiers pour le bon affichage de l'instance (css, images, fonts, etc.)
- **swig** : Package nécessaire pour le fonctionnement de l'instance
- **templates** :
  - **documentation**, **tutoriel** : dossier contenant les documentation/tutoriel par tâche
  - **taches** : pages de navigation vers les outils
  - **outils** : outils par page
  - _413.html_, _500.html_, _500_custom.html_, _validation_contact.html_, _contact.html_, _code_source.html_, _copyright.html_, _projet.html_ : pages d'informations spécifiques
  - _documentation.html_, _index.html_, _pandore.html_, _tutoriel.html_ : pages de navigation
- **translations** : Traduction anglaise du contenu de l'instance
- _toolbox_app.py_ : Fichier de fonctionnement de l'instance (importation des libraires, déclaration des routes et des fonctions de l'application)
- _ner_camembert.py_, _ocr.py_, _tei_ner.py_, _txt_ner.py_ : Fonctions supplémentaires
- _requirements.txt_, _requirements_new.txt_ : Fichier de libraires pour l'installation locale
- _README.md_, _README_en.md_ : Présentation de la boîte à outils
- _cluster.py_, _environment.yml_, _forms.py_, _install.sh_, _nginx.conf_, _wsgi.py_ : TBD

-----

## Installation locale (Linux / Mac)
### Récupérer le dépôt GitHub
- Créer un dossier ObTIC-Toolbox et ouvrir un terminal dans celui-ci.

- Cloner le répertoire Toobox-site :
```bash
git clone https://github.com/obtic-scai/Toolbox-site.git
```

- Créer et activer un environnement virtuel (Python 3.6 et au-dessus) :
```bash
python3 -m venv toolbox-env
source toolbox-env/bin/activate
```

- Se placer dans le répertoire Toolbox-site :
```bash
cd Toolbox-site
```

### Installer les éléments nécessaires

- Il est nécessaire de [télécharger le paquet Swig](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz/download?use_mirror=netix). Pour l'installer, lancer :
```bash
./swig-3.0.12/configure && make && sudo make install
```

- Installer les paquets nécessaires à l'exécution de l'application :
```bash
pip install -r requirements.txt
```

- Finalement, il faut lancer :
```bash
chmod +x install.sh
bash install.sh
```

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

-----

## Pandore en ligne

Une [version de démonstration](https://obtic-gpu1.mesu.sorbonne-universite.fr:8550/) est disponible en ligne.

-----

## Bibliographie

Floriane Chiffoleau, Mikhail Biriuchinskii, Motasem Alrahabi, Glenn Roe. _Pandore: automating text-processing workflows for humanities researchers_. DH2025 - Accessibility & Citizenship, Jul 2025, Lisbon, Portugal. ⟨[hal-04986730]()https://hal.science/hal-04986730⟩ 

Motasem ALRAHABI, Valentina FEDCHENKO, Ljudmila PETKOVIC, Glenn ROE. _Pandore: a toolbox for digital humanities text-based workflows_. [soumission acceptée [DH2023](https://dh2023.adho.org/?page_id=390)]

Johanna Cordova, Yoann Dupont, Ljudmila Petkovic, James Gawley, Motasem Alrahabi, et al.. _Toolbox : une chaîne de traitement de corpus pour les humanités numériques_. Traitement Automatique des Langues Naturelles, 2022, Avignon, France. pp.11-13. ⟨[hal-03701464](https://hal.archives-ouvertes.fr/TALN-RECITAL2022/hal-03701464)⟩

-----

## Mentions légales

Le code est distribué sous licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) par l'[équipe ObTIC](https://obtic.sorbonne-universite.fr/) (Sorbonne Université).

