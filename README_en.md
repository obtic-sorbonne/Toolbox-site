# Pandore: a toolbox for digital humanities

French version of this README : https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md

-----
## Summary of this README
* [Project presentation](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md#project-presentation)
* [Content of the GitHub repository](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md#content-of-the-github-repository)
* [Local installation (Linux / Mac)](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md#local-installation-linux--mac)
* [Pandore online](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md#pandore-online)
* [Bibliography](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md#bibliography)
* [Legal notices](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README_en.md#legal-notices)

-----

## Project presentation
Pandore is a toolkit designed for researchers, teachers, and students in the humanities and social sciences, in particular those without advanced technical coding skills. It enables users to efficiently manage their data through intuitive graphical interfaces, allowing them to collect, prepare, convert, and analyze their data, extract relevant insights, and create interpretive visualizations.   
Building on extensive feedback from the previous beta version, several improvements have been implemented, including the resolution of identified bugs and enhancements to the graphical interface to improve user experience. Interoperable and modular Python scripts have also been integrated to extend the platform's functionalities. Tutorials have been finalized to guide users in mastering the tools, and the application has been deployed on a GPU-equipped server to optimize performance for computationally intensive tasks.   
These advancements position Pandore as a versatile and effective tool for researchers across a diverse range of humanities and social science disciplines. 

------

## Content of the GitHub repository
- **static** : Files for the correct display of the instance (css, images, fonts, etc.)
- **swig** : Package required for the instance to run
- **templates** :
  - **documentation**, **tutoriel** : folder containing the documentation/tutorial by task
  - **taches** : navigation pages to tools
  - **outils** : tools per page
  - _413.html_, _500.html_, _500_custom.html_, _validation_contact.html_, _contact.html_, _code_source.html_, _copyright.html_, _projet.html_ : specific information pages
  - _documentation.html_, _index.html_, _pandore.html_, _tutoriel.html_ : navigation pages
- **translations** : English translation of the contents of the instance
- _toolbox_app.py_ : Instance operating file (import of libraries, declaration of routes and application functions)
- _ner_camembert.py_, _ocr.py_, _tei_ner.py_, _txt_ner.py_ : Additional functions
- _requirements.txt_, _requirements_new.txt_ : Libraries file for local installation
- _README.md_, _README_en.md_ : Presentation of the toolbox
- _cluster.py_, _environment.yml_, _forms.py_, _install.sh_, _nginx.conf_, _wsgi.py_ : TBD

-----

## Local installation (Linux / Mac)
### Retrieve the GitHub repository
- Create a folder ObTIC-Toolbox and open a terminal in it.

- Clone the Toolbox-site directory:
```bash
git clone https://github.com/obtic-scai/Toolbox-site.git
```

- Create and activate a virtual environment (Python 3.6 and above):
```bash
python3 -m venv toolbox-env
source toolbox-env/bin/activate
```

- Place yourself in the directory Toolbox-site:
```bash
cd Toolbox-site
```

### Install the necessary elements

- It is necessary to [download the Swig package](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz/download?use_mirror=netix). To install it, launch:
```bash
./swig-3.0.12/configure && make && sudo make install
```

- Install the packages needed to run the application:
```bash
pip install -r requirements.txt
```

- Finally, you have to launch:
```bash
chmod +x install.sh
bash install.sh
```

You may need to run the following commands: 

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

### Launch the application

Placed in the Toolbox-site folder, run the command:

```bash
python3 toolbox_app.py
```

Open the link http://127.0.0.1:5000 in a browser to access the ObTIC Toolbox interface.

-----

## Pandore online

A [demo version](https://obtic-gpu1.mesu.sorbonne-universite.fr:8550/) is available online.

-----

## Bibliography

Floriane Chiffoleau, Mikhail Biriuchinskii, Motasem Alrahabi, Glenn Roe. _Pandore: automating text-processing workflows for humanities researchers_. DH2025 - Accessibility & Citizenship, Jul 2025, Lisbon, Portugal. ⟨[hal-04986730]()https://hal.science/hal-04986730⟩ 

Motasem ALRAHABI, Valentina FEDCHENKO, Ljudmila PETKOVIC, Glenn ROE. _Pandore: a toolbox for digital humanities text-based workflows_. [soumission acceptée [DH2023](https://dh2023.adho.org/?page_id=390)]

Johanna Cordova, Yoann Dupont, Ljudmila Petkovic, James Gawley, Motasem Alrahabi, et al.. _Toolbox : une chaîne de traitement de corpus pour les humanités numériques_. Traitement Automatique des Langues Naturelles, 2022, Avignon, France. pp.11-13. ⟨[hal-03701464](https://hal.archives-ouvertes.fr/TALN-RECITAL2022/hal-03701464)⟩

-----

## Legal notices

The code is distributed under license [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by the [ObTIC team](https://obtic.sorbonne-universite.fr/) (Sorbonne Université).

