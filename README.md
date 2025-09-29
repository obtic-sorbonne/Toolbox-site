# Pandore : une boîte à outil pour les humanités numériques

[English version of this README 🌍](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#pandore-a-toolbox-for-digital-humanities)

## Sommaire de ce README
* [Présentation du projet](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#présentation-du-projet)
* [Contenu du dépôt GitHub](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#contenu-du-dépôt-github)
* [Installation locale](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#installation-locale-linux--mac)
* [Pandore en ligne](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#pandore-en-ligne)
* [Bibliographie](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#bibliographie)
* [Mentions légales](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#mentions-légales)

-----

## Présentation du projet
Pandore est une boîte à outils conçue pour les chercheurs, les enseignants et les étudiants en sciences humaines et sociales, en particulier ceux qui n'ont pas de compétences techniques avancées en programmation. Elle permet aux utilisateurs de gérer efficacement leurs données grâce à des interfaces graphiques intuitives, leur permettant de collecter, préparer, convertir et analyser leurs données, d'extraire des informations pertinentes et de créer des visualisations interprétatives.   

En s'appuyant sur les nombreux retours de la version bêta précédente, plusieurs améliorations ont été mises en œuvre, notamment la résolution de bugs identifiés et les améliorations de l'interface graphique pour améliorer l'expérience utilisateur. Des scripts Python interopérables et modulaires ont également été intégrés pour étendre les fonctionnalités de la plateforme. Des tutoriels ont été finalisés pour guider les utilisateurs dans la maîtrise des outils, et l'application a été déployée sur un serveur équipé d'un GPU pour optimiser les performances des tâches à forte intensité de calcul.   

Ces avancées positionnent Pandore comme un outil polyvalent et efficace pour les chercheurs dans un large éventail de disciplines des sciences humaines et sociales. 

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

## Installation avec Docker (Linux / Mac / Windows)
Si vous n'avez jamais installé Python ou d'autres logiciels, Docker simplifie totalement l'installation. Pandore fonctionne dans un conteneur préconfiguré avec tous les paquets nécessaires.

**1. Installer Docker**   
- Linux / Mac / Windows : téléchargez et installez Docker Desktop depuis https://www.docker.com/get-started
- Lancez l'application.
  
**2. Récupérer le dépôt GitHub**  
- Après installation, ouvrez un terminal (ou PowerShell sous Windows).

```bash
git clone https://github.com/obtic-scai/Toolbox-site.git
cd Toolbox-site
```
**3. Construire l'image Docker de Pandore**
```bash
docker build -t pandore-toolbox .
```

- Cette commande télécharge Ubuntu, Miniconda, installe tous les paquets Python et spaCy, et crée une image Docker prête à l'emploi.

⚠️ Cela peut prendre plusieurs minutes, selon votre connexion Internet et votre machine.

**4. Lancer Pandore dans un conteneur Docker**

```bash
docker run --rm -p 5000:5000 pandore-toolbox
```

- -p 5000:5000 signifie : le port 5000 dans le conteneur sera accessible depuis le port 5000 sur votre machine.
- Accédez ensuite à Pandore dans votre navigateur à l'adresse : http://localhost:5000

**5. Arrêter Pandore**   
- Dans le terminal où Pandore tourne, appuyez sur CTRL+C pour arrêter le serveur.
- Le conteneur sera automatiquement supprimé grâce à l’option --rm.

💡 Conseil pour les utilisateurs avancés : vous pouvez changer le port local (5000) si le port est déjà pris, par exemple -p 8000:5000 pour accéder à http://localhost:8000

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


----

# Pandore: a toolbox for digital humanities

French version of this README : https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#pandore--une-boîte-à-outil-pour-les-humanités-numériques

-----
## Summary of this README
* [Project presentation](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#project-presentation)
* [Content of the GitHub repository](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#content-of-the-github-repository)
* [Docker Installation](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#docker-installation)
* [Pandore online](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#pandore-online)
* [Bibliography](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#bibliography)
* [Legal notices](https://github.com/obtic-sorbonne/Toolbox-site/blob/main/README.md#legal-notices)

-----

## Project presentation
Pandora is a toolbox designed for researchers, teachers, and students in the humanities and social sciences, particularly those without advanced technical programming skills. It enables users to efficiently manage their data through intuitive graphical interfaces, allowing them to collect, prepare, convert, and analyze their data, extract relevant information, and create interpretive visualizations.

Building on extensive feedback from the previous beta version, several improvements have been implemented, including the resolution of identified bugs and enhancements to the graphical interface to improve user experience. Interoperable and modular Python scripts have also been integrated to extend the platform's functionality. Tutorials have been finalized to guide users in mastering the tools, and the application has been deployed on a GPU-equipped server to optimize performance for computationally intensive tasks.

These advances position Pandora as a versatile and effective tool for researchers across a wide range of humanities and social sciences disciplines.

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

## Docker Installation

If you have never installed Python or other software, Docker completely simplifies installation. Pandora runs in a preconfigured container with all necessary packages.

### 1. Install Docker

- Linux / Mac / Windows: download and install Docker Desktop from https://www.docker.com/get-started
- Launch the application.

### 2. Clone the GitHub Repository

- After installation, open a terminal (or PowerShell on Windows).

```bash
git clone https://github.com/obtic-scai/Toolbox-site.git
cd Toolbox-site```


### 3. Build the Pandora Docker Image
```bash
docker build -t pandore-toolbox .
```

- This command downloads Ubuntu, Miniconda, installs all Python and spaCy packages, and creates a ready-to-use Docker image.

⚠️ This may take several minutes, depending on your internet connection and machine.

### 4. Launch Pandora in a Docker Container

```bash
docker run --rm -p 5000:5000 pandore-toolbox
```

- -p 5000:5000 means: port 5000 in the container will be accessible from port 5000 on your machine.
- Then access Pandora in your browser at: http://localhost:5000

### 5. Stop Pandora

- In the terminal where Pandora is running, press CTRL+C to stop the server.
- The container will be automatically removed thanks to the --rm option.

💡 Tip for advanced users: you can change the local port (5000) if the port is already in use, for example -p 8000:5000 to access http://localhost:8000

---

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

