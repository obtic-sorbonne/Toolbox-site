#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Standard library imports
import csv
import difflib
import json
import os
import random
import re
import shutil
import string
import unicodedata
import urllib.request
from datetime import timedelta
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urlparse
import time

import nltk
import numpy as np
import pandas as pd
from charset_normalizer import from_bytes
# Third-party imports
import requests
from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, session,
                   stream_with_context, url_for)
from flask_babel import Babel, get_locale
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFError, CSRFProtect
from lxml import etree
from nltk import FreqDist, Text, ngrams
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

# Local application imports
from forms import ContactForm

# NLTK
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download error: {e}")

import logging
import sys

# Configure logging to be VERY verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout  # Print to Docker logs
)
logger = logging.getLogger(__name__)

# Log when app starts
logger.info("="*50)
logger.info("PANDORE TOOLBOX STARTING")
logger.info("="*50)


UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'static/models'
UTILS_FOLDER = 'static/utils'
ROOT_FOLDER = Path(__file__).parent.absolute()

# Flask's CSRF (Cross-Site Request Forgery) protection
csrf = CSRFProtect()

app = Flask(__name__)

# Babel config
#def get_locale():
#    return request.accept_languages.best_match(['fr', 'en'])
#babel = Babel(app, locale_selector=get_locale)
babel = Babel(app)

# App config
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = '6kGcDYu04nLGQZXGv8Sqg0YzTeE8yeyL'
app.config['WTF_CSRF_TIME_LIMIT'] = None  # No time limit on tokens
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)  # Session lasts 1 day

app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024 # Limit file upload to 150MB
app.config['CHUNK_SIZE'] = 1024 * 1024 # 1MB chunks for processing

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['UTILS_FOLDER'] = UTILS_FOLDER
app.config['LANGUAGES'] = {
    'fr': 'FR',
    'en': 'EN',
}
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)

csrf.init_app(app)


#--------------------------------------------------------------
# Generic functions
#-------------------------------------------------------------

def create_zip_and_response(result_path, rand_name):
    if len(os.listdir(result_path)) > 0:
        shutil.make_archive(result_path, 'zip', result_path)
        output_stream = BytesIO()
        with open(str(result_path) + '.zip', 'rb') as res:
            content = res.read()
        output_stream.write(content)
        response = Response(
            output_stream.getvalue(),
            mimetype='application/zip',
            headers={"Content-disposition": f"attachment; filename={rand_name}.zip"}
        )
        output_stream.seek(0)
        output_stream.truncate(0)
        shutil.rmtree(result_path)
        os.remove(str(result_path) + '.zip')
        return response
    else:
        return Response(
            json.dumps({"error": "Aucune donnée à archiver."}),
            status=500,
            mimetype='application/json'
        )

def generate_rand_name(prefix, length=5):
    """
    Génère un nom unique avec un préfixe et une chaîne aléatoire.

    Args:
        prefix (str): Le préfixe pour le nom généré.
        length (int): Longueur de la chaîne aléatoire (par défaut 5).

    Returns:
        str: Nom généré unique.
    """
    return prefix + ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def create_named_directory(rand_name, base_dir=None):
    """
    Crée un répertoire avec un nom spécifique fourni.

    Args:
        rand_name (str): Le nom unique du répertoire à créer.
        base_dir (str, optional): Le répertoire de base où le répertoire sera créé.
                                  Par défaut, le répertoire de travail actuel (`os.getcwd()`).

    Returns:
        str: Le chemin complet du répertoire créé.
    """
    base_dir = base_dir or os.getcwd()
    result_path = os.path.join(base_dir, rand_name)
    os.makedirs(result_path, exist_ok=True)
    return result_path


#-----------------------------------------------------------------
# error handlers to catch CSRF errors gracefully
#-----------------------------------------------------------------
@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('errors/csrf_error.html', reason=e.description), 400

#-----------------------------------------------------------------
# BABEL
#-----------------------------------------------------------------
@app.route('/language=<language>')
def set_language(language=None):
    session['language'] = language
    return redirect(url_for('index'))

def get_locale():
    if request.args.get('language'):
        session['language'] = request.args.get('language')
    return session.get('language', 'fr')

babel.init_app(app, locale_selector=get_locale)

@app.context_processor
def inject_conf_var():
    return dict(AVAILABLE_LANGUAGES=app.config['LANGUAGES'], CURRENT_LANGUAGE=session.get('language', request.accept_languages.best_match(app.config['LANGUAGES'])))


#-----------------------------------------------------------------
# ROUTES
#-----------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pandore')
def pandore():
    return render_template('pandore.html')

@app.route('/projet')
def projet():
    return render_template('projet.html')

@app.route('/code_source')
def code_source():
    return render_template('code_source.html')

@app.route('/contact')
def contact():
    form = ContactForm()
    return render_template('contact.html', form=form)

@app.route('/copyright')
def copyright():
    return render_template('copyright.html')

#-----------------------------------------------------------------
# DOCUMENTATION
#-----------------------------------------------------------------

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/documentation_recognition')
def documentation_recognition():
    return render_template('documentation/documentation_recognition.html')

@app.route('/documentation_preprocessing')
def documentation_preprocessing():
    return render_template('documentation/documentation_preprocessing.html')

@app.route('/documentation_conversion')
def documentation_conversion():
    return render_template('documentation/documentation_conversion.html')

@app.route('/documentation_annotation')
def documentation_annotation():
    return render_template('documentation/documentation_annotation.html')

@app.route('/documentation_extraction')
def documentation_extraction():
    return render_template('documentation/documentation_extraction.html')

@app.route('/documentation_analyses')
def documentation_analyses():
    return render_template('documentation/documentation_analyses.html')

#-----------------------------------------------------------------
# TUTORIELS
#-----------------------------------------------------------------

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')
    
@app.route('/tutoriel_conversion')
def tutorial_conversion():
    return render_template('tutorial/tutorial_conversion.html')

@app.route('/tutoriel_annotation')
def tutorial_annotation():
    return render_template('tutorial/tutorial_annotation.html')

@app.route('/tutoriel_extraction')
def tutorial_extraction():
    return render_template('tutorial/tutorial_extraction.html')
    
#-----------------------------------------------------------------
# TASKS
#-----------------------------------------------------------------

@app.route('/atr_tools')
def atr_tools():
    return render_template('tasks/atr_tools.html')

@app.route('/preprocessing')
def preprocessing():
    return render_template('tasks/preprocessing.html')

@app.route('/conversion')
def conversion():
    return render_template('tasks/conversion.html')

@app.route('/automatic_annotation')
def automatic_annotation():
    return render_template('tasks/automatic_annotation.html')

@app.route('/information_extraction')
def information_extraction():
    return render_template('tasks/information_extraction.html')

@app.route('/analyses')
def analyses():
    return render_template('tasks/analyses.html')

@app.route('/search_tools')
def search_tools():
    return render_template('tasks/search_tools.html')

@app.route('/visualisation_tools')
def visualisation_tools():
    return render_template('tasks/visualisation_tools.html')

@app.route('/corpus_collection')
def corpus_collection():
    return render_template('tasks/corpus_collection.html')

@app.route('/pipelines')
def pipelines():
    return render_template('tasks/pipelines.html')

#-----------------------------------------------------------------
# TOOLS
#-----------------------------------------------------------------

@app.route('/text_recognition')
def text_recognition():
    form = FlaskForm()
    return render_template('tools/text_recognition.html', form=form)

@app.route('/handwritten_text_recognition')
def handwritten_text_recognition():
    form = FlaskForm()
    return render_template('tools/handwritten_text_recognition.html', form=form)

@app.route('/speech_recognition')
def speech_recognition():
    form = FlaskForm()
    return render_template('tools/speech_recognition.html', form=form)

@app.route('/error_correction')
def error_correction():
    form = FlaskForm()
    return render_template('tools/error_correction.html', form=form)

@app.route('/text_cleaning')
def text_cleaning():
    form = FlaskForm()
    return render_template('tools/text_cleaning.html', form=form)

@app.route('/text_normalisation')
def text_normalisation():
    form = FlaskForm()
    return render_template('tools/text_normalisation.html', form=form)

@app.route('/text_separation')
def text_separation():
    form = FlaskForm()
    return render_template('tools/text_separation.html', form=form)

@app.route('/conversion_xml')
def conversion_xml():
    form = FlaskForm()
    return render_template('tools/conversion_xml.html', form=form)

@app.route('/partofspeech_tagging')
def partofspeech_tagging():
    form = FlaskForm()
    err = ""
    return render_template('tools/partofspeech_tagging.html', form=form, err=err)

@app.route('/named_entities_recognition')
def named_entities_recognition():
    form = FlaskForm()
    return render_template('tools/named_entities_recognition.html', form=form)

@app.route('/semantic_categories')
def semantic_categories():
    return render_template('tools/semantic_categories.html')

@app.route('/keywords_extraction')
def keywords_extraction():
    form = FlaskForm()
    return render_template('tools/keywords_extraction.html', form=form, res={})

@app.route('/topic_modelling')
def topic_modelling():
    form = FlaskForm()
    return render_template('tools/topic_modelling.html', form=form, res={})

@app.route('/quotation_extraction')
def quotation_extraction():
    form = FlaskForm()
    return render_template('tools/quotation_extraction.html', form=form)

@app.route('/summarizer')
def summarizer():
    return render_template('tools/summarizer.html')

@app.route('/linguistic_analysis')
def linguistic_analysis():
    form = FlaskForm()
    return render_template('tools/linguistic_analysis.html', form=form)

@app.route('/statistic_analysis')
def statistic_analysis():
    form = FlaskForm()
    return render_template('tools/statistic_analysis.html', form=form)

@app.route('/text_analysis')
def text_analysis():
    form = FlaskForm()
    return render_template('tools/text_analysis.html', form=form)

@app.route('/comparison')
def comparison():
    form = FlaskForm()
    return render_template('tools/comparison.html', form=form)

@app.route('/tanagra')
def tanagra():
    return render_template('tools/tanagra.html')

@app.route('/renard')
def renard():
    form = FlaskForm()
    return render_template('tools/renard.html', form=form, graph="", fname="")

@app.route('/extraction_gallica')
def extraction_gallica():
    form = FlaskForm()
    return render_template('tools/extraction_gallica.html', form=form)

@app.route('/extraction_wikisource')
def extraction_wikisource():
    form = FlaskForm()
    return render_template('tools/extraction_wikisource.html', form=form)

@app.route('/extraction_gutenberg')
def extraction_gutenberg():
    form = FlaskForm()
    return render_template('tools/extraction_gutenberg.html', form=form)

@app.route('/extraction_urls')
def extraction_urls():
    form = FlaskForm()
    return render_template('tools/extraction_urls.html', form=form)

@app.route('/ocr_ner')
def ocr_ner():
    form = FlaskForm()
    return render_template('tools/ocr_ner.html', form=form)

@app.route('/ocr_map')
def ocr_map():
    form = FlaskForm()
    return render_template('tools/ocr_map.html', form=form)

#-----------------------------------------------------------------
# ERROR HANDLERS
#-----------------------------------------------------------------
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.errorhandler(413)
def file_too_big(e):
    return render_template('413.html'), 413

@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    # now you're handling non-HTTP exceptions only
    return render_template("500_custom.html", e=e), 500

#-----------------------------------------------------------------
# FONCTIONS
#-----------------------------------------------------------------
@app.route('/send_msg',  methods=["GET","POST"])
def send_msg():
    if request.method == 'POST':
        name =  request.form["name"]
        email = request.form["email"]
        message = request.form["message"]
        res = pd.DataFrame({'name':name, 'email':email,'message':message}, index=[0])
        res.to_csv('./contactMsg.csv')
        return render_template('validation_contact.html')
    return render_template('contact.html', form=form)

# TELECHARGEMENT DE FICHIER
@app.route('/download')
def download():
    path = 'static/textolab.zip'
    return send_file(path, as_attachment=True)


#-----------------------------------------------------------------
# Extraction de corpus
#-----------------------------------------------------------------

@app.route('/generate_corpus',  methods=["GET","POST"])
@stream_with_context
def generate_corpus():
    if request.method == 'POST':
        nb = int(request.form['nbtext'])
        result_path, rand_name = createRandomDir('wiki_', 8)
        all_texts = generate_random_corpus(nb)
        
        #Crée les fichiers .txt
        for clean_text,text_title in all_texts:
            filename = text_title
            with open(os.path.join(result_path, filename)+'.txt', 'w', encoding='utf-8') as output:
                output.write(clean_text)

        # ZIP le dossier résultat
        if len(os.listdir(result_path)) > 0:
            shutil.make_archive(result_path, 'zip', result_path)
            output_stream = BytesIO()
            with open(str(result_path) + '.zip', 'rb') as res:
                content = res.read()
            output_stream.write(content)
            response = Response(output_stream.getvalue(), mimetype='application/zip',
                                    headers={"Content-disposition": "attachment; filename=" + rand_name + '.zip'})
            output_stream.seek(0)
            output_stream.truncate(0)
            return response
        else:
            shutil.rmtree(result_path)
                
    return render_template('/corpus_collection.html')

@app.route('/corpus_from_url',  methods=["GET","POST"])
@stream_with_context

#Modifiée pour travail local + corrections
def corpus_from_url():
    from bs4 import BeautifulSoup
    from lxml.html.clean import clean_html
    if request.method == 'POST':
        keys = request.form.keys()
        urls = [k for k in keys if k.startswith('url')]
        #urls = sorted(urls)
        
        result_path, rand_name = createRandomDir('wiki_', 8)

        # PARCOURS DES URLS UTILISATEUR
        for url_name in urls:
            url = request.form.get(url_name)
            if not url:
                continue
            n = url_name.split('_')[1]
            s = 's' + n
            path_elems = urlparse(url).path.split('/')

            # L'URL pointe vers un sommaire
            if path_elems[-1] != 'Texte_entier' and request.form.get(s) == 'on':
                # Escape URL if not already escaped
                url_temp = url.replace("https://fr.wikisource.org/wiki/", "")
                if not '%' in url_temp:
                    url = "".join(["https://fr.wikisource.org/wiki/", urllib.parse.quote(url_temp)])
                try:
                    index_page = urllib.request.urlopen(url)
                    index_soup = BeautifulSoup(index_page, 'html.parser')
                    nodes = index_soup.select('div.prp-pages-output div[class="tableItem"] a')
                    for a in nodes:
                        link = 'https://fr.wikisource.org' + a['href']
                        name = a['title']
                        if '/' in name:
                            name = name.split('/')[-1]
                        text = getWikiPage(link)
                        if text != -1:
                            if not name:
                                name = path_elems[-1]
                            with open(os.path.join(result_path, name)+'.txt', 'w', encoding='utf-8') as output:
                                output.write(text)
                            with open(os.path.join(result_path, "rapport.txt"), 'a') as rapport:
                                rapport.write(link + '\t' + 'OK\n')

                except urllib.error.HTTPError:
                    print(" ".join(["The page", url, "cannot be opened."]))
                    with open(os.path.join(result_path, "rapport.txt"), 'a') as rapport:
                                rapport.write(url + '\t' + "Erreur : l'URL n'a pas pu être ouverte.\n")
                    continue

                filename = urllib.parse.unquote(path_elems[-1])

            # URL vers texte intégral
            else:
                try:
                    clean_text = getWikiPage(url)
                    if clean_text == -1:
                        print("Erreur lors de la lecture de la page {}".format(url))
                        with open(os.path.join(result_path, "rapport.txt"), 'a') as rapport:
                                rapport.write(url + '\t' + "Erreur : le contenu de la page n'a pas pu être lu.\n")

                    else:
                        if path_elems[-1] != 'Texte_entier':
                            filename = urllib.parse.unquote(path_elems[-1])
                        else:
                            filename = urllib.parse.unquote(path_elems[-2])

                        with open(os.path.join(result_path, filename)+'.txt', 'w', encoding='utf-8') as output:
                            output.write(clean_text)

                except Exception as e:
                    print("Erreur sur l'URL {}".format(url))
                    continue


        # ZIP le dossier résultat
        response = create_zip_and_response(result_path, rand_name)
        return response

    return render_template('corpus_collection.html')

#----------------------- Wikisource -------------------

def generate_random_corpus(nb):
    from bs4 import BeautifulSoup

    # Read list of urls
    with open(ROOT_FOLDER / 'static/wikisource_bib.txt', 'r') as bib:
        random_texts = bib.read().splitlines()

    # Pick random urls
    urls = random.sample(random_texts, nb)
    all_texts = []

    for text_url in urls:
        #removes the subsidiary part of the url path ("/Texte_entier" for example) so it does not mess with the filename
        if(re.search('/',text_url)):
            text_title = urllib.parse.unquote(text_url.split('/')[0])
        else:
            text_title = urllib.parse.unquote(text_url)
        location = "".join(["https://fr.wikisource.org/wiki/", text_url])
        try:
            req = urllib.request.Request(
                location,
                headers={'User-Agent': 'Pandore-Toolbox/1.0 (Educational; contact@sorbonne-universite.fr)'}
            )
            page = urllib.request.urlopen(req)
            logger.debug(f"✓ Successfully opened URL: {location}")
        except Exception as e:
            logger.error(f"✗ Failed to open URL: {location}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Error message: {str(e)}")
            with open('pb_url.log', 'a') as err_log:
                err_log.write("No server is associated with the following page:" + location + '\n')
                err_log.write(str(e) + '\n')  # FIX APPLIED
            continue

        soup = BeautifulSoup(page, 'html.parser')
        text = soup.findAll("div", attrs={'class': 'prp-pages-output'})

        if len(text) == 0:
            print("This does not appear to be part of the text (no prp-pages-output tag at this location).")
            with open('pb_url.log', 'a') as err_log:
                err_log.write(text_url)
        else:
            # Remove end of line inside sentence
            clean_text = re.sub(r"[^\.:!?»[A-Z]]\n", ' ', text[0].text)
            all_texts.append((clean_text,text_title))

    return all_texts

#---------------- Gallica ------------------------

@app.route('/extract_gallica', methods=["GET", "POST"])
@stream_with_context
def extract_gallica():
    from bs4 import BeautifulSoup

    form = FlaskForm()
    input_format = request.form['input_format']
    res_ok = ""
    res_err = ""
    res = ""

    # Arks dans fichier
    if request.files['ark_upload'].filename != '':
        f = request.files['ark_upload']
        text = f.read().decode('utf-8')
        arks_list = re.split(r"[~\r\n]+", text)
    # Arks dans textarea
    else:
        arks_list = re.split(r"[~\r\n]+", request.form['ark_input'])

    # Prépare le dossier résultat
    rand_name =  generate_rand_name('corpus_gallica_')
    result_path = create_named_directory(rand_name)

    # Définir les détails de connexion à l'API Gallica. Fermer la connexion pour éviter d'être coupé par Gallica
    GALLICA_HEADERS = {
        'User-Agent': 'Pandore-Toolbox/1.0 (Educational; contact@sorbonne-universite.fr)',
        'Connection': 'close'
    }
    
    for arkEntry in arks_list:
        # Vérifie si une plage de pages est indiquée
        arkEntry = arkEntry.strip()
        # Ignorer les ark vides
        if not arkEntry:
            continue
        arkEntry = arkEntry.replace(' ', '\t')
        elems = arkEntry.split('\t')

        # Cas 1 : une plage est précisée
        if len(elems) == 3:
            arkName = elems[0]
            debut = elems[1]
            nb_p = elems[2]
            suffixe = '/f' + debut + 'n' + nb_p
        
        # Cas 2 : on télécharge tout le document
        else:
            arkName = elems[0].strip()
            debut = 1
            nb_p = 0
            suffixe = ''

        if input_format == 'txt':
            url = 'https://gallica.bnf.fr/ark:/12148/{}{}.texteBrut'.format(arkName, suffixe)
            outfile = arkName + '.html'
            path_file = os.path.join(result_path, outfile)
            try:
                resp = requests.get(url, headers=GALLICA_HEADERS, timeout=60)
                resp.raise_for_status()
                with open(path_file, 'wb') as out_file:
                    out_file.write(resp.content)
                res_ok += url + '\n'
            except Exception as exc:
                logger.error(f"Erreur telechargement txt {url}: {exc}")
                res_err += url + '\n'
                continue
        
        elif input_format == 'img':
            # Nb de pages à télécharger : si tout le document, aller chercher l'info dans le service pagination de l'API
            if nb_p == 0:
                url_pagination = "https://gallica.bnf.fr/services/Pagination?ark={}".format(arkName)
                try:
                    resp = requests.get(url_pagination, headers=GALLICA_HEADERS, timeout=30)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, features="xml")
                    nb_p = soup.find('nbVueImages').get_text()
                except Exception as exc:
                    print(exc)

            # Parcours des pages à télécharger
            for i in range(int(debut), int(debut) + int(nb_p)):
                logger.debug(f"Processing page {i} of {arkName}")

                url = "https://gallica.bnf.fr/iiif/ark:/12148/{}/f{}/full/full/0/default.jpg".format(arkName, i)
                outfile = "{}_{:04}.jpg".format(arkName, i)
                path_file = os.path.join(result_path, outfile)
                for attempt in range(3):
                    try:
                        resp = requests.get(url, headers=GALLICA_HEADERS, timeout=60)
                        resp.raise_for_status()
                        with open(path_file, 'wb') as out_file:
                            out_file.write(resp.content)
                        res_ok += url + '\n'
                        break
                    except Exception as exc:
                        logger.error(f"Erreur telechargement image {url} (tentative {attempt+1}/3): {exc}")
                        if attempt < 2:
                            retry_after = int(resp.headers.get('Retry-After', 60))
                            logger.info(f"Attente de {retry_after}s avant nouvelle tentative...")
                            time.sleep(retry_after)
                else:
                    res_err += url + '\n'

        else:
            print("Erreur de paramètre")
            abort(400)
    

    
    with open(os.path.join(result_path, 'download_report.txt'), 'w') as report:
        if res_err != "":
            report.write("Erreur de téléchargement pour : \n {}".format(res_err))
        else:
            res = len(arks_list)
            report.write("{} ARK(s) traité(s) avec succès.\n".format(res))
            report.write(res_ok)

    response = create_zip_and_response(result_path, rand_name)
    download_token = request.form.get('download_token', '')
    if download_token:
        response.set_cookie('download_ready', download_token, max_age=60)
    return response

#---------------- Gutenberg ------------------------
@app.route('/extract_gutenberg', methods=["POST"])
def extract_gutenberg():
    # Vérification que les données du formulaire sont présentes
    if 'files' not in request.form:
        response = {"error": "Book IDs not specified"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    # Extraire les IDs des livres depuis le formulaire
    input_text = request.form.get('files', '').strip()
    if not input_text:
        response = {"error": "No book IDs provided"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    # Créer une liste d'IDs (une ID par ligne)
    book_ids = input_text.splitlines()

    # Générer un nom de répertoire unique pour sauvegarder les fichiers
    rand_name = generate_rand_name('gutenberg_')
    result_path = create_named_directory(rand_name)

    # Format des fichiers à télécharger depuis Gutenberg
    file_format = request.form['file_format']

    for book_id in book_ids:
        try:
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}{file_format}"
            response = requests.get(url)

            if response.status_code == 200:
                # Sauvegarder le contenu du livre dans un fichier
                output_name = f"book_{book_id}{file_format}"
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(response.text)
                print(f"Livre ID {book_id} sauvegardé sous : {output_name}")
            else:
                print(f"Erreur lors du téléchargement du livre ID {book_id}")
        except Exception as e:
            print(f"Une erreur est survenue : {str(e)}")

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#---------------- URL EXTRACTION ------------------------

@app.route('/extract_urls', methods=['POST'])
def extract_urls():
    from newspaper import Article

    if 'files' not in request.form:
        return Response(json.dumps({"error": "URLs not specified"}), status=400, mimetype='application/json')
    
    input_text = request.form.get('files', '').strip()
    if not input_text:
        return Response(json.dumps({"error": "No URLs provided"}), status=400, mimetype='application/json')
    
    urls = input_text.splitlines()
    rand_name = generate_rand_name('extract_urls_')
    result_path = create_named_directory(rand_name)
    
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text_content = article.text[:1000]
            
            filename = os.path.join(result_path, f"{url.replace('https://', '').replace('http://', '').replace('/', '_')}.txt")
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(text_content)
        except Exception as e:
            error_filename = os.path.join(result_path, f"error_{url.replace('https://', '').replace('http://', '').replace('/', '_')}.txt")
            with open(error_filename, 'w', encoding='utf-8') as file:
                file.write(f"Error extracting content: {str(e)}")
    
    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')

#-----------------------------------------------------------------
# Reconnaissance de texte
#-----------------------------------------------------------------

#----------- NUMERISATION TESSERACT -------------------
@app.route('/run_tesseract',  methods=["GET","POST"])
@stream_with_context
def run_tesseract():
    import ocr
    if request.method == 'POST':
        uploaded_files = request.files.getlist("tessfiles")
        model = request.form['tessmodel']
        if 'model2' in request.form:
            model_bis = request.form['model2']
        else:
            model_bis = ''


        up_folder = app.config['UPLOAD_FOLDER']
        rand_name =  generate_rand_name('ocr_')


        text = ocr.tesseract_to_txt(uploaded_files, model, model_bis, rand_name, ROOT_FOLDER, up_folder)
        response = Response(text, mimetype='text/plain',
                            headers={"Content-disposition": "attachment; filename=" + rand_name + '.txt'})

        return response
    return render_template('text_recognition.html', erreur=erreur)


#----------- NUMERISATION KRAKEN -------------------
def list_models(model_dir):
    try:
        return [f for f in os.listdir(model_dir) if f.endswith('.mlmodel') or f.endswith('.json')]
    except Exception as e:
        print(f"Erreur lecture modèles dans {model_dir} : {e}")
        return []

@app.route('/run_kraken', methods=["GET", "POST"])
@stream_with_context
def run_kraken():
    import ocr
    
    erreur = None
    model_base_dir = os.path.join(ROOT_FOLDER, 'kraken_models')
    seg_model_dir = os.path.join(model_base_dir, 'seg_models')
    ocr_model_dir = os.path.join(model_base_dir, 'ocr_models')

    seg_models = list_models(seg_model_dir)
    ocr_models = list_models(ocr_model_dir)

    if request.method == 'POST':
        try:
            uploaded_files = request.files.getlist("krakenfiles")
            seg_model_filename = request.form['segmodel']
            ocr_model_filename = request.form['recomodel']

            rand_name = generate_rand_name('kraken_')
            up_folder = app.config['UPLOAD_FOLDER']

            seg_model_path = os.path.join(seg_model_dir, seg_model_filename)
            ocr_model_path = os.path.join(ocr_model_dir, ocr_model_filename)

            # Concatène tous les textes reconnus dans un seul fichier texte
            all_text = ""
            for file in uploaded_files:
                text = ocr.kraken_to_txt(
                    uploaded_files=[file],
                    model_seg=seg_model_path,
                    model_ocr=ocr_model_path,
                    rand_name=rand_name,
                    ROOT_FOLDER=ROOT_FOLDER,
                    UPLOAD_FOLDER=up_folder
                )
                all_text += text.strip() + "\n\n"

            return Response(all_text, mimetype='text/plain',
                            headers={"Content-Disposition": f"attachment; filename={rand_name}.txt"})

        except Exception as e:
            erreur = str(e)
            print(f"Erreur dans run_kraken : {erreur}")

    return render_template('handwritten_text_recognition.html', erreur=erreur,
                           seg_models=seg_models, ocr_models=ocr_models)

#-------------- Reconnaissance de discours --------------

# Variable globale pour stocker le modèle après le premier chargement
model_cache = None

def get_model():
    import whisper
    global model_cache
    if model_cache is None:
        model_cache = whisper.load_model("base")  # Chargement différé
    return model_cache

@app.route('/automatic_speech_recognition', methods=['POST'])
def automatic_speech_recognition():
    import subprocess

    if 'files' not in request.files and 'audio_urls' not in request.form and 'video_urls' not in request.form:
        return Response(json.dumps({"error": "No files part or URLs provided"}), status=400, mimetype='application/json')

    audio_urls = request.form.get('audio_urls', '').splitlines()
    video_urls = request.form.get('video_urls', '').splitlines()
    file_type = request.form['file_type']

    rand_name = generate_rand_name("asr_")
    result_path = create_named_directory(rand_name, base_dir=UPLOAD_FOLDER)

    model = get_model()  # Appel différé

    try:
        if file_type == 'audio_urls':
            for audio_url in audio_urls:
                if not audio_url.strip():
                    continue

                url_path = urlparse(audio_url).path
                file_name = os.path.basename(url_path)
                output_path = os.path.join(result_path, file_name)

                try:
                    subprocess.run(["wget", audio_url, "-O", output_path], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERREUR] Échec du téléchargement : {audio_url} — {e}")
                    continue

                if not os.path.isfile(output_path):
                    print(f"[ERREUR] Fichier non trouvé : {output_path}")
                    continue

                result = model.transcribe(output_path)
                output_name = os.path.splitext(file_name)[0] + '_transcription.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(result['text'])

        elif file_type == 'video_urls':
            for i, video_url in enumerate(video_urls):
                if not video_url.strip():
                    continue

                audio_output = os.path.join(result_path, f"video_{i}.m4a")
                try:
                    subprocess.run([
                        "yt-dlp",
                        "-f", "bestaudio",
                        video_url,
                        "-o", audio_output
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERREUR] yt-dlp a échoué pour {video_url} — {e}")
                    continue

                if not os.path.isfile(audio_output):
                    print(f"[ERREUR] Fichier audio manquant : {audio_output}")
                    continue

                result = model.transcribe(audio_output)
                output_name = f"video_{i}_transcription.txt"
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(result['text'])

        if not os.listdir(result_path):
            print(f"[DEBUG] Aucun fichier généré dans {result_path}")
            return Response(json.dumps({"error": "Aucune donnée à archiver"}), status=400, mimetype='application/json')

        response = create_zip_and_response(result_path, rand_name)
        return response

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        if os.path.exists(result_path):
            shutil.rmtree(result_path, ignore_errors=True)
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


#-----------------------------------------------------------------
# Prétraitement
#-----------------------------------------------------------------

#------------- Correction Erreurs ---------------------

from wordfreq import zipf_frequency

def languagetool_check(text, lang):
    """Appelle l'API LanguageTool pour corriger un morceau de texte."""
    url = 'https://api.languagetool.org/v2/check'
    data = {
        'text': text,
        'language': lang,
        'enabledOnly': False,
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()

def protect_proper_names(text):
    """Protéger les mots capitalisés (>5 lettres) pour éviter leur correction."""
    protected = {}
    def replacer(match):
        word = match.group(0)
        key = f"PROTECTEDTOKEN{len(protected)}"
        protected[key] = word
        return key

    pattern = r'\b[A-ZÉÈÀÂÊÎÔÛÄËÏÖÜ][A-Za-zÉÈÀÂÊÎÔÛÄËÏÖÜéèàâêîôûäëïöü]{5,}\b'
    text = re.sub(pattern, replacer, text)
    return text, protected

def restore_proper_names(text, protected):
    for key, word in protected.items():
        text = text.replace(key, word)
    return text

def is_valid_word(word, lang, min_zipf=2.5):
    """Retourne True si le mot est suffisamment fréquent dans la langue."""
    # zipf_frequency retourne -1 pour les mots inconnus
    freq = zipf_frequency(word.lower(), lang)
    return freq >= min_zipf

def highlight_corrections(text, matches, lang, protected=None):
    """
    Applique les corrections avec surlignage **…**, en filtrant les mots rares.
    """
    corrections = []
    for match in matches:
        replacements = match.get('replacements')
        if replacements:
            replacement = replacements[0]['value']
            offset = match['offset']
            length = match['length']
            # vérifier fréquence
            if protected and text[offset:offset+length] in protected.values():
                continue
            if not is_valid_word(replacement, lang):
                continue
            corrections.append((offset, length, replacement))

    corrected_text = text
    for offset, length, replacement in sorted(corrections, key=lambda x: x[0], reverse=True):
        corrected_text = (
            corrected_text[:offset] +
            "**" + replacement + "**" +
            corrected_text[offset + length:]
        )
    return corrected_text

def chunk_text(text, max_size=18000):
    """Découpe le texte en morceaux compatibles avec l’API LanguageTool."""
    for i in range(0, len(text), max_size):
        yield text[i:i+max_size]

@app.route('/autocorrect', methods=["GET", "POST"])
def autocorrect():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    selected_language = request.form.get('selected_language', 'fr')

    rand_name = generate_rand_name('autocorrected_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            clean_output = request.form.get("clean_output") == "yes"
            input_text = f.read().decode('utf-8')
            highlighted_corrected_text = ""

            # Découper en chunks
            for chunk in chunk_text(input_text):
                protected_chunk, protected_map = protect_proper_names(chunk)
                result_json = languagetool_check(protected_chunk, selected_language)
                matches = result_json.get('matches', [])
                corrected_chunk = highlight_corrections(protected_chunk, matches, selected_language, protected_map)
                corrected_chunk = restore_proper_names(corrected_chunk, protected_map)
                highlighted_corrected_text += corrected_chunk

            filename, _ = os.path.splitext(f.filename)
            output_name = filename + '_corrected.txt'

            # Choix du contenu
            if clean_output:
                final_text = highlighted_corrected_text.replace("**", "")
            else:
                final_text = highlighted_corrected_text

            # Écriture
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write(final_text)

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response


#-------------- Nettoyage de texte -------------------------

import re

def remove_excessive_lines(text):
    """
    Removes empty lines and lines with only non-alphanumeric characters.
    """
    # Remove lines with only whitespace or weird characters
    cleaned_lines = []
    for line in text.splitlines():
        # Keep line if it has at least one alphanumeric character
        if re.search(r'\w', line):
            cleaned_lines.append(line.strip())
    # Join with single newline
    return "\n".join(cleaned_lines)

def fix_ocr_linebreaks(text):
    paragraphs = []
    current_para = []

    # dash types
    text = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2212]', '-', text)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip empty or garbage lines
        if not line or not re.search(r'\w', line):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            i += 1
            continue

        # hyphens at end of line
        if line.endswith('-') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Join current line without hyphen + next line (no space)
            current_para.append(line[:-1] + next_line)
            i += 2  # skip next line since already joined
        else:
            current_para.append(line + " ")
            i += 1

    if current_para:
        paragraphs.append("".join(current_para).strip())

    return "\n\n".join(paragraphs)

def clean_double_stars(text):
    while "**" in text:
        text = text.replace("**", "")
    return text

loaded_stopwords = {}

def get_stopwords(language):
    from nltk.corpus import stopwords

    if language not in loaded_stopwords:
        if language == 'english':
            loaded_stopwords[language] = set(stopwords.words('english'))
        elif language == 'french':
            loaded_stopwords[language] = set(stopwords.words('french'))
        elif language == 'spanish':
            loaded_stopwords[language] = set(stopwords.words('spanish'))
        elif language == 'german':
            loaded_stopwords[language] = set(stopwords.words('german'))
        elif language == 'danish':
            loaded_stopwords[language] = set(stopwords.words('danish'))
        elif language == 'finnish':
            loaded_stopwords[language] = set(stopwords.words('finnish'))
        elif language == 'greek':
            loaded_stopwords[language] = set(stopwords.words('greek'))
        elif language == 'italian':
            loaded_stopwords[language] = set(stopwords.words('italian'))
        elif language == 'dutch':
            loaded_stopwords[language] = set(stopwords.words('dutch'))
        elif language == 'portuguese':
            loaded_stopwords[language] = set(stopwords.words('portuguese'))
        elif language == 'russian':
            loaded_stopwords[language] = set(stopwords.words('russian'))
        else:
            return set()
    return loaded_stopwords[language]


def keep_accented_only(tokens):
    clean_tokens = []
    for token in tokens:
        parts = token.replace('\u2019', "'").split("'")
        for part in parts:
            clean = ''.join(char for char in part if char.isalpha() or unicodedata.category(char) == 'Mn')
            if clean:
                clean_tokens.append(clean)
    return clean_tokens

@app.route('/removing_elements', methods=['POST'])
def removing_elements():
    from nltk.tokenize import word_tokenize

    if 'files' not in request.files:
        return Response(json.dumps({"error": "No files part"}), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return Response(json.dumps({"error": "No selected files"}), status=400, mimetype='application/json')

    removing_types = request.form.getlist('removing_type')
    selected_language = request.form.get('selected_language', 'french')
    rand_name = generate_rand_name('removing_')
    result_path = create_named_directory(rand_name)
    for f in files:
        try:
            input_text = f.read().decode('utf-8')

            # Step 1: line-level operations on raw text (order matters)
            if 'clean_double_stars' in removing_types:
                input_text = clean_double_stars(input_text)
            if 'fix_ocr_linebreaks' in removing_types:
                input_text = fix_ocr_linebreaks(input_text)
            if 'remove_excessive_lines' in removing_types:
                input_text = remove_excessive_lines(input_text)

            # Step 2 : tokenisation par ligne
            lines = input_text.splitlines()
            tokens_per_line = [word_tokenize(line) for line in lines]

            # Step 3 : opérations token-level dans l'ordre correct
            processed_lines = []

            for tokens in tokens_per_line:

                # 1. punctuation
                if 'punctuation' in removing_types:
                    tokens = keep_accented_only(tokens)

                # 2. lowercases
                if 'lowercases' in removing_types:
                    tokens = [t.lower() for t in tokens]

                # 3. stopwords
                if 'stopwords' in removing_types:
                    stop_words = get_stopwords(selected_language)
                    tokens = [t for t in tokens if t not in stop_words]

                processed_lines.append(" ".join(tokens))

            # Reconstruction avec conservation des lignes
            processed_text = "\n".join(processed_lines)
            filename, _ = os.path.splitext(f.filename)
            output_name = f"{filename}_{'_'.join(sorted(removing_types))}.txt"
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write(processed_text)

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    download_token = request.form.get('download_token', '')
    if download_token:
        response.set_cookie('download_ready', download_token, max_age=60)
    return response


#-------------- Normalisation de texte -------------------------

loaded_nlp_models = {}

def get_nlp(language):
    import spacy
    if language not in loaded_nlp_models:
        if language == 'english':
            loaded_nlp_models[language] = spacy.load('en_core_web_md')
        elif language == 'french':
            loaded_nlp_models[language] = spacy.load('fr_core_news_md')
        elif language == 'spanish':
            loaded_nlp_models[language] = spacy.load('es_core_news_md')
        elif language == 'german':
            loaded_nlp_models[language] = spacy.load('de_core_news_md')
        elif language == 'italian':
            loaded_nlp_models[language] = spacy.load('it_core_news_md')
        elif language == 'danish':
            loaded_nlp_models[language] = spacy.load("da_core_news_md")
        elif language == 'dutch':
            loaded_nlp_models[language] = spacy.load("nl_core_news_md")
        elif language == 'finnish':
            loaded_nlp_models[language] = spacy.load("fi_core_news_md")
        elif language == 'polish':
            loaded_nlp_models[language] = spacy.load("pl_core_news_md")
        elif language == 'portuguese':
            loaded_nlp_models[language] = spacy.load("pt_core_news_md")
        elif language == 'greek':
            loaded_nlp_models[language] = spacy.load("el_core_news_md")
        elif language == 'russian':
            loaded_nlp_models[language] = spacy.load("ru_core_news_md")
        else:
            return set()
    return loaded_nlp_models[language]

@app.route('/normalize_text', methods=['POST'])
def normalize_text():
    from nltk.tokenize import word_tokenize

    if 'files' not in request.files:
        return Response(json.dumps({"error": "No files part"}), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return Response(json.dumps({"error": "No selected files"}), status=400, mimetype='application/json')

    normalisation_types = request.form.getlist('normalisation_type')
    selected_language = request.form['selected_language']

    rand_name = generate_rand_name('normalized_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            filename, _ = os.path.splitext(f.filename)

            # --- TOKENISATION (plate, comme à l'origine) ---
            tokens = word_tokenize(input_text)

            # --- FILTRES SUR LES TOKENS ---
            def apply_token_filters(token_list):
                # 1) garder uniquement les tokens alphabétiques
                if 'tokens_alpha' in normalisation_types:
                    token_list = [t for t in token_list if t.isalpha()]

                # 2) garder tokens alphanumériques
                if 'tokens_alphanum' in normalisation_types:
                    token_list = [t for t in token_list if any(c.isalnum() for c in t)]

                # 3) garder tokens avec apostrophe interne
                if 'tokens_apostrophe' in normalisation_types:
                    token_list = [t for t in token_list if "'" in t and len(t) > 1]

                return token_list

            tokens = apply_token_filters(tokens)

            # --- LEMMATISATION (plate, sans structure) ---
            need_lemmas = (
                'lemmas' in normalisation_types or
                'lemmas_lower' in normalisation_types
            )

            if need_lemmas:
                nlp = get_nlp(selected_language)
                lemmas = [token.lemma_ for token in nlp(input_text)]
                lemmas = apply_token_filters(lemmas)
            else:
                lemmas = []

            # --- CONSTRUCTION DES SECTIONS ---
            output_sections = []

            # TOKENS
            if 'tokens' in normalisation_types:
                output_sections.append("TOKENS:\n" + ", ".join(tokens))

            # TOKENS LOWER
            if 'tokens_lower' in normalisation_types:
                tokens_lower = [t.lower() for t in tokens]
                output_sections.append("TOKENS (minuscules):\n" + ", ".join(tokens_lower))

            # LEMMAS
            if 'lemmas' in normalisation_types:
                clean_lemmas = [l.replace("\n", " ") for l in lemmas]
                output_sections.append("LEMMES:\n" + ", ".join(clean_lemmas))


            # LEMMAS LOWER
            if 'lemmas_lower' in normalisation_types:    
                lemmas_lower = [l.lower().replace("\n", " ") for l in lemmas]
                output_sections.append("LEMMES (minuscules):\n" + ", ".join(lemmas_lower))

            # Assemblage final
            output_text = "\n\n".join(output_sections)

            output_name = filename + '_' + '_'.join(sorted(normalisation_types)) + '.txt'

            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write(output_text)

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    download_token = request.form.get('download_token', '')
    if download_token:
        response.set_cookie('download_ready', download_token, max_age=60)
    return response


#-------------- Séparation de texte -------------------------
@app.route('/split_text', methods=['POST'])
def split_text():
    from nltk.tokenize import sent_tokenize

    if 'files' not in request.files:
        return Response(json.dumps({"error": "No files part"}), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return Response(json.dumps({"error": "No selected files"}), status=400, mimetype='application/json')

    split_types = request.form.getlist("split_type")

    rand_name = generate_rand_name('splittext_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')

            # Préparation des deux types de découpages
            sentences = [s.strip() for s in sent_tokenize(input_text)]
            f.seek(0)
            lines = [line.decode('utf-8').strip() for line in f.readlines()]

            filename, _ = os.path.splitext(f.filename)

            # Récupération des cases cochées

            parts = []
            output_name_parts = []

            if "sentences" in split_types:
                numbered_sentences = [
                    f"S{i+1} : {s}" for i, s in enumerate(sentences)
                ]
                parts.append("SENTENCES:\n" + "\n".join(numbered_sentences))
                output_name_parts.append("sentences")

            if "lines" in split_types:
                numbered_lines = [
                    f"L{i+1} : {l}" for i, l in enumerate(lines)
                ]
                parts.append("LINES:\n" + "\n".join(numbered_lines))
                output_name_parts.append("lines")

            output_text = "\n\n".join(parts)
            output_name = filename + "_" + "_".join(output_name_parts) + ".txt"

            # Écriture du fichier
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write(output_text)

        finally:
            f.close()

    return create_zip_and_response(result_path, rand_name)


#-----------------------------------------------------------------
# Conversion XML
#-----------------------------------------------------------------

@app.route('/xmlconverter', methods=["GET", "POST"])
@stream_with_context
def xmlconverter():
    if request.method == 'POST':

        if 'file' not in request.files:
            response = {"error": "No files part"}
            return Response(json.dumps(response), status=400, mimetype='application/json')

        files = request.files.getlist('file')
        if not files or all(f.filename == '' for f in files):
            response = {"error": "No selected files"}
            return Response(json.dumps(response), status=400, mimetype='application/json')

        fields = {
            'title': request.form.get('title', ''),
            'title_lang': request.form.get('title_lang', ''),
            'author': request.form.get('author'),
            'respStmt_name': request.form.get('nameresp'),
            'respStmt_resp': request.form.get('resp'),
            'pubStmt': request.form.get('pubStmt', ''),
            'sourceDesc': request.form.get('sourceDesc', ''),
            'revisionDesc_change': request.form.get('change', ''),
            'change_who': request.form.get('who', ''),
            'change_when': request.form.get('when', ''),
            'licence': request.form.get('licence', ''),
            'divtype': request.form.get('divtype', ''),
            'creation': request.form.get('creation', ''),
            'lang': request.form.get('lang', ''),
            'projet_p': request.form.get('projet_p', ''),
            'edit_correction_p': request.form.get('edit_correction_p', ''),
            'edit_hyphen_p': request.form.get('edit_hyphen_p', ''),
            'publication_date' : request.form.get('publication_date', '')
        }

        rand_name = generate_rand_name("tei_")
        result_path = create_named_directory(rand_name)

        try:
            for f in files:
                filename = secure_filename(f.filename)
                if filename == '':
                    continue

                # Sauvegarder TXT temporairement
                path_to_file = os.path.join(result_path, filename)
                f.save(path_to_file)

                # Vérifier UTF-8 lisibilité
                try:
                    with open(path_to_file, "r", encoding="utf-8") as file_check:
                        _ = file_check.read(1024)
                except UnicodeDecodeError:
                    return Response(json.dumps({"error": "Format de fichier incorrect (UTF-8 requis)."}),
                                    status=400, mimetype='application/json')

                # Conversion en XML TEI
                root = txt_to_xml(path_to_file, fields)

                # Supprimer TXT juste après conversion
                os.remove(path_to_file)

                # Nom fichier XML
                output_filename = os.path.splitext(filename)[0] + '.xml'
                xml_path = os.path.join(result_path, output_filename)

                # Sauvegarder XML
                with open(xml_path, 'wb') as out_xml:
                    etree.ElementTree(root).write(out_xml, xml_declaration=True, encoding="utf-8")

            # Créer réponse ZIP avec seulement les XML dans result_path
            response = create_zip_and_response(result_path, rand_name)
            return response

        finally:
            pass  # nettoyage géré dans create_zip_and_response

    return render_template("conversion_xml.html")

# CONVERSION XML-TEI
# Construit un fichier TEI à partir des métadonnées renseignées dans le formulaire.
# Renvoie le chemin du fichier ainsi créé
# Paramètres :
# - filename : emplacement du fichier uploadé par l'utilisateur
# - fields : dictionnaire des champs présents dans le form metadata
import xml.etree.ElementTree as etree


def encode_text(filename, doc_type="undefined"):
    div = etree.Element("div")

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if doc_type == "poem":
        _handle_poem(lines, div)
    elif doc_type == "play":
        _handle_play(lines, div)
    elif doc_type == "book":
        _handle_book(lines, div)
    elif doc_type == "report":
        _handle_report(lines, div)
    elif doc_type == "letter":
        _handle_letter(lines, div)
    else:
        _handle_generic_text(lines, div)

    return div

def _handle_poem(lines, div):
    if not lines:
        return

    # Titre = première ligne
    title = lines[0].strip()
    head_element = etree.Element("head")
    head_element.text = title
    div.append(head_element)

    # Regex pour détecter une ligne du type : par / de / by / from + nom
    author_pattern = re.compile(r'^\s*(par|de|by|from)\s+.+', re.IGNORECASE)

    stanza_start = 1  # par défaut, on commence à la 2e ligne
    if len(lines) > 1 and author_pattern.match(lines[1]):
        author = lines[1].strip()
        byline_element = etree.Element("byline")
        byline_element.text = author
        div.append(byline_element)
        stanza_start = 2  # on commence les strophes à la 3e ligne

    def add_stanza_to_div(stanza, number):
        stanza_element = etree.Element("lg", type="stanza", n=str(number))
        for verse in stanza:
            verse_element = etree.Element("l")
            verse_element.text = verse.strip()
            stanza_element.append(verse_element)
        div.append(stanza_element)

    stanza = []
    stanza_count = 1

    for line in lines[stanza_start:]:
        if not line.strip():
            if stanza:
                add_stanza_to_div(stanza, stanza_count)
                stanza = []
                stanza_count += 1
        else:
            stanza.append(line)

    if stanza:
        add_stanza_to_div(stanza, stanza_count)


def _handle_play(lines, div):
    acte_element = None
    scene_element = None

    ACT_REGEX = re.compile(r"^\s*(acte|act|act\s+\w+).*", re.IGNORECASE)
    SCENE_REGEX = re.compile(r"^\s*(scène|scene|sc\.\s*\w+).*", re.IGNORECASE)
    # Ligne qui contient un seul "mot" alphabétique éventuellement suivi de ':' et rien d'autre
    SPEAKER_REGEX = re.compile(r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ\-]+):?\s*$")  

    current_sp = None
    current_speaker_name = None
    current_paragraph_lines = []

    def flush_current_sp():
        nonlocal current_sp, current_paragraph_lines, scene_element
        if current_sp is not None and current_paragraph_lines:
            p = etree.Element("p")
            p.text = " ".join(line.strip() for line in current_paragraph_lines)
            current_sp.append(p)
            current_paragraph_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Ignorer lignes vides ou que blancs
            continue

        if ACT_REGEX.match(stripped):
            flush_current_sp()
            if scene_element is not None and acte_element is not None:
                acte_element.append(scene_element)
                scene_element = None
            if acte_element is not None:
                div.append(acte_element)

            acte_element = etree.Element("div", type="act")
            head = etree.Element("head")
            head.text = stripped
            acte_element.append(head)

            scene_element = None
            current_sp = None
            current_speaker_name = None
            current_paragraph_lines = []

        elif SCENE_REGEX.match(stripped):
            flush_current_sp()
            if scene_element is not None and acte_element is not None:
                acte_element.append(scene_element)

            scene_element = etree.Element("div", type="scene")
            head = etree.Element("head")
            head.text = stripped
            scene_element.append(head)

            current_sp = None
            current_speaker_name = None
            current_paragraph_lines = []

        elif SPEAKER_REGEX.match(stripped):
            flush_current_sp()

            speaker_name = SPEAKER_REGEX.match(stripped).group(1).strip()
            current_sp = etree.Element("sp")
            speaker_elem = etree.Element("speaker")
            speaker_elem.text = speaker_name
            current_sp.append(speaker_elem)

            if scene_element is None:
                scene_element = etree.Element("div", type="scene")
                head = etree.Element("head")
                head.text = "Scène non nommée"
                scene_element.append(head)

            scene_element.append(current_sp)
            current_speaker_name = speaker_name
            current_paragraph_lines = []

        else:
            if current_sp is None:
                current_sp = etree.Element("sp")
                speaker_elem = etree.Element("speaker")
                speaker_elem.text = "Inconnu"
                current_sp.append(speaker_elem)

                if scene_element is None:
                    scene_element = etree.Element("div", type="scene")
                    head = etree.Element("head")
                    head.text = "Scène non nommée"
                    scene_element.append(head)

                scene_element.append(current_sp)
                current_speaker_name = "Inconnu"
                current_paragraph_lines = []

            current_paragraph_lines.append(stripped)

    flush_current_sp()
    if scene_element is not None and acte_element is not None:
        acte_element.append(scene_element)
    if acte_element is not None:
        div.append(acte_element)

def _handle_book(lines, div):
    PAGE_NUM_ONLY_REGEX = re.compile(r"^\s*(\d+)\s*$")
    CHAPTER_REGEX = re.compile(r"^\s*(chapitre|chapter|ch\.?|chapter)\s+[\w\s\-–—]*$", re.IGNORECASE)

    # Regex pour détecter un numéro de page en début ou fin de ligne, avec éventuellement un header dans la ligne
    PAGE_HEADER_REGEX = re.compile(r"^\s*(\d+)?\s*(.*?)(\d+)?\s*$")

    def looks_like_header(text):
        stripped = text.strip()
        # Simple check : au moins une lettre majuscule, et pas vide
        if not stripped:
            return False
        # Vérifie que le texte contient au moins une majuscule
        return any(c.isupper() for c in stripped)

    current_section = etree.Element("div", type="text")
    div.append(current_section)

    paragraph_lines = []

    for line in lines:
        stripped = line.strip()

        # Ligne avec juste un numéro de page
        if PAGE_NUM_ONLY_REGEX.match(stripped):
            if paragraph_lines:
                _finalize_paragraph(paragraph_lines, current_section)
                paragraph_lines = []

            pb = etree.Element("pb")
            pb.set("n", stripped)
            current_section.append(pb)

            note = etree.Element("note", type="foliation")
            note.text = stripped
            current_section.append(note)
            continue

        # Ligne avec numéro de page au début ou à la fin, et header entre les deux
        m = PAGE_HEADER_REGEX.match(stripped)
        if m:
            start_page, middle_text, end_page = m.groups()
            # Choisir le numéro de page, priorité à début, sinon fin
            page_num = start_page or end_page

            # Si on a un numéro et que le texte ressemble à un header
            if page_num and looks_like_header(middle_text):
                if paragraph_lines:
                    _finalize_paragraph(paragraph_lines, current_section)
                    paragraph_lines = []

                pb = etree.Element("pb")
                pb.set("n", page_num)
                current_section.append(pb)

                note = etree.Element("note", type="foliation")
                note.text = page_num
                current_section.append(note)

                fw = etree.Element("fw", type="header")
                fw.text = middle_text.strip()
                current_section.append(fw)
                continue

        # Détection chapitre
        if CHAPTER_REGEX.match(stripped):
            if paragraph_lines:
                _finalize_paragraph(paragraph_lines, current_section)
                paragraph_lines = []

            current_section = etree.Element("div", type="chapter")
            head = etree.Element("head")
            head.text = stripped
            current_section.append(head)
            div.append(current_section)
            continue

        # Ajout au paragraphe courant
        if stripped:
            paragraph_lines.append(stripped)
            if stripped[-1] in ".!?…":
                _finalize_paragraph(paragraph_lines, current_section)
                paragraph_lines = []

    # Fin dernier paragraphe
    if paragraph_lines:
        _finalize_paragraph(paragraph_lines, current_section)


def _finalize_paragraph(paragraph_lines, parent):
    if paragraph_lines:
        # Fusionner les mots coupés par un tiret suivi d'un saut de ligne
        merged_lines = []
        skip_next = False
        for i, line in enumerate(paragraph_lines):
            if skip_next:
                skip_next = False
                continue

            if line.endswith("-") and i + 1 < len(paragraph_lines):
                # Retirer le tiret et concaténer sans espace avec la ligne suivante
                merged_word = line[:-1] + paragraph_lines[i + 1].lstrip()
                merged_lines.append(merged_word)
                skip_next = True
            else:
                merged_lines.append(line)

        # Joindre les lignes corrigées avec espace
        p = etree.Element("p")
        p.text = ' '.join(merged_lines)
        parent.append(p)

def _handle_report(lines, div):
    # Chiffres romains validés (de 1 à 3999) + ponctuation obligatoire
    ROMAN_NUMERAL_REGEX = re.compile(
        r"^\s*(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))[\.\)\-:]\s+")
    SUBSECTION_REGEX = re.compile(r"^\s*([0-9]|[a-z])\s*[\)\-]\s+(.*)",
        re.IGNORECASE)
    PAGE_NUM_REGEX = re.compile(r"^-+\s*([1-9]\d*)\s*-+$")

    def looks_like_title(line):
        stripped = line.strip()
        return stripped.isupper() and len(stripped) > 0

    current_section = etree.Element("div", type="section")
    div.append(current_section)
    current_subsection = None
    paragraph_lines = []

    def _finalize_paragraphs(target):
        nonlocal paragraph_lines
        if paragraph_lines:
            merged_lines = []
            skip_next = False
            for i, line in enumerate(paragraph_lines):
                if skip_next:
                    skip_next = False
                    continue
                # Fusionner mots coupés par tiret + espace
                if line.endswith("-") and i + 1 < len(paragraph_lines):
                    merged_word = line[:-1] + paragraph_lines[i + 1].lstrip()
                    merged_lines.append(merged_word)
                    skip_next = True
                else:
                    merged_lines.append(line)
            p = etree.Element("p")
            p.text = " ".join(merged_lines)
            target.append(p)
            paragraph_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Numéro de page encadré par tirets
        page_match = PAGE_NUM_REGEX.match(stripped)
        if page_match:
            _finalize_paragraphs(current_subsection if current_subsection is not None else current_section)

            page_num = page_match.group(1)
            pb = etree.Element("pb")
            pb.set("n", page_num)
            div.append(pb)

            note = etree.Element("note", type="foliation")
            note.text = page_num
            div.append(note)
            continue

        # Titre tout en MAJ -> nouvelle section
        if looks_like_title(stripped):
            _finalize_paragraphs(current_subsection if current_subsection is not None else current_section)

            current_section = etree.Element("div", type="section")
            div.append(current_section)
            current_subsection = None

            head = etree.Element("head")
            head.text = stripped
            current_section.append(head)
            continue

        # Section chiffre romain + ponctuation obligatoire
        if ROMAN_NUMERAL_REGEX.match(stripped):
            _finalize_paragraphs(current_subsection if current_subsection is not None else current_section)

            current_section = etree.Element("div", type="section")
            div.append(current_section)
            current_subsection = None

            head = etree.Element("head")
            head.text = stripped
            current_section.append(head)
            continue

        # Sous-section chiffre + ponctuation + texte
        sub_match = SUBSECTION_REGEX.match(stripped)
        if sub_match:
            _finalize_paragraphs(current_subsection if current_subsection is not None else current_section)

            number = sub_match.group(1)
            text = sub_match.group(2)

            current_subsection = etree.Element("div", type="subsection")
            current_section.append(current_subsection)

            head = etree.Element("head")
            head.text = f"{number}. {text}"
            current_subsection.append(head)
            continue

        # Ajout aux paragraphes
        paragraph_lines.append(stripped)
        if stripped[-1] in ".!?…":
            _finalize_paragraphs(current_subsection if current_subsection is not None else current_section)

    _finalize_paragraphs(current_subsection if current_subsection is not None else current_section)

def _handle_letter(lines, div):
    salutation_done = False
    closing_started = False
    postscript_started = False
    paragraph_lines = []

    def looks_like_title(line):
        stripped = line.strip()
        return stripped.isupper() and len(stripped) > 0

    def looks_like_date(line):
        return bool(DATE_LINE_REGEX.match(line.strip()))

    def _finalize_paragraphs(target):
        nonlocal paragraph_lines
        if paragraph_lines:
            merged_lines = []
            skip_next = False
            for i, line in enumerate(paragraph_lines):
                if skip_next:
                    skip_next = False
                    continue
                if line.endswith("-") and i + 1 < len(paragraph_lines):
                    merged_word = line[:-1] + paragraph_lines[i + 1].lstrip()
                    merged_lines.append(merged_word)
                    skip_next = True
                else:
                    merged_lines.append(line)

            paragraph_text = " ".join(merged_lines).strip()

            # Si le paragraphe contient uniquement une date (pas de ponctuation, pas de mot de liaison)
            if looks_like_date(paragraph_text):
                date_el = etree.Element("date")
                date_el.text = paragraph_text
                target.append(date_el)
            else:
                p = etree.Element("p")
                p.text = paragraph_text
                target.append(p)

            paragraph_lines = []

    current_block = div

    PAGE_NUM_REGEX = re.compile(r"-\s*([0-9]+)\s*-")
    DATE_LINE_REGEX = re.compile(
        r'^[A-Za-zÀ-ÖØ-öø-ÿ-]+(( |-)[A-Za-zÀ-ÖØ-öø-ÿ-]+)?, (le |on )?[0-9]{1,2}(er|e|)?[ ]?(January|February|March|April|May|June|July|August|September|October|November|December|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre) [12][0-9]{3}\.?\s*$',
        re.IGNORECASE
    )
    SALUTATION_REGEX = re.compile(r'^(Dear|My dear|Mon cher|Ma chère)\b')
    SIGNATURE_REGEX = re.compile(r'^(Yours|Sincerely|Faithfully|Votre|Affectueusement)')
    POSTSCRIPT_REGEX = re.compile(r"^P\s*\.?\s*S(?:\.|,)?[-:]?")

    postscript_element = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Page number: "- 4 -"
        if PAGE_NUM_REGEX.match(stripped):
            _finalize_paragraphs(current_block)
            page_num = PAGE_NUM_REGEX.match(stripped).group(1)
            pb = etree.Element("pb", n=page_num)
            div.append(pb)

            note = etree.Element("note", type="foliation")
            note.text = stripped
            div.append(note)
            continue

        # Title in uppercase
        if looks_like_title(stripped):
            _finalize_paragraphs(current_block)
            head = etree.Element("head")
            head.text = stripped
            div.append(head)
            continue

        # Dateline (dates seules, sans contexte)
        if looks_like_date(stripped):
            _finalize_paragraphs(current_block)
            dateline = etree.Element("dateline")
            dateline.text = stripped
            div.append(dateline)
            continue

        # Salutation (opener)
        if not salutation_done and SALUTATION_REGEX.match(stripped):
            _finalize_paragraphs(current_block)
            opener = etree.Element("opener")
            p = etree.Element("salute")
            p.text = stripped
            opener.append(p)
            div.append(opener)
            salutation_done = True
            continue

        # Signature (closer)
        if SIGNATURE_REGEX.match(stripped):
            _finalize_paragraphs(current_block)
            closer = etree.Element("closer")
            signed = etree.Element("signed")
            signed.text = stripped
            closer.append(signed)
            div.append(closer)
            closing_started = True
            continue

        # Postscript
        if POSTSCRIPT_REGEX.match(stripped):
            _finalize_paragraphs(current_block)
            postscript_started = True
            postscript_element = etree.Element("postscript")
            label = etree.Element("label")
            label.text = POSTSCRIPT_REGEX.match(stripped).group(0).strip()
            postscript_element.append(label)
            div.append(postscript_element)
            current_block = postscript_element
            content = POSTSCRIPT_REGEX.sub("", stripped, count=1).strip()
            if content:
                paragraph_lines.append(content)
            continue

        # Regular paragraph
        paragraph_lines.append(stripped)
        if stripped.endswith((".", "!", "?", "…")):
            _finalize_paragraphs(current_block)

    # Final flush
    _finalize_paragraphs(current_block)


def _handle_generic_text(lines, parent):
    text_div = etree.Element("div", type="text")
    paragraph_lines = []

    def _finalize_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            merged = " ".join(paragraph_lines).strip()
            if merged:
                p = etree.Element("p")
                p.text = merged
                text_div.append(p)
            paragraph_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _finalize_paragraph()
            continue

        paragraph_lines.append(stripped)

        # Si la ligne finit par une ponctuation forte, on clôt le paragraphe
        if stripped.endswith((".", "!", "?", "…")):
            _finalize_paragraph()

    _finalize_paragraph()  # Dernier flush
    parent.append(text_div)


def txt_to_xml(filename, fields):
    # Initialise TEI
    root = etree.Element("TEI", {'xmlns': "http://www.tei-c.org/ns/1.0"})

    # TEI header
    teiHeader = etree.Element("teiHeader")
    fileDesc = etree.Element("fileDesc")
    titleStmt = etree.Element("titleStmt")
    editionStmt = etree.Element("editionStmt")
    publicationStmt = etree.Element("publicationStmt")
    sourceDesc = etree.Element("sourceDesc")
    profileDesc = etree.Element("profileDesc")
    encodingDesc = etree.Element("encodingDesc")
    revisionDesc = etree.Element("revisionDesc")

    #- TitleStmt
    #-- Title
    title = etree.Element("title")
    title_lang = fields["title_lang"]
    title.set("{http://www.w3.org/XML/1998/namespace}lang", title_lang)
    title.text = fields['title']
    titleStmt.append(title)

    #-- Author
    if fields['author']:
        author = etree.Element("author")
        author.text = fields['author']
        titleStmt.append(author)

    #- EditionStmt
    #-- respStmt
    if fields['respStmt_resp']:
        respStmt = etree.Element("respStmt")
        resp = etree.Element("resp")
        resp.text = fields['respStmt_resp']
        respStmt.append(resp)

        if fields['respStmt_name']:
            name = etree.Element("name")
            name.text = fields['respStmt_name']
            respStmt.append(name)

        titleStmt.append(respStmt)

    #-- Automatic encoding info
    respStmt_auto = etree.Element("respStmt")
    resp_auto = etree.Element("resp")
    resp_auto.text = "Encoded by"
    name_auto = etree.Element("name")
    name_auto.text = "Pandore Toolbox"
    respStmt_auto.append(resp_auto)
    respStmt_auto.append(name_auto)
    titleStmt.append(respStmt_auto)

    #- PublicationStmt
    publishers_list = fields['pubStmt'].split('\n') # Get publishers list
    publishers_list = list(map(str.strip, publishers_list)) # remove trailing characters
    publishers_list = [x for x in publishers_list if x] # remove empty strings
    for pub in publishers_list:
        publisher = etree.Element("publisher")
        publisher.text = pub
        publicationStmt.append(publisher)

    licence = etree.Element("licence")
    availability = etree.Element("availability")
    licence.text = fields["licence"]
    if licence.text == "CC-BY":
        licence.set("target", "https://creativecommons.org/licenses/by/4.0/")
    if licence.text == "CC-BY-SA":
        licence.set("target", "https://creativecommons.org/licenses/by-sa/4.0/")
    if licence.text == "CC-BY-ND":
        licence.set("target", "https://creativecommons.org/licenses/by-nd/4.0/")
    if licence.text == "CC-BY-NC":
        licence.set("target", "https://creativecommons.org/licenses/by-nc/4.0/")
    if licence.text == "CC-BY-ND-NC":
        licence.set("target", "https://creativecommons.org/licenses/by-nc-nd/4.0/")
    if licence.text == "CC-BY-NC-SA":
        licence.set("target", "https://creativecommons.org/licenses/by-nc-sa/4.0/")
    availability.append(licence)
    publicationStmt.append(availability)

    if fields['publication_date']:
        date = etree.Element("date")
        publication_date = fields['publication_date']
        date.set('when-iso', publication_date)
        publicationStmt.append(date)

    #- SourceDesc
    paragraphs = fields['sourceDesc'].split('\n')
    for elem in paragraphs:
        p = etree.Element('p')
        p.text = elem
        sourceDesc.append(p)

    #- ProfileDesc
    creation = etree.Element("creation")
    creation_date = fields["creation"]
    creation.set('when', creation_date)
    profileDesc.append(creation)
    langUsage = etree.Element("langUsage")
    language = etree.Element("language")
    lang = fields["lang"]
    language.set("ident", lang)
    langUsage.append(language)
    profileDesc.append(langUsage)

    #- EncodingDesc
    projectDesc = etree.Element("projectDesc")
    project_p = etree.Element("p")
    project_p.text = fields["projet_p"]
    projectDesc.append(project_p)
    encodingDesc.append(projectDesc)

    editorialDecl = etree.Element("editorialDecl")
    edit_correction = etree.Element("correction")
    edit_hyphen = etree.Element("hyphenation")
    edit_correction_p = etree.Element("p")
    edit_correction_p.text = fields["edit_correction_p"]
    edit_correction.append(edit_correction_p)
    edit_hyphen_p = etree.Element("p")
    edit_hyphen_p.text = fields["edit_hyphen_p"]
    edit_hyphen.append(edit_hyphen_p)
    if edit_hyphen_p.text == "all end-of-line hyphenation has been retained, even though the lineation of the original may not have been":
        edit_hyphen.set("eol", "all")
    if edit_hyphen_p.text == "end-of-line hyphenation has been retained in some cases":
        edit_hyphen.set("eol", "some")
    if edit_hyphen_p.text == "all soft end-of-line hyphenation has been removed: any remaining end-of-line hyphenation should be retained":
        edit_hyphen.set("eol", "hard")
    if edit_hyphen_p.text == "all end-of-line hyphenation has been removed: any remaining hyphenation occurred within the line":
        edit_hyphen.set("eol", "none")
    editorialDecl.append(edit_correction)
    editorialDecl.append(edit_hyphen)
    encodingDesc.append(editorialDecl)

    #- RevisionDesc
    if fields['revisionDesc_change']:
        revisionDesc = etree.Element("revisionDesc")
        change = etree.Element("change")
        change.text = fields['revisionDesc_change']
        who = fields["change_who"]
        change.set("who", who)
        when = fields["change_when"]
        change.set("when-iso", when)
        revisionDesc.append(change)

    # Header
    fileDesc.append(titleStmt)
    fileDesc.append(publicationStmt)
    fileDesc.append(sourceDesc)
    teiHeader.append(fileDesc)
    teiHeader.append(encodingDesc)
    teiHeader.append(profileDesc)
    teiHeader.append(revisionDesc)
    root.append(teiHeader)

    # Text
    text = etree.Element("text")
    body = etree.Element("body")
    text.append(body)
    div = etree.Element("div")
    divtype = fields["divtype"]
    div.set("type", divtype)
    body.append(div)

    # Utilisation de la fonction encode_text
    content_div = encode_text(filename, doc_type=fields["divtype"])
    div.extend(content_div)

    root.append(text)

    return root
    
#-----------------------------------------------------------------
# Annotation automatique
#-----------------------------------------------------------------

#------------- POS ----------------------

@app.route('/pos_tagging', methods=['POST'])
def pos_tagging():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    from collections import Counter

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    selected_language = request.form['selected_language']

    rand_name = generate_rand_name('postagging_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            nlp = get_nlp(selected_language)
            doc = nlp(input_text)
            filename, file_extension = os.path.splitext(f.filename)
            pos_counts = Counter(token.pos_ for token in doc)
            
            output_name = filename + '.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                for token in doc:
                    out.write(f"Token: {token.text} --> POS: {token.pos_}\n")

                out.write("\n--- Comptage des POS ---\n")
                for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
                    out.write(f"{pos}: {count}\n")

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response


#------------------ NER ---------------------------

@app.route('/named_entity_recognition', methods=["POST"])
@stream_with_context
def named_entity_recognition():

    from tei_ner import ner_tei_params
    import spacy

    uploaded_files = request.files.getlist("entityfiles")
    if uploaded_files == []:
            print("Outil REN : aucun fichier fourni")
            abort(400)
    
    # Prépare le dossier résultat
    rand_name =  generate_rand_name('ner_')
    result_path = create_named_directory(rand_name)

    # Paramètres généraux
    input_format = request.form['input_format']
    moteur_REN = request.form['moteur_REN']
    modele_REN = request.form['modele_REN']
    encodage = 'UTF-8' # par défaut si non XML

    for f in uploaded_files:
        filename, file_extension = os.path.splitext(f.filename)
        try:
            contenu = f.read()
            
            #-------------------------------------
            # Case 1 : XML -> NER
            #-------------------------------------
            if input_format in ['xml-ariane', 'xml-tei']:

                output_name = os.path.join(result_path, f.filename)

                xmlnamespace = request.form['xmlnamespace']
                balise_racine = request.form['balise_racine']
                balise_parcours = request.form['balise_parcours']
                encodage = request.form['encodage']

                try:
                    etree.fromstring(contenu)
                except etree.ParseError as err:
                    erreur = "Le fichier XML est invalide.\n{}".format(err)

                # choix du mode
                if input_format == "xml-ariane":
                    mode = "ariane"
                else:
                    mode = "tei"

                root = ner_tei_params(
                    contenu,
                    xmlnamespace,
                    balise_racine,
                    balise_parcours,
                    moteur_REN,
                    modele_REN,
                    mode=mode,
                    encodage=encodage
                )

                root.write(
                    output_name,
                    pretty_print=True,
                    xml_declaration=True,
                    encoding="utf-8"
                )

            #-------------------------------------
            # Case 2 : txt -> Spacy, Flair, Camembert
            #-------------------------------------
            else:
                if moteur_REN == 'spacy' or moteur_REN == 'flair':
                    from txt_ner import txt_ner_params
                    entities = txt_ner_params(contenu, moteur_REN, modele_REN, encodage=encodage)
                    output_name = os.path.join(result_path, filename + ".txt")
                    with open(output_name, 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter="\t")
                        for nth, entity in enumerate(entities, 1):
                            ne_type, start, end, text = entity
                            row = [f"T{nth}", f"{ne_type} {start} {end}", f"{text}"]
                            writer.writerow(row)
                
                elif moteur_REN == 'camembert':
                    from ner_camembert import ner_camembert
                    output_name = os.path.join(result_path, filename + '.csv')
                    ner_camembert(contenu.decode("utf-8"), output_name, modele_REN)

    
        finally: # ensure file is closed
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response
        
#-----------------------------------------------------------------
# Extraction d'informations
#-----------------------------------------------------------------

#--------------- Mots-clés -----------------------

@app.route('/keyword_extraction', methods=['POST'])
@stream_with_context
def keyword_extraction():
    """
    Endpoint for extracting keywords from uploaded text files. 
    - 'default': Standard KeyBERT keyword extraction.
    - 'mmr': Maximal Marginal Relevance (MMR) for diverse keywords.
    - 'mss': Multi-word phrases extraction using MaxSum with n-gramms from 1 to 4 elements dividing it into chunks.
    """
    form = FlaskForm()
    if request.method == 'POST':
        try:
            uploaded_files = request.files.getlist("keywd-extract")
            methods = request.form.getlist('extraction-method')
            
            if not uploaded_files:
                return render_template('tools/keywords_extraction.html', 
                                    form=form, 
                                    res={}, 
                                    error="Please upload at least one file.")

            print(f"Received files: {[f.filename for f in uploaded_files]}")
            print(f"Selected methods: {methods}")

            def chunk_text(text, max_length=25000):
                """Split text into smaller chunks to process. The txt book 'Les Bienveillantes' (2,52MB or ~2 433 258 characters) is devided into 98 chunks with max_length=25000 taking up to 7 minutes to process the data."""
                words = text.split()
                chunks = []
                current_chunk = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= max_length:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                    else:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                return chunks

            # Load models only when needed to save memory
            try:
                import gc

                import torch
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
                print("Loading models...")
                sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
                if torch.cuda.is_available():
                    sentence_model = sentence_model.to('cuda')
                    print("Using CUDA")
                kw_model = KeyBERT(model=sentence_model)
                print("Models loaded successfully")
            except Exception as e:
                print(f"Error loading models: {e}")
                return render_template('tools/keywords_extraction.html',
                                    form=form,
                                    res={},
                                    error=f"Failed to initialize models: {str(e)}")
            
            res = {}
            for f in uploaded_files:
                try:
                    fname = secure_filename(f.filename)
                    fname = os.path.splitext(fname)[0]
                    fname = fname.replace(' ', '_')
                    
                    print(f"Processing file: {fname}")
                    res[fname] = {}
                    
                    text = f.read().decode("utf-8")
                    if not text.strip():
                        print(f"Empty file: {fname}")
                        continue

                    # Clear cache before processing
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Split text into chunks for all methods
                    chunks = chunk_text(text)
                    print(f"Split text into {len(chunks)} chunks")

                    method_results = {
                        'default': [],
                        'mmr': [],
                        'mss': []
                    }

                    for i, chunk in enumerate(chunks):
                        print(f"Processing chunk {i+1}/{len(chunks)}")
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        try:
                            if 'default' in methods:
                                keywords_def = kw_model.extract_keywords(chunk)
                                method_results['default'].extend([{'word': word, 'score': float(score)} 
                                                               for word, score in keywords_def])
                            
                            if 'mmr' in methods:
                                diversity = float(request.form.get('diversity', '7')) / 10
                                keywords_mmr = kw_model.extract_keywords(chunk, use_mmr=True, diversity=diversity)
                                method_results['mmr'].extend([{'word': word, 'score': float(score)} 
                                                           for word, score in keywords_mmr])
                            
                            if 'mss' in methods:
                                keywords_mss = kw_model.extract_keywords(
                                    chunk,
                                    keyphrase_ngram_range=(1, 4),
                                    use_maxsum=True,
                                    nr_candidates=10,
                                    top_n=3
                                )
                                filtered_keywords_mss = []
                                for word, score in keywords_mss:
                                    if len(word.split()) > 1 and word.strip():
                                        filtered_keywords_mss.append({'word': word, 'score': float(score)})
                                method_results['mss'].extend(filtered_keywords_mss)
                                
                        except Exception as e:
                            print(f"Error in chunk {i+1}: {str(e)}")
                            continue
                        
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Process results for each method
                    for method in methods:
                        if method_results[method]:

                            # Deduplicate and sort
                            unique_keywords = {}
                            for item in method_results[method]:
                                word = item['word']
                                score = item['score']
                                if word not in unique_keywords or score > unique_keywords[word]['score']:
                                    unique_keywords[word] = item
                            
                            sorted_keywords = sorted(
                                [(item['word'], item['score']) for item in unique_keywords.values()],
                                key=lambda x: x[1],
                                reverse=True
                            )[:10]  # Keeping only top 10 results
                            
                            res[fname][method] = sorted_keywords
                    
                    print(f"Completed processing {fname}")
                    
                except Exception as e:
                    print(f"Error processing file {fname}: {e}")
                    res[fname] = {'error': str(e)}

            print(f"Final results: {res}")
            return render_template('tools/keywords_extraction.html', 
                                form=form, 
                                res=res)

        except Exception as e:
            print(f"General error: {e}")
            return render_template('tools/keywords_extraction.html', 
                                form=form, 
                                res={}, 
                                error=str(e))
    
    return render_template('tools/keywords_extraction.html', 
                         form=form, 
                         res={})

# Test route to verify template loading
@app.route('/test_template')
def test_template():
    form = FlaskForm()
    return render_template('tools/keywords_extraction.html', 
                         form=form, 
                         res={}, 
                         error=None)


#----------------- Topic Modelling -----------------------------

@app.route('/topic_extraction', methods=["POST"])
@stream_with_context
def topic_extraction():
    import spacy
    form = FlaskForm()
    msg = ""
    res = {}

    
    if request.method == 'POST':
        try:
            uploaded_files = request.files.getlist("topic_model")
            if not uploaded_files:
                return render_template('tools/topic_modelling.html', 
                                    form=form, 
                                    res=res, 
                                    msg="Please upload at least one file.")
                
            # Process single file case
            if len(uploaded_files) == 1:
                text = uploaded_files[0].read().decode("utf-8")
                if len(text) < 4500:
                    return render_template('tools/topic_modelling.html', 
                                        form=form, 
                                        res=res, 
                                        msg="Le texte est trop court, merci de charger un corpus plus grand pour des résultats significatifs. A défaut, vous pouvez utiliser l'outil d'extraction de mot-clés.")

            print("Loading required libraries...")
            from sklearn.decomposition import NMF, LatentDirichletAllocation
            from sklearn.feature_extraction.text import (CountVectorizer,
                                                         TfidfVectorizer)

            # Loading stop words
            stop_words_path = ROOT_FOLDER / os.path.join(app.config['UTILS_FOLDER'], "stop_words_fr.txt")
            print(f"Loading stop words from {stop_words_path}")
            try:
                with open(stop_words_path, 'r', encoding="utf-8") as sw:
                    stop_words_fr = sw.read().splitlines()
            except Exception as e:
                print(f"Error loading stop words: {e}")
                return render_template('tools/topic_modelling.html', 
                                    form=form, 
                                    res=res, 
                                    msg=f"Error loading stop words: {e}")
            
            # Form options
            methods = request.form.getlist('modelling-method')
            lemma_state = request.form.getlist('lemma-opt')

            print(f"Selected methods: {methods}")
            print(f"Lemmatization: {lemma_state}")

            # Loading corpus
            corpus = []
            max_f = 0

            # Single file processing
            if len(uploaded_files) == 1:
                print("Processing single file...")
                sents = sentencizer(text)
                chunks = [x.tolist() for x in np.array_split(sents, 3)]
                total_tokens = set()
                
                for l in chunks:
                    if lemma_state:
                        txt_part = spacy_lemmatizer("\n".join(l))
                    else:
                        txt_part = "\n".join(l)
                    
                    corpus.append(txt_part)
                    total_tokens.update(set(txt_part.split(' ')))
                
                no_topics = 2
                
            # Multiple files processing
            else:
                print(f"Processing {len(uploaded_files)} files...")
                total_tokens = set()
                for f in uploaded_files:
                    text = f.read().decode("utf-8")
                    if lemma_state:
                        text = spacy_lemmatizer(text)
                    
                    corpus.append(text)
                    total_tokens.update(set(text.split(' ')))

                no_topics = min(8, len(uploaded_files))
                
            # Corpus size
            max_f = len(total_tokens)
            print(f"Vocabulary size: {max_f}")

            # Matrix parameters
            no_features = int(max_f - (10 * max_f / 100))
            no_top_words = 5

            print(f"Features: {no_features}, Topics: {no_topics}, Top words: {no_top_words}")

            # Topic extraction
            if 'nmf' in methods:
                print("Running NMF...")
                try:
                    tfidf_vectorizer = TfidfVectorizer(
                        max_df=0.95, 
                        min_df=1, 
                        max_features=no_features, 
                        stop_words=stop_words_fr
                    )
                    tfidf = tfidf_vectorizer.fit_transform(corpus)
                    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

                    nmf = NMF(
                        n_components=no_topics, 
                        random_state=1, 
                        l1_ratio=.5, 
                        init='nndsvda'
                    ).fit(tfidf)
                    
                    res_nmf = display_topics(nmf, tfidf_feature_names, no_top_words)
                    res['nmf'] = res_nmf
                except Exception as e:
                    print(f"NMF error: {e}")
                    msg += f"NMF processing error: {e}\n"

            if 'lda' in methods:
                print("Running LDA...")
                try:
                    tf_vectorizer = CountVectorizer(
                        max_df=0.95, 
                        min_df=1, 
                        max_features=no_features, 
                        stop_words=stop_words_fr
                    )
                    tf = tf_vectorizer.fit_transform(corpus)
                    tf_feature_names = tf_vectorizer.get_feature_names_out()

                    lda = LatentDirichletAllocation(
                        n_components=no_topics, 
                        max_iter=5, 
                        learning_method='online', 
                        learning_offset=50.,
                        random_state=0
                    ).fit(tf)
                    
                    res_lda = display_topics(lda, tf_feature_names, no_top_words)
                    res['lda'] = res_lda
                except Exception as e:
                    print(f"LDA error: {e}")
                    msg += f"LDA processing error: {e}\n"
            
            print(f"Final results: {res}")
            return render_template('tools/topic_modelling.html', 
                                form=form, 
                                res=res, 
                                msg=msg)

        except Exception as e:
            print(f"General error in topic extraction: {e}")
            return render_template('tools/topic_modelling.html', 
                                form=form, 
                                res={}, 
                                msg=f"Error processing request: {e}")

    return render_template('tools/topic_modelling.html', 
                         form=form, 
                         res=res, 
                         msg=msg)


#-------------- Quotation Extraction -------------------------


@app.route('/quotation', methods=['POST'])
def quotation():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    rand_name = generate_rand_name('quotation_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            filename, file_extension = os.path.splitext(f.filename)


            patterns = [
                r'"(.*?)"',  # Double quotes ASCII
                r"'(.*?)'",  # Single quotes ASCII
                r"`(.*?)`",  # Backticks

                r'«\s*(.*?)\s*»',  # Guillemets français
                r'‹\s*(.*?)\s*›',  # Guillemets français (niche)
                r'»\s*(.*?)\s*«',  # Guillemets inversés
                r'›\s*(.*?)\s*‹',  # Guillemets inversés (niche)

                r'„\s*(.*?)\s*“',  # Allemand bas-haut
                r'‚\s*(.*?)\s*‘',  # Allemand bas-haut (variante)
                r',\s*(.*?)\s*‘',  # Allemand bas-haut (virgule + apostrophe)

                r'“\s*(.*?)\s*”',  # Guillemets anglais typographiques
                r'”\s*(.*?)\s*”',  # Variante fermante-fermante (cas réel dans ton exemple)
                r'‘\s*(.*?)\s*’',  # Guillemets simples typographiques

                r'「\s*(.*?)\s*」',  # Guillemets japonais
            ]
            
            quotations = []
            for pattern in patterns:
                quotes = re.findall(pattern, input_text)
                quotations.extend(quotes)


            output_name = filename + '.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write("The quotes of the text are:\n")
                for i, quote in enumerate(quotations, start=1):
                    out.write(f"{i}. {quote}\n")

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#-----------------------------------------------------------------
# Analyses
#-----------------------------------------------------------------

#--------------- Analyse linguistique --------------------------

def find_hapaxes(input_text):
    from collections import Counter

    words = input_text.lower().replace(',', '').replace('?', '').split()
    word_counts = Counter(words)
    hapaxes = [word for word, count in word_counts.items() if count == 1]
    return hapaxes

def generate_ngrams(input_text, n, r):
    from nltk.tokenize import word_tokenize
    from collections import Counter

    tokens = word_tokenize(input_text.lower())
    n_grams = ngrams(tokens, n)
    n_grams_counts = Counter(n_grams)
    most_frequent_ngrams = n_grams_counts.most_common(r)
    return most_frequent_ngrams

def write_to_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as out:
        out.write(content)

def analyze_hapax(filename, result_path, hapaxes_list):
    output_name = filename + '_hapaxes.txt'
    write_to_file(os.path.join(result_path, output_name), "The hapaxes are: " + ", ".join(hapaxes_list))

def analyze_ngrams(filename, result_path, input_text, n, r):
    most_frequent_ngrams = generate_ngrams(input_text, n, r)
    output_name = filename + '_ngrams.txt'
    with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
        for n_gram, count in most_frequent_ngrams:
            out.write(f"{n}-gram: {' '.join(n_gram)} --> Count: {count}\n")

def analyze_dependency(filename, result_path, input_text, nlp):
    import spacy
    from spacy import displacy
    doc = nlp(input_text)
    syntax_info = "\n".join(
        [f"{token.text} ({token.pos_}) <--{token.dep_} ({spacy.explain(token.dep_)})-- {token.head.text} ({token.head.pos_})"
         for token in doc]
    )
    # Write dependency parsing text
    output_name_text = filename + '_syntax.txt'
    write_to_file(os.path.join(result_path, output_name_text), syntax_info)

    # Write SVG visualization
    svg = displacy.render(doc, style='dep', jupyter=False)
    output_name_svg = filename + '_syntax.svg'
    write_to_file(os.path.join(result_path, output_name_svg), svg)

def analyze_combined(filename, result_path, analysis_type, hapaxes_list, detected_languages_str, input_text, n, r, nlp):
    import spacy
    from spacy import displacy
    content = ""
    if 'lang' in analysis_type:
        content += f"Detected languages:\n{detected_languages_str}\n\n"
    if 'hapax' in analysis_type:
        content += "The hapaxes are: " + ", ".join(hapaxes_list) + "\n\n"
    if 'ngrams' in analysis_type:
        most_frequent_ngrams = generate_ngrams(input_text, n, r)
        for n_gram, count in most_frequent_ngrams:
            content += f"{n}-gram: {' '.join(n_gram)} --> Count: {count}\n"
        content += "\n\n"
    if 'dependency' in analysis_type:
        doc = nlp(input_text)
        syntax_info = "\n".join(
            [f"{token.text} ({token.pos_}) <--{token.dep_} ({spacy.explain(token.dep_)})-- {token.head.text} ({token.head.pos_})"
             for token in doc]
        )
        content += syntax_info
        # Write SVG visualization
        svg = displacy.render(doc, style='dep', jupyter=False)
        output_name_svg = filename + '_syntax.svg'
        write_to_file(os.path.join(result_path, output_name_svg), svg)
    
    # Write combined content
    output_name = filename + f'_{"_".join(analysis_type)}.txt'
    write_to_file(os.path.join(result_path, output_name), content)


@app.route('/analyze_linguistic', methods=['POST'])
def analyze_linguistic():
    from langdetect import detect_langs
    import spacy

    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    analysis_types = request.form.getlist('analysis_type')
    selected_language = request.form['selected_language']
    nlp = get_nlp(selected_language)

    n = int(request.form.get('n', 2))
    r = int(request.form.get('r', 5))

    rand_name = generate_rand_name('linguistic_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            hapaxes_list = find_hapaxes(input_text)
            langues_detectees = detect_langs(input_text)
            langues_probabilites = [f"Langue : {lang.lang}, Probabilité : {lang.prob * 100:.2f}%" for lang in langues_detectees]
            detected_languages_str = "\n".join(langues_probabilites)
            filename, file_extension = os.path.splitext(f.filename)

            analyze_combined(filename, result_path, analysis_types, hapaxes_list, detected_languages_str, input_text, n, r, nlp)

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    download_token = request.form.get('download_token', '')
    if download_token:
        response.set_cookie('download_ready', download_token, max_age=60)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#--------------- Analyse statistique --------------------------
@app.route('/analyze_statistic', methods=['POST'])
def analyze_statistic():
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from collections import Counter

    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    analysis_types = request.form.getlist('analysis_type')
    context_window = int(request.form.get('context_window', 2)) 
    target_word = str(request.form.get('target_word'))
    selected_language = request.form.get('selected_language')

    rand_name = generate_rand_name('statistics_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            # Sentence length average
            sentences = sent_tokenize(input_text)
            total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
            total_sentences = len(sentences)
            average_words_per_sentence = total_words / total_sentences
            average_words_per_sentence_rounded = round(average_words_per_sentence, 3)

            # Sentence lengths for visualization 
            sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]

            # Tokens called
            tokens = word_tokenize(input_text)

            # Word frequency
            abs_frequency = Counter(tokens)
            total_tokens = len(tokens)
            rel_frequency = {word: count / total_tokens for word, count in abs_frequency.items()}

            # Word frequency (without stopwords)
            stop_words = get_stopwords(selected_language)
            tokens_no_stop = [t for t in tokens if t.lower() not in stop_words]
            abs_frequency_no_stop = Counter(tokens_no_stop)
            total_tokens_no_stop = len(tokens_no_stop)
            rel_frequency_no_stop = {word: count / total_tokens_no_stop for word, count in abs_frequency_no_stop.items()}


            # Co-occurrences
            #stop_words = set(stopwords.words('english'))
            context_pairs = [(target_word, tokens[i + j].lower()) for i, word in enumerate(tokens)
                             for j in range(-context_window, context_window + 1)
                             if i + j >= 0 and i + j < len(tokens) and j != 0
                             if word.lower() == target_word.lower()]
            co_occurrences = FreqDist(context_pairs)

            filename, file_extension = os.path.splitext(f.filename)

            if 'sla' in analysis_types:
                fig, ax = plt.subplots()
                ax.bar(range(1, total_sentences + 1), sentence_lengths, color='blue', alpha=0.7)
                ax.axhline(average_words_per_sentence, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlabel(f'Sentence Number (Total: {str(total_sentences)})')
                ax.set_ylabel(f'Number of Words  (Total: {str(total_words)})')
                ax.set_title('Number of Words per Sentence')
                ax.legend([f'Average Words per Sentence\n({str(average_words_per_sentence_rounded)})', 'Words per Sentence'])
                vis_path = os.path.join(result_path, filename + '_sentence_lengths.png')
                plt.savefig(vis_path, format='png')
                plt.close()

            if 'wf' in analysis_types:
                output_name = filename + '_wordsfrequency.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Absolute frequency of words: " + str(abs_frequency) + "\n\nRelative frequency of words: " + str(rel_frequency) + "\n\nTotal number of words:" + str(total_tokens))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(os.path.join(result_path, filename + '_wordcloud.png'), format='png')
                plt.close()

            if 'wf_stopwords' in analysis_types:
                output_name = filename + '_wordsfrequency_stopwords.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Absolute frequency of words: " + str(abs_frequency_no_stop) + "\n\nRelative frequency of words: " + str(rel_frequency_no_stop) + "\n\nTotal number of words:" + str(total_tokens_no_stop))

                # Wordcloud sans stopwords
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens_no_stop))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(os.path.join(result_path, filename + '_wordcloud.png'), format='png')
                plt.close()

            if 'coocc' in analysis_types:
                output_name = filename + '_cooccurrences.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    for pair, count in co_occurrences.items():
                        out.write(f"Co-occurrence of '{pair[0]}' & '{pair[1]}' --> {count}\n")

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    download_token = request.form.get('download_token', '')
    if download_token:
        response.set_cookie('download_ready', download_token, max_age=60)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#--------------- Analyse de texte --------------------------

# Cache des pipelines
pipeline_cache = {}

def get_pipeline(task, model, **kwargs):
    key = f"{task}_{model}"
    if key not in pipeline_cache:
        from transformers import pipeline
        pipeline_cache[key] = pipeline(task, model=model, **kwargs)
    return pipeline_cache[key]


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    import textstat
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    if 'input_text' not in request.form:
        response = {"error": "No text part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    # ---------------------------------------------------------
    # Récupération et nettoyage de input_text
    # ---------------------------------------------------------
    raw_text = request.form.get('input_text', '')

    # splitlines() = segmentation par lignes du textarea
    input_text = raw_text.splitlines()

    # On retire les lignes vides
    input_text = [line.strip() for line in input_text if line.strip()]

    analysis_type = request.form.getlist('analysis_type')
    emotion_type = request.form['emotion_type']


    rand_name = generate_rand_name('textanalysis_')
    result_path = create_named_directory(rand_name)

    # ---------------------------------------------------------
    # Normalisation de input_text
    # ---------------------------------------------------------
    if isinstance(input_text, str):
        input_text = [input_text]

    # ---------------------------------------------------------
    # Chargement des modèles UNE SEULE FOIS
    # ---------------------------------------------------------
    classifier_subjectivity = get_pipeline(
        "text-classification",
        model="GroNLP/mdebertav3-subjectivity-multilingual"
    )

    classifier_sentiment = get_pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    classifier_emotion_1 = get_pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True,
        top_k=None
    )

    classifier_emotion_2 = get_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        return_all_scores=True,
        top_k=None
    )

    # Construction dynamique du nom de fichier
    task_suffixes = []

    if "subjectivity_detection" in analysis_type:
        task_suffixes.append("subjectivity")

    if "sentiment_analysis" in analysis_type:
        task_suffixes.append("sentiment")

    if "emotion_analysis" in analysis_type:
        if emotion_type == "analyse1":
            task_suffixes.append("emotion1")
        elif emotion_type == "analyse2":
            task_suffixes.append("emotion2")

    if "readibility_scoring" in analysis_type:
        task_suffixes.append("readability")

    def safe_filename(text, max_length=20):
        text = re.sub(r'[\\/*?:"<>|]', '', text)   # caractères interdits
        text = re.sub(r'\W+', '_', text)           # remplace tout par _
        text = text.strip('_')                     # nettoie les bords
        return text[:max_length]                   # limite la longueur

    def truncate(text, max_len=80):
        return text if len(text) <= max_len else text[:max_len] + "..."

    # Création du nom final
    suffix = "_".join(task_suffixes)
    output_name = f"analysis_{suffix}.txt"
    output_path = os.path.join(result_path, output_name)

    with open(output_path, "w", encoding="utf-8") as out:

        # ---------------------------------------------------------
        # SUBJECTIVITY DETECTION
        # ---------------------------------------------------------
        if "subjectivity_detection" in analysis_type:
            out.write("===== SUBJECTIVITY DETECTION =====\n\n")
            for text in input_text:
                result = classifier_subjectivity(text)[0]
                label = "objective" if result["label"] == "LABEL_0" else "subjective"
                out.write(f"[{truncate(text)}] → {label} (Score: {result['score']:.2f})\n\n")

        # ---------------------------------------------------------
        # SENTIMENT ANALYSIS
        # ---------------------------------------------------------
        if "sentiment_analysis" in analysis_type:
            out.write("===== SENTIMENT ANALYSIS =====\n\n")
            for text in input_text:
                results = classifier_sentiment(text)
                star_rating = int(results[0]["label"].split()[0])
                sentiment = (
                    "negative" if star_rating in [1, 2]
                    else "neutral" if star_rating == 3
                    else "positive"
                )
                out.write(
                    f"Sentence: {truncate(text)}\n"
                    f"Star Rating: {star_rating}\n"
                    f"Sentiment: {sentiment}\n"
                    f"Score: {results[0]['score']:.2f}\n\n"
                )

        # ---------------------------------------------------------
        # EMOTION ANALYSIS — ANALYSE 1
        # ---------------------------------------------------------
        if "emotion_analysis" in analysis_type and emotion_type == "analyse1":
            out.write("===== EMOTION ANALYSIS (Analyse 1) =====\n\n")

            text_base = "emotion_viz1_"

            for i, text in enumerate(input_text):

                # Analyse
                vis = classifier_emotion_1(text)
                emotions = [e["label"] for e in vis[0]]
                scores = [e["score"] for e in vis[0]]

                # Écriture dans le fichier texte
                out.write(f"Emotions for [{truncate(text)}]:\n")
                for emo, score in zip(emotions, scores):
                    out.write(f"{emo}: {score:.4f}\n")
                out.write("\n")

                # Visualisation Sankey
                source = [0] * len(emotions)
                target = list(range(1, len(emotions) + 1))
                value = scores

                node_colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
                ]
                node_colors = (node_colors * ((len(emotions) // len(node_colors)) + 1))[:len(emotions) + 1]

                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=55,
                        thickness=55,
                        line=dict(color="black", width=0.5),
                        label=["Input Text"] + emotions,
                        color=node_colors
                    ),
                    link=dict(source=source, target=target, value=value)
                )])

                fig.update_layout(title_text="Emotion Classification", font_size=15)

                # Nom du fichier basé sur la ligne soumise
                clean_name = safe_filename(text)
                vis_filename = f"{text_base}{clean_name}.png"
                vis_path = os.path.join(result_path, vis_filename)

                fig.write_image(vis_path, format="png")

                # Résumé dans le fichier texte
                out.write(f"- Visualisation générée : {vis_filename}\n\n")

        # ---------------------------------------------------------
        # EMOTION ANALYSIS — ANALYSE 2 (résumé dans le fichier)
        # ---------------------------------------------------------
        if "emotion_analysis" in analysis_type and emotion_type == "analyse2":
            out.write("===== EMOTION ANALYSIS (Analyse 2) =====\n\n")
            out.write("Visualisations générées pour chaque texte :\n\n")

            text_base = "emotion_viz2_"

            for i, text in enumerate(input_text):
                vis = classifier_emotion_2(text)
                labels = [e["label"] for e in vis[0]]
                scores = [e["score"] for e in vis[0]]

                combined = list(zip(labels, scores))
                random.shuffle(combined)
                labels, scores = zip(*combined)

                scores = list(scores)
                scores_closed = scores + [scores[0]]

                angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
                angles_closed = angles + [0]

                fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                ax.fill(angles_closed, scores_closed, color="blue", alpha=0.5)
                ax.plot(angles_closed, scores_closed, color="blue", linewidth=1)
                ax.set_xticks(angles)
                ax.set_xticklabels(labels)
                plt.title("Emotion Classification")

                clean_name = safe_filename(text)
                vis_path = os.path.join(result_path, f"{text_base}{clean_name}.png")
                plt.savefig(vis_path, format="png")
                plt.close()

                # Résumé dans le fichier texte
                out.write(f"- Visualisation générée : {text_base}{clean_name}.png\n")

            out.write("\n")

        # ---------------------------------------------------------
        # READABILITY SCORING
        # ---------------------------------------------------------
        if "readibility_scoring" in analysis_type:
            out.write("===== READABILITY SCORING =====\n\n")
            for text in input_text:
                score = textstat.flesch_reading_ease(text)
                out.write(f"Flesch Reading Ease Score: {score}\n\n")

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#--------------- Comparison --------------------------
import textdistance

def highlight_diffs(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1, text2)
    output1 = []
    output2 = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            output1.append('<span style="background-color: #ffd700;">' + ''.join(text1[i1:i2]) + '</span>')
            output2.append('<span style="background-color: #ffd700;">' + ''.join(text2[j1:j2]) + '</span>')
        elif tag == 'delete':
            output1.append('<span style="background-color: #fbb6c2;">' + ''.join(text1[i1:i2]) + '</span>')
            output2.append('<span></span>')
        elif tag == 'insert':
            output1.append('<span></span>')
            output2.append('<span style="background-color: #d4fcbc;">' + ''.join(text2[j1:j2]) + '</span>')
        elif tag == 'equal':
            output1.append(''.join(text1[i1:i2]))
            output2.append(''.join(text2[j1:j2]))
    return ''.join(output1), ''.join(output2)

def calculate_comparisons(text1, text2):

    # Edit-based distance (Levenshtein...): Counts minimum edits needed (insertions, deletions, substitutions) to change one text into another.
    levenshtein_score = textdistance.levenshtein(text1, text2)
    # Token-based similarity (Jaccard...): Assesses similarity by comparing the sets of words from both texts
    jaccard_score = textdistance.jaccard(text1.split(), text2.split())
    # Sequence-based distance (Longest Common Subsequence): Measures the distance based on the length of the longest common subsequence
    lcs_score = textdistance.lcsstr.distance(text1, text2)
    # Compression-based similarity (Ratcliff-Obershelp...): Uses the size of the compressed texts to estimate their similarity
    ratcliff_obershelp_score = textdistance.ratcliff_obershelp(text1, text2)
    return levenshtein_score, jaccard_score, lcs_score, ratcliff_obershelp_score

def compare_texts(text1, text2, output_file):
    # Character by character comparison
    char_diff1, char_diff2 = highlight_diffs(text1, text2)

    # Calculate comparison metrics
    levenshtein_score, jaccard_score, lcs_score, ratcliff_obershelp_score = calculate_comparisons(text1, text2)

    # HTML result
    html_result = '<table><tr><td><pre>' + char_diff1 + '</pre></td><td><pre>' + char_diff2 + '</pre></td></tr></table>'

    # Write results to the HTML file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f'<br><h4>Comparison Results</h4><br/><ul><li>Levenshtein Distance (Edit-based): {levenshtein_score:.2f}</li><li>Jaccard Index (Token-based): {jaccard_score:.2f}</li><li>LCS Distance (Sequence-based): {lcs_score:.2f}</li><li>Ratcliff-Obershelp Similarity (Compression-based): {ratcliff_obershelp_score:.2f}</li></ul><br><span style="background-color: #ffd700;">Substitution</span><br><span style="background-color: #fbb6c2;">Deletion</span><br><span style="background-color: #d4fcbc;">Insertion</span><br><br>')
        file.write(html_result)

def read_file(file_path):
    with open(file_path, 'rb') as file:
        raw = file.read()
    result = from_bytes(raw).best()
    if result is None:
        return raw.decode('utf-8', errors='replace')
    return str(result)

@app.route('/compare', methods=['POST'])
def compare():

    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    file_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_paths.append(file_path)

    if len(file_paths) >= 2:
        text1 = read_file(file_paths[0])
        text2 = read_file(file_paths[1])
        rand_name = generate_rand_name('comparison_')
        result_path = create_named_directory(rand_name)

        output_file = os.path.join(result_path, 'comparison.html')
        compare_texts(text1, text2, output_file)

        # On lit directement le fichier HTML généré
        with open(output_file, 'rb') as f:
            content = f.read()

        # On supprime le dossier temporaire
        shutil.rmtree(result_path)

        # On renvoie le fichier HTML directement
        return Response(
            content,
            mimetype='text/html',
            headers={
                "Content-Disposition": f"attachment; filename={rand_name}.html"
            }
        )

        return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')
    
    else:
        return jsonify({"error": "Please upload at least two text files"}), 400

#---------------------------------------------------------
# Visualisation
#---------------------------------------------------------

@app.route("/run_renard",  methods=["GET", "POST"])
@stream_with_context
def run_renard():

    """
    try: # For debugging  
        from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
        print("Available parameters for CoOccurrencesGraphExtractor:")
        print(help(CoOccurrencesGraphExtractor))
    except Exception as e:
        print(f"Error importing renard: {str(e)}")
    """

    form = FlaskForm()
    if request.method == 'POST':
        try:
            min_appearances = int(request.form['min_appearances'])
            lang = request.form.get('toollang')

            if request.files['renard_upload'].filename != '':
                f = request.files['renard_upload']
                text = f.read().decode('utf-8')
            else:
                text = request.form['renard_txt_input']

            rand_name = 'renard_graph_' + ''.join(random.choice(string.ascii_lowercase) for _ in range(8)) + '.gexf'
            result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)

            import base64
            from renard.graph_utils import graph_with_names
            from renard.pipeline import Pipeline
            from renard.pipeline.character_unification import \
                GraphRulesCharacterUnifier
            from renard.pipeline.graph_extraction import \
                CoOccurrencesGraphExtractor
            from renard.pipeline.ner import BertNamedEntityRecognizer
            from renard.pipeline.tokenization import NLTKTokenizer
            from renard.plot_utils import plot_nx_graph_reasonably
            import matplotlib.pyplot as plt

            BERT_MODELS = {
                "fra": "compnet-renard/camembert-base-literary-NER",
                "eng": "compnet-renard/bert-base-cased-literary-NER",
            }

            pipeline = Pipeline(
                [
                    NLTKTokenizer(),
                    BertNamedEntityRecognizer(model=BERT_MODELS[lang]),
                    GraphRulesCharacterUnifier(min_appearances=min_appearances),
                    CoOccurrencesGraphExtractor(co_occurrences_dist=35)
                ],
                lang=lang
            )

            out = pipeline(text)

            # Export GEXF graph
            out.export_graph_to_gexf(result_path)

            # Render graph
            G = graph_with_names(out.characters_graph)
            plot_nx_graph_reasonably(G)
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.clf()
            figdata_png = base64.b64encode(img.getvalue()).decode('ascii')

            return render_template('tools/renard.html', form=form, graph=figdata_png, fname=str(rand_name))

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return render_template('tools/renard.html', form=form, error=f"Pipeline error: {str(e)}")

    # GET request fallback
    return render_template('tools/renard.html', form=form, graph="", fname="")


#-----------------------------------------------------------------
# Génération de texte
#-----------------------------------------------------------------

import torch #Not sure if this is still useful

#-----------------------------------------------------------------
# Model caching for better performance
#-----------------------------------------------------------------
cache_model = {}

def retrieve_model(task, model_name):
    """Get a model from cache or load it if not cached"""
    key = f"{task}_{model_name}"
    if key not in cache_model:
        from transformers import pipeline
        cache_model[key] = pipeline(task, model=model_name)
    return cache_model[key]

#-----------------------------------------------------------------
@app.route('/bios_converter', methods=["GET", "POST"])
@stream_with_context
def bios_converter():
    fields = {}
    if request.method == 'POST':
        # Read form parameters
        biosfile_a = request.files['file1']
        biosfile_b = request.files['file2']
        annotator_a = request.form.get('annotator_a')
        annotator_b = request.form.get('annotator_b')
        if not annotator_a:
            annotator_a = 'A'
            annotator_b = 'B'

        # Parse both files to find all tags
        pattern = re.compile("[BIES]-(.*)")
        tagset = set()


        annotations = []
        csv_header = ['token', annotator_a, annotator_b]
        annotation = []
        filename_extensions = ("_a", "_b")
        extension_address = 0
        filenames = []

        # Save files (BytesIO -> StringIO)
        for f in [biosfile_a, biosfile_b]:
            filename, file_extension = os.path.splitext(secure_filename(f.filename))
            filename = filename + filename_extensions[extension_address] + ".tsv"
            path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            extension_address = extension_address + 1
            f.save(path_to_file)
            filenames.append(path_to_file)

            # Parse file to find all tags
            with open(path_to_file, 'r') as f:
                tsv = csv.reader(f, delimiter="\t")
                for row in tsv:
                    m = pattern.match(row[1])
                    if m != None:
                        tagset.add(m.group(1))

        # Map tag to number
        map_tag = {"O" : 0}
        for i, x in enumerate(tagset):
            map_tag[x] = i + 1


        for path_to_file in filenames:
            with open(path_to_file, "r") as fin:
                #print(path_to_file, file=sys.stderr)
                for line in fin:

                    line = line.strip()
                    if not line:
                        continue

                    annotation.append(line.split('\t'))
                annotations.append(annotation)

            os.remove(path_to_file)

        output = []
        for i in range(0, len(annotations[1])):

            line = []
            for y in range(0, len(annotations)):
                if y == 0:
                    # on ajoute le token une seule fois
                    line.append(annotations[y][i][0])

                # ajout du tag
                tag = re.sub('[BIES]-', '', annotations[y][i][1])
                line.append(map_tag[tag])
            output.append(line)

        fout = StringIO()
        csv_writer = csv.writer(fout, delimiter=';')
        csv_writer.writerow(csv_header)
        csv_writer.writerows(output)
        response = Response(fout.getvalue(), mimetype='application/xml',
                            headers={"Content-disposition": "attachment; filename=outfile.tsv"})
        fout.seek(0)
        fout.truncate(0)

        return response

    return render_template("/conversion_xml")


#-----------------------------------------------------------------
# Utils
#-----------------------------------------------------------------
def sentencizer(text):
    # Marqueurs de fin de phrase
    split_regex=[r'\.{1,}',r'\!+',r'\?+']
    delimiter_token='<SPLIT>'

    # Escape acronyms or abbreviations
    regex_abbr = re.compile(r"\b(?:[a-zA-Z]\.){1,}")
    abbr_list = re.findall(regex_abbr, text)
    for i, abbr in enumerate(abbr_list):
        text = text.replace(abbr, '<ABBR'+str(i)+'>')

    # Find sentence delimiters
    for regex in split_regex:
        text = re.sub(regex, lambda x:x.group(0)+""+delimiter_token, text)

    # Restore acronyms
    text = re.sub(r"<ABBR([0-9]+)>", lambda x: abbr_list[int(x.group(1))], text)

    # Phrases
    sentences = [x for x in text.split(delimiter_token) if x !='']

    return sentences

def spacy_lemmatizer(text: str) -> str:
    """
    Lemmatize the input French text using spaCy's 'fr_core_news_md' model.

    Parameters:
        text (str): Raw input text.

    Returns:
        str: Lemmatized text.
    """
    import spacy
    nlp = spacy.load('fr_core_news_md')
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def createRandomDir(prefix, length):
    rand_name =  prefix + ''.join((random.choice(string.ascii_lowercase) for x in range(length)))
    result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
    os.mkdir(result_path)
    return (result_path, rand_name)

def getWikiPage(url):
    from bs4 import BeautifulSoup

    # Renvoie le contenu d'un texte wikisource à partir de son url, -1 en cas d'erreur
    req = urllib.request.Request(url, headers={'User-Agent': 'Pandore-Toolbox/1.0 (Educational; contact@sorbonne-universite.fr)'})
    page = urllib.request.urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    text = soup.findAll("div", attrs={'class': 'prp-pages-output'})
    if len(text) == 0:
        print("This does not appear to be part of the text (no prp-pages-output tag at this location).")
        return -1
    else:
        # Remove end of line inside sentence
        clean_text = re.sub(r"[^\.:!?»[A-Z]]\n", ' ', text[0].text)
        clean_text = clean_text.replace('\\xa0', ' ')
        return clean_text

# Extrait les thématiques calculées avec NMF et LDA
# Clé : id, Valeur : liste de termes
def display_topics(model, feature_names, no_top_words):
    res = {}
    for topic_idx, topic in enumerate(model.components_):
        res[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return res

#-----------------------------------------------------------------
# Chaînes de traitement
#-----------------------------------------------------------------

@app.route("/run_ocr_ner", methods=["POST"])
@stream_with_context
def run_ocr_ner():
    import ocr
    from ocr import tesseract_to_txt
    from txt_ner import txt_ner_params
    import spacy

    # Récupération des fichiers uploadés
    uploaded_files = request.files.getlist("inputfiles")
    if not uploaded_files:
        return Response("Aucun fichier reçu", status=400)

    # Récupération des paramètres OCR et NER
    ocr_model = request.form.get('tessmodel', 'default_model')
    encodage = request.form.get('encodage', 'utf-8')
    moteur_REN = "spaCy"
    modele_REN = request.form.get('modele_REN', 'default_ner_model')

    up_folder = app.config.get('UPLOAD_FOLDER', '/tmp')

    # Génération d’un nom de fichier unique
    rand_name = generate_rand_name('ocr_ner_')

    # Extraction de texte via OCR ou lecture brute
    if ocr_model != "raw_text":
        contenu = tesseract_to_txt(uploaded_files, ocr_model, '', rand_name, ROOT_FOLDER, up_folder)
    else:
        liste_contenus = []
        for uploaded_file in uploaded_files:
            try:
                liste_contenus.append(uploaded_file.read().decode(encodage))
            except UnicodeDecodeError:
                return Response(f"Erreur d'encodage du fichier {uploaded_file.filename}", status=400)
        contenu = "\n\n".join(liste_contenus)

    # Extraction des entités nommées
    entities = txt_ner_params(contenu, moteur_REN, modele_REN, encodage=encodage)

    # Création du fichier de sortie
    output_stream = StringIO()
    output = rand_name + ".txt"
    writer = csv.writer(output_stream, delimiter="\t")

    for nth, entity in enumerate(entities, 1):
        ne_type, start, end, text = entity
        writer.writerow([f"T{nth}", f"{ne_type} {start} {end}", f"{text}"])

    # Retourne le fichier en téléchargement
    return Response(
        output_stream.getvalue(),
        mimetype='text/plain',
        headers={"Content-Disposition": f"attachment; filename={output}"}
    )


#--------------------------
#OCR2MAP
#--------------------------

def to_geoJSON_point(coordinates, name):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [coordinates.longitude, coordinates.latitude]
        },
        "properties": {
            "name": name
        },
    }

#---------------------------------------------------------
#AFFICHAGE MAP des résultats pour plusieurs outils de NER
#---------------------------------------------------------

@app.route("/run_ocr_map_intersection", methods=["GET", "POST"])
def run_ocr_map_intersection():
    import asyncio
    from collections import Counter
    from cluster import freqs2clustering

    import aiohttp

    import ocr
    from ocr import tesseract_to_txt
    from txt_ner import txt_ner_params

    geo_cache_file = "cache_geoloc.json"
    if os.path.exists(geo_cache_file):
        with open(geo_cache_file, "r", encoding="utf-8") as f:
            geo_cache = json.load(f)
    else:
        geo_cache = {}

    uploaded_files = request.files.getlist("inputfiles")
    ocr_model = request.form['tessmodel']
    up_folder = app.config['UPLOAD_FOLDER']
    encodage = request.form['encodage']
    moteur_REN1 = request.form['moteur_REN1']
    modele_REN1 = request.form['modele_REN1']
    moteur_REN2 = request.form['moteur_REN2']
    modele_REN2 = request.form['modele_REN2']
    frequences_1 = Counter()
    frequences_2 = Counter()
    frequences = Counter()
    outil_1 = f"{moteur_REN1}/{modele_REN1}"
    outil_2 = (f"{moteur_REN2}/{modele_REN2}" if moteur_REN2 != "None" else "None")

    rand_name = generate_rand_name('ocr_ner_')

    if ocr_model != "raw_text":
        contenu = tesseract_to_txt(uploaded_files, ocr_model, '', rand_name, ROOT_FOLDER, up_folder)
    else:
        liste_contenus = [uploaded_file.read().decode(encodage) for uploaded_file in uploaded_files]
        contenu = "\n\n".join(liste_contenus)

    entities_1 = txt_ner_params(contenu, moteur_REN1, modele_REN1, encodage=encodage)
    ensemble_mentions_1 = {text for label, start, end, text in entities_1 if label == "LOC"}
    ensemble_positions_1 = {(text, start, end) for label, start, end, text in entities_1 if label == "LOC"}
    ensemble_positions = set(ensemble_positions_1)

    if moteur_REN2 != "None":
        entities_2 = txt_ner_params(contenu, moteur_REN2, modele_REN2, encodage=encodage)
        ensemble_mentions_2 = {text for label, start, end, text in entities_2 if label == "LOC"}
        ensemble_positions_2 = {(text, start, end) for label, start, end, text in entities_2 if label == "LOC"}
        ensemble_positions |= ensemble_positions_2
    else:
        entities_2 = ()
        ensemble_mentions_2 = set()
        ensemble_positions_2 = set()

    ensemble_mentions_commun = ensemble_mentions_1 & ensemble_mentions_2
    ensemble_mentions_1 -= ensemble_mentions_commun
    ensemble_mentions_2 -= ensemble_mentions_commun

    for text, _, _ in ensemble_positions_1:
        frequences_1[text] += 1
    for text, _, _ in ensemble_positions_2:
        frequences_2[text] += 1
    for text, _, _ in ensemble_positions:
        frequences[text] += 1

    text2coord = {}

    async def fetch_geocode(session, semaphore, text):
        cleaned_text = text.strip().lower()
        if len(cleaned_text) < 3:
            return text, None

        if cleaned_text in geo_cache:
            return text, geo_cache[cleaned_text]

        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": cleaned_text, "format": "json", "limit": 1}
        headers = {"User-Agent": "your-app-name/1.0 (your-email@example.com)"}

        async with semaphore:
            try:
                async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            lat = float(data[0]["lat"])
                            lon = float(data[0]["lon"])
                            result = [lat, lon]
                            geo_cache[cleaned_text] = result
                            return text, result
            except Exception as e:
                print(f"Error for {text}: {e}")
        return text, None

    async def geocode_all(texts):
        semaphore = asyncio.Semaphore(5)
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_geocode(session, semaphore, text) for text in texts]
            for future in asyncio.as_completed(tasks):
                text, coords = await future
                if coords:
                    text2coord[text] = coords

    asyncio.run(geocode_all({p[0] for p in ensemble_positions}))

    with open(geo_cache_file, "w", encoding="utf-8") as f:
        json.dump(geo_cache, f, ensure_ascii=False, indent=2)

    clusters_1 = freqs2clustering(frequences_1)
    clusters_2 = freqs2clustering(frequences_2)
    clusters = freqs2clustering(frequences)

    def cumulative_freq(clusters, freqs):
        return {
            centroid: sum(freqs[form] for form in clusters[centroid]["Termes"])
            for centroid in clusters
        }

    frequences_cumul_1 = cumulative_freq(clusters_1, frequences_1)
    frequences_cumul_2 = cumulative_freq(clusters_2, frequences_2)
    frequences_cumul = cumulative_freq(clusters, frequences)

    liste_keys = ["Common", outil_1, outil_2]
    liste_ensemble_mention = [ensemble_mentions_commun, ensemble_mentions_1, ensemble_mentions_2]
    dico_mention_marker = {key: [] for key in liste_keys}

    for key, ensemble in zip(liste_keys, liste_ensemble_mention):
        if key == "Common":
            my_clusters = clusters
            my_frequences = frequences_cumul
        elif key == outil_1:
            my_clusters = clusters_1
            my_frequences = frequences_cumul_1
        elif key == outil_2:
            my_clusters = clusters_2
            my_frequences = frequences_cumul_2
        else:
            raise NotImplementedError(f"Clustering pour {key} non implémenté")

        sous_ensemble = [texte for texte in my_frequences if texte in ensemble]
        for texte in sous_ensemble:
            forms = [[form, text2coord.get(form, [0.0, 0.0])] for form in my_clusters[texte]["Termes"]]
            location = text2coord.get(texte)
            if location:
                dico_mention_marker[key].append((location[0], location[1], texte, my_frequences[texte], forms))

    return dico_mention_marker


@app.route("/nermap_to_csv2", methods=['GET', "POST"])
@stream_with_context
def nermap_to_csv2():

    keys = ["nom", "latitude", "longitude", "outil", "cluster"]
    output_stream = StringIO()
    writer = csv.DictWriter(output_stream, fieldnames=keys, delimiter=",")
    writer.writeheader()

    input_json = json.loads(request.data)
    html = etree.fromstring(input_json["html"])
    base_clusters = input_json["clusters"]
    name2coordinates = {}
    print(base_clusters)
    for root_cluster in base_clusters.values():
        for *_, clusters in root_cluster:
            for txt, coords in clusters:
                name2coordinates[txt] = coords
        for e in root_cluster:
            coords = [e[0], e[1]]
            name2coordinates[e[2]] = coords

    print(name2coordinates)

    for toolnode in list(html):
        for item in list(toolnode):
            tool = item.text.strip()
            for centroid_node in list(list(item)[0]):
                print(centroid_node)
                centroid = etree.tostring(next(centroid_node.iterfind("div")), method="text", encoding=str).strip()
                # centroid = centroid_node.text_content().strip()
                try:
                    data = next(centroid_node.iterfind('ol'))
                except StopIteration:  # cluster with no children
                    data = []
                the_cluster = []
                for cluster_item_node in list(data):
                    try:
                        cluster_item = etree.tostring(cluster_item_node, method="text", encoding=str).strip()
                        the_cluster.append(cluster_item.split(" / ")[0])
                    except Exception:
                        stderr.write("\t\tDid not work")
                nom = centroid  # .split(' / ')[0]
                #  latitude = centroid.split(' / ')[1].split(',')[0],
                #  longitude = centroid.split(' / ')[1].split(',')[1],
                print(nom, nom in name2coordinates)
                try:
                    latitude, longitude = name2coordinates[nom]
                except KeyError:
                    stderr.write(f"Could not find {nom} in coordinates")
                    continue
                writer.writerow(
                    {
                        "nom": nom,
                        "latitude": latitude,
                        "longitude": longitude,
                        "outil": tool,
                        "cluster": ', '.join(the_cluster),
                    }
                )

    response = Response(output_stream.getvalue(), mimetype='text/csv',
                        headers={"Content-disposition": "attachment; filename=export.csv"})
    output_stream.seek(0)
    output_stream.truncate(0)
    return response

@app.route("/get_file/<path:filename>")
def get_file(filename):
    return send_from_directory(ROOT_FOLDER / app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":

    print("Starting Pandore Toolbox...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


#=====================================================================
# Adding this part to try to make the website faster 
#=====================================================================

import mimetypes

# Configure static file serving
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year
app.config['STATIC_FOLDER'] = 'static'

@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        if request.path.startswith('/static/'):
            # Cache static files
            response.cache_control.public = True
            response.cache_control.max_age = 31536000
            response.cache_control.immutable = True
            
            # Add content-type for images
            if request.path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
                mime_type, _ = mimetypes.guess_type(request.path)
                if mime_type:
                    response.headers['Content-Type'] = mime_type
    return response

# Serve static files with custom headers
@app.route('/static/<path:filename>')
def serve_static(filename):
    response = send_from_directory(app.static_folder, filename)
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return response