#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Standard library imports
import collections
import csv
import json
import os
import random
import re
import shutil
import string
import math
from datetime import timedelta
from io import StringIO, BytesIO
from pathlib import Path
import urllib
import urllib.request
from urllib.parse import urlparse
import zipfile
import requests

# Third-party imports
from bs4 import BeautifulSoup
from flask import (
    Flask, abort, request, render_template, render_template_string, 
    url_for, redirect, send_from_directory, Response, 
    stream_with_context, session, send_file, jsonify
)
from flask_babel import Babel, get_locale
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFError, CSRFProtect
from langdetect import detect_langs
from lxml import etree
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import ngrams, FreqDist, sent_tokenize, Text
from collections import Counter
import pandas as pd
import spacy
from spacy import displacy
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from wordcloud import WordCloud
import contextualSpellCheck
import textdistance
import difflib
from transformers import pipeline
import textstat
import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np
import torch
import unicodedata
import whisper
from newspaper import Article

# Local application imports
from forms import ContactForm, SearchForm
import ocr
import sem
from cluster import freqs2clustering


# NLTK
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download error: {e}")


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

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # Limit file upload to 35MB
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

@app.route('/documentation_correction')
def documentation_correction():
    return render_template('documentation/documentation_correction.html')

@app.route('/documentation_workflow')
def documentation_workflow():
    return render_template('documentation/documentation_workflow.html')

@app.route('/documentation_generation')
def documentation_generation():
    return render_template('documentation/documentation_generation.html')

#-----------------------------------------------------------------
# TUTORIELS
#-----------------------------------------------------------------

@app.route('/tutoriel')
def tutoriel():
    return render_template('tutoriel.html')
    
@app.route('/tutoriel_conversion')
def tutoriel_conversion():
    return render_template('tutoriel/tutoriel_conversion.html')

@app.route('/tutoriel_annotation')
def tutoriel_annotation():
    return render_template('tutoriel/tutoriel_annotation.html')

@app.route('/tutoriel_extraction')
def tutoriel_extraction():
    return render_template('tutoriel/tutoriel_extraction.html')

@app.route('/tutoriel_analyses')
def tutoriel_analyses():
    return render_template('tutoriel/tutoriel_analyses.html')

@app.route('/tutoriel_correction')
def tutoriel_correction():
    return render_template('tutoriel/tutoriel_correction.html')

@app.route('/tutoriel_workflow')
def tutoriel_workflow():
    return render_template('tutoriel/tutoriel_workflow.html')

@app.route('/tutoriel_generation')
def tutoriel_generation():
    return render_template('tutoriel/tutoriel_generation.html')
    
#-----------------------------------------------------------------
# TACHES
#-----------------------------------------------------------------

@app.route('/atr_tools')
def atr_tools():
    return render_template('taches/atr_tools.html')

@app.route('/pretraitement')
def pretraitement():
    return render_template('taches/pretraitement.html')

@app.route('/conversion')
def conversion():
    return render_template('taches/conversion.html')

@app.route('/annotation_automatique')
def annotation_automatique():
    return render_template('taches/annotation_automatique.html')

@app.route('/extraction_information')
def extraction_information():
    return render_template('taches/extraction_information.html')

@app.route('/analyses')
def analyses():
    return render_template('taches/analyses.html')

@app.route('/search_tools')
def search_tools():
    return render_template('taches/search_tools.html')

@app.route('/outils_visualisation')
def outils_visualisation():
    return render_template('taches/visualisation.html')

@app.route('/outils_correction')
def outils_correction():
    return render_template('taches/correction.html')

@app.route('/collecter_corpus')
def collecter_corpus():
    return render_template('taches/collecter_corpus.html')

@app.route('/outils_pipeline')
def outils_pipeline():
    return render_template('taches/pipeline.html')

@app.route('/generation_texte')
def generation_texte():
    return render_template('taches/generation_texte.html')

#-----------------------------------------------------------------
# OUTILS
#-----------------------------------------------------------------

@app.route('/numeriser')
def numeriser():
    form = FlaskForm()
    return render_template('outils/numeriser.html', form=form)

@app.route('/speech')
def speech():
    form = FlaskForm()
    return render_template('outils/speech.html', form=form)

@app.route('/nettoyage_texte')
def nettoyage_texte():
    form = FlaskForm()
    return render_template('outils/nettoyage_texte.html', form=form)

@app.route('/text_normalisation')
def text_normalisation():
    form = FlaskForm()
    return render_template('outils/text_normalisation.html', form=form)

@app.route('/separation_texte')
def separation_texte():
    form = FlaskForm()
    return render_template('outils/separation_texte.html', form=form)

@app.route('/conversion_xml')
def conversion_xml():
    form = FlaskForm()
    return render_template('outils/conversion_xml.html', form=form)

@app.route('/entites_nommees')
def entites_nommees():
    form = FlaskForm()
    return render_template('outils/entites_nommees.html', form=form)

@app.route('/etiquetage_morphosyntaxique')
def etiquetage_morphosyntaxique():
    form = FlaskForm()
    err = ""
    return render_template('outils/etiquetage_morphosyntaxique.html', form=form, err=err)

@app.route('/categories_semantiques')
def categories_semantiques():
    return render_template('outils/categories_semantiques.html')

@app.route('/extraction_mots_cles')
def extraction_mots_cles():
    form = FlaskForm()
    return render_template('outils/extraction_mots_cles.html', form=form, res={})

@app.route('/quotation_extraction')
def quotation_extraction():
    form = FlaskForm()
    return render_template('outils/quotation_extraction.html', form=form)

@app.route('/topic_modelling')
def topic_modelling():
    form = FlaskForm()
    return render_template('outils/topic_modelling.html', form=form, res={})

@app.route('/analyse_linguistique')
def analyse_linguistique():
    form = FlaskForm()
    return render_template('outils/analyse_linguistique.html', form=form)

@app.route('/analyse_statistique')
def analyse_statistique():
    form = FlaskForm()
    return render_template('outils/analyse_statistique.html', form=form)

@app.route('/analyse_lexicale')
def analyse_lexicale():
    form = FlaskForm()
    return render_template('outils/analyse_lexicale.html', form=form)

@app.route('/analyse_texte')
def analyse_texte():
    form = FlaskForm()
    return render_template('outils/analyse_texte.html', form=form)

@app.route('/comparison')
def comparison():
    form = FlaskForm()
    return render_template('outils/comparison.html', form=form)

@app.route('/embeddings')
def embeddings():
    form = FlaskForm()
    return render_template('outils/embeddings.html', form=form)

@app.route('/tanagra')
def tanagra():
    return render_template('outils/tanagra.html')

@app.route('/renard')
def renard():
    form = FlaskForm()
    return render_template('outils/renard.html', form=form, graph="", fname="")

@app.route('/extraction_gallica')
def extraction_gallica():
    form = FlaskForm()
    return render_template('outils/extraction_gallica.html', form=form)

@app.route('/extraction_wikisource')
def extraction_wikisource():
    form = FlaskForm()
    return render_template('outils/extraction_wikisource.html', form=form)

@app.route('/extraction_gutenberg')
def extraction_gutenberg():
    form = FlaskForm()
    return render_template('outils/extraction_gutenberg.html', form=form)

@app.route('/extraction_urls')
def extraction_urls():
    form = FlaskForm()
    return render_template('outils/extraction_urls.html', form=form)

@app.route('/correction_erreur')
def correction_erreur():
    form = FlaskForm()
    return render_template('outils/correction_erreur.html', form=form)

@app.route('/normalisation')
def normalisation():
    return render_template('outils/normalisation.html')

@app.route('/ocr_ner')
def ocr_ner():
    form = FlaskForm()
    return render_template('outils/ocr_ner.html', form=form)

@app.route('/ocr_map')
def ocr_map():
    form = FlaskForm()
    return render_template('outils/ocr_map.html', form=form)

@app.route('/text_completion')
def text_completion():
    form = FlaskForm()
    return render_template('outils/text_completion.html', form=form)

@app.route('/qa_and_conversation')
def qa_and_conversation():
    form = FlaskForm()
    return render_template('outils/qa_and_conversation.html', form=form)

@app.route('/translation')
def translation():
    form = FlaskForm()
    return render_template('outils/translation.html', form=form)

@app.route('/adjusting_text_readibility_level')
def adjusting_text_readibility_level():
    form = FlaskForm()
    return render_template('outils/adjusting_text_readibility_level.html', form=form)

@app.route('/resume_automatique')
def resume_automatique():
    return render_template('outils/resume_automatique.html')

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
# Reconnaissance de texte
#-----------------------------------------------------------------

#----------- NUMERISATION TESSERACT -------------------
@app.route('/run_tesseract',  methods=["GET","POST"])
@stream_with_context
def run_tesseract():
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
    return render_template('numeriser.html', erreur=erreur)


#-------------- Reconnaissance de discours --------------

# Variable globale pour stocker le modèle après le premier chargement
model_cache = None

def get_model():
    global model_cache
    if model_cache is None:
        model_cache = whisper.load_model("base")  # Chargement différé
    return model_cache

@app.route('/automatic_speech_recognition', methods=['POST'])
def automatic_speech_recognition():
    if 'files' not in request.files and 'audio_urls' not in request.form and 'video_urls' not in request.form:
        response = {"error": "No files part or URLs provided"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    audio_urls = request.form.get('audio_urls', '').splitlines()
    video_urls = request.form.get('video_urls', '').splitlines()

    file_type = request.form['file_type']

    rand_name = generate_rand_name("asr_")  # Génère le nom
    result_path = create_named_directory(rand_name, base_dir=UPLOAD_FOLDER)


    # Appel différé et conditionnel au modèle
    model = get_model()


    if file_type == 'audio_urls':
        # Process audio URLs
        for audio_url in audio_urls:
            if audio_url.strip():
                url_path = urlparse(audio_url).path
                file_name = os.path.basename(url_path)
                os.system(f"wget {audio_url} -O {result_path}/{file_name}")
                audio_file = f"{result_path}/{file_name}"

                if not os.path.isfile(audio_file):
                    print(f"Erreur : {audio_file} n'est pas un fichier.")
                    continue

                result = model.transcribe(audio_file)
                output_name = os.path.splitext(file_name)[0] + '_transcription.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(result['text'])
    elif file_type == 'video_urls':
        # Process video URLs
        for video_url in video_urls:
            if video_url.strip():
                os.system(f"yt-dlp -f 'bestaudio[ext=m4a]' {video_url} -o '{result_path}/video_{video_urls.index(video_url)}.%(ext)s'")
                audio_file = f"{result_path}/video_{video_urls.index(video_url)}.m4a"

                if not os.path.isfile(audio_file):
                    print(f"Erreur : {audio_file} n'est pas un fichier.")
                    continue

                result = model.transcribe(audio_file)
                output_name = f"video_{video_urls.index(video_url)}_transcription.txt"
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(result['text'])


    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')

#-----------------------------------------------------------------
# Prétraitement
#-----------------------------------------------------------------

#-------------- Nettoyage de texte -------------------------

loaded_stopwords = {}

def get_stopwords(language):
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
        # Vérifier chaque caractère : doit être alphabétique ou un caractère accentué
        if all(char.isalpha() or unicodedata.category(char) == 'Mn' for char in token):
            clean_tokens.append(token)
    return clean_tokens

@app.route('/removing_elements', methods=['POST'])
def removing_elements():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    removing_type = request.form['removing_type']
    selected_language = request.form['selected_language']

    rand_name = generate_rand_name('removing_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            lowercases = [token.lower() for token in tokens]
            removing_punctuation = [token for token in keep_accented_only(tokens)]
            lower_removing_punctuation = [token.lower() for token in keep_accented_only(tokens)]
            stop_words = get_stopwords(selected_language)
            removing_stopwords = [token for token in tokens if token.lower() not in stop_words]
            lower_removing_stopwords = [token.lower() for token in tokens if token.lower() not in stop_words]
            removing_punctuation_and_stopwords = [token for token in keep_accented_only(tokens) if token.lower() not in stop_words]
            lower_removing_punctuation_and_stopwords = [token.lower() for token in keep_accented_only(tokens) if token.lower() not in stop_words]
            filename, file_extension = os.path.splitext(f.filename)

            if removing_type == 'punctuation':
                output_name = filename + '_punctuation.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text without punctuation is :\n"' + " ".join(removing_punctuation) + '"')
            elif removing_type == 'lowercases':
                output_name = filename + '_lowercases.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text in lowercases is :\n"' + " ".join(lowercases) + '"')
            elif removing_type == 'stopwords':
                output_name = filename + '_stopwords.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text without stopwords is :\n"' + " ".join(removing_stopwords) + '"')
            elif removing_type == 'lowercases_punctuation':
                output_name = filename + '_lowercasespunctuation.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text in lowercases and without punctuation is :\n"' + " ".join(lower_removing_punctuation) + '"')
            elif removing_type == 'lowercases_stopwords':
                output_name = filename + '_lowercasesstopwords.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text in lowercases and without stopwords is :\n"' + " ".join(lower_removing_stopwords) + '"')
            elif removing_type == 'punctuation_stopwords':
                output_name = filename + '_punctuationstopwords.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text without punctuation and stopwords is :\n"' + " ".join(removing_punctuation_and_stopwords) + '"')
            elif removing_type == 'lowercases_punctuation_stopwords':
                output_name = filename + '_lowercases_punctuation_stopwords.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The text in lowercases and without stopwords and punctuation is :\n"' + " ".join(lower_removing_punctuation_and_stopwords) + '"')



        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#-------------- Normalisation de texte -------------------------

loaded_nlp_models = {}

def get_nlp(language):
    if language not in loaded_nlp_models:
        if language == 'english':
            loaded_nlp_models[language] = spacy.load('en_core_web_sm')
        elif language == 'french':
            loaded_nlp_models[language] = spacy.load('fr_core_news_sm')
        elif language == 'spanish':
            loaded_nlp_models[language] = spacy.load('es_core_news_sm')
        elif language == 'german':
            loaded_nlp_models[language] = spacy.load('de_core_news_sm')
        elif language == 'italian':
            loaded_nlp_models[language] = spacy.load('it_core_news_sm')
        elif language == 'danish':
            loaded_nlp_models[language] = spacy.load("da_core_news_sm")
        elif language == 'dutch':
            loaded_nlp_models[language] = spacy.load("nl_core_news_sm")
        elif language == 'finnish':
            loaded_nlp_models[language] = spacy.load("fi_core_news_sm")
        elif language == 'polish':
            loaded_nlp_models[language] = spacy.load("pl_core_news_sm")
        elif language == 'portuguese':
            loaded_nlp_models[language] = spacy.load("pt_core_news_sm")
        elif language == 'greek':
            loaded_nlp_models[language] = spacy.load("el_core_news_sm")
        elif language == 'russian':
            loaded_nlp_models[language] = spacy.load("ru_core_news_sm")
        else:
            return set()
    return loaded_nlp_models[language]


@app.route('/normalize_text', methods=['POST'])
def normalize_text():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    normalisation_type = request.form['normalisation_type']
    selected_language = request.form['selected_language']

    rand_name = generate_rand_name('normalized_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            tokens_lower = [token.lower() for token in word_tokenize(input_text)]
            nlp = get_nlp(selected_language)
            lemmas = [token.lemma_ for token in nlp(input_text)]
            lemmas_lower = [token.lemma_.lower() for token in nlp(input_text)]
            filename, file_extension = os.path.splitext(f.filename)

            if normalisation_type == 'tokens':
                output_name = filename + '_tokens.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The tokens of the text are: " + ", ".join(tokens))
            elif normalisation_type == 'tokens_lower':
                output_name = filename + '_tokenslower.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The tokens (in lowercases) of the text are: " + ", ".join(tokens_lower))
            elif normalisation_type == 'lemmas':
                output_name = filename + '_lemmas.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The lemmas of the text are: " + ", ".join(lemmas))
            elif normalisation_type == 'lemmas_lower':
                output_name = filename + '_lemmaslower.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The lemmas (in lowercases) of the text are: " + ", ".join(lemmas_lower))
            elif normalisation_type == 'tokens_lemmas':
                output_name = filename + '_tokenslemmas.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The tokens of the text are: " + ", ".join(tokens))
                    out.write("\n\nThe lemmas of the text are: " + ", ".join(lemmas))
            elif normalisation_type == 'tokens_lemmas_lower':
                output_name = filename + '_tokenslemmaslower.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The tokens (in lowercases) of the text are: " + ", ".join(tokens_lower))
                    out.write("\n\nThe lemmas (in lowercases) of the text are: " + ", ".join(lemmas_lower))


        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')

#-------------- Séparation de texte -------------------------

@app.route('/split_text', methods=['POST'])
def split_text():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    split_type = request.form['split_type']

    rand_name = generate_rand_name('splittext_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            sentences = nltk.sent_tokenize(text)
            splitsentence = [sentence.strip() for sentence in sentences]
            f.seek(0)
            lines = f.readlines()
            splitline = [line.decode('utf-8').strip() for line in lines]
            filename, file_extension = os.path.splitext(f.filename)
            
            if split_type == 'sentences':
                output_name = filename + '_sentences.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The sentences of the text are: " + ",\n".join(splitsentence))
            elif split_type == 'lines':
                output_name = filename + '_lines.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The lines of the text are: " + ",\n".join(splitline))
            elif split_type == 'sentences_lines':
                output_name = filename + '.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The sentences of the text are: " + ",\n".join(splitsentence))
                    out.write("\n\nThe lines of the text are: " + ",\n".join(splitline))
        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#-----------------------------------------------------------------
# Conversion XML
#-----------------------------------------------------------------

@app.route('/xmlconverter', methods=["GET", "POST"])
@stream_with_context
def xmlconverter():
    if request.method == 'POST':
        fields = {}
        fields['title'] = request.form['title']
        fields['title_lang'] = request.form['title_lang'] # required
        fields['author'] = request.form.get('author')
        fields['respStmt_name'] = request.form.get('nameresp')
        fields['respStmt_resp'] = request.form.get('resp')
        fields['pubStmt'] = request.form['pubStmt'] # required
        fields['sourceDesc'] = request.form['sourceDesc'] # required
        fields['revisionDesc_change'] = request.form['change']
        fields['change_who'] = request.form['who']
        fields['change_when'] = request.form['when']
        fields['licence'] = request.form['licence']
        fields['divtype'] = request.form['divtype']
        fields["creation"] = request.form['creation']
        fields["lang"] = request.form['lang']
        fields["projet_p"] = request.form['projet_p']
        fields["edit_correction_p"] = request.form['edit_correction_p']
        fields["edit_hyphen_p"] = request.form['edit_hyphen_p']

        files = request.files.getlist('file')
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for f in files:
                filename = secure_filename(f.filename)
                path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(path_to_file)

                try:
                    with open(path_to_file, "r") as file:
                        for l in file:
                            break

                    # Returning xml string
                    root = txt_to_xml(path_to_file, fields)

                    # Writing in stream
                    output_stream = BytesIO()
                    output_filename = os.path.splitext(filename)[0] + '.xml'
                    etree.ElementTree(root).write(output_stream, xml_declaration=True, encoding="utf-8")
                    output_stream.seek(0)
                    zip_file.writestr(output_filename, output_stream.getvalue())
                    output_stream.truncate(0)

                except UnicodeDecodeError:
                    return 'format de fichier incorrect'

        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='encoded_files.zip')

    return render_template("/conversion_xml")

# CONVERSION XML-TEI
# Construit un fichier TEI à partir des métadonnées renseignées dans le formulaire.
# Renvoie le chemin du fichier ainsi créé
# Paramètres :
# - filename : emplacement du fichier uploadé par l'utilisateur
# - fields : dictionnaire des champs présents dans le form metadata
import xml.etree.ElementTree as etree

def encode_text(filename, is_text_standard=True, is_poem=False, is_play=False, is_book=False):
    div = etree.Element("div")

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if is_poem:
        stanza = []
        for line in lines:
            if line.strip() == "":
                if stanza:
                    stanza_element = etree.Element("lg", type="stanza")
                    for verse in stanza:
                        verse_element = etree.Element("l")
                        verse_element.text = verse.strip()
                        stanza_element.append(verse_element)
                    div.append(stanza_element)
                    stanza = []
            else:
                stanza.append(line)
        
        if stanza:
            stanza_element = etree.Element("lg", type="stanza")
            for verse in stanza:
                verse_element = etree.Element("l")
                verse_element.text = verse.strip()
                stanza_element.append(verse_element)
            div.append(stanza_element)
    elif is_play:
        scene = []
        acte_element = None
        scene_element = None

        for line in lines:
            if re.match(r"Act|Acte", line.strip()):
                if acte_element is not None:
                    div.append(acte_element)
                acte_element = etree.Element("div")
                acte_element.set("type", "act")
                head_element = etree.Element("head")
                head_element.text = line.strip()
                acte_element.append(head_element)
                scene = []

            elif re.match(r"Scène|Scene", line.strip()):
                if scene_element is not None:
                    acte_element.append(scene_element)
                scene_element = etree.Element("div")
                scene_element.set("type", "scene")
                head_element = etree.Element("head")
                head_element.text = line.strip()
                scene_element.append(head_element)
                scene = []

            else:
                scene.append(line)

        if scene_element is not None:
            for dialogue in scene:
                dialogue_element = etree.Element("p")
                dialogue_element.text = dialogue.strip()
                scene_element.append(dialogue_element)
            acte_element.append(scene_element)

        if acte_element is not None:
            div.append(acte_element)

    elif is_book:
        scene = []
        chapter_element = None
        text_element = None

        for line in lines:
            if re.match(r"Chapter|Chapitre", line.strip()):
                if chapter_element is not None:
                    div.append(chapter_element)
                chapter_element = etree.Element("div")
                chapter_element.set("type", "chapter")
                head_element = etree.Element("head")
                head_element.text = line.strip()
                chapter_element.append(head_element)
                scene = []
            else:
                scene.append(line)

        if chapter_element is not None:
            text_element = etree.Element("div")
            text_element.set("type", "text")
            for dialogue in scene:
                dialogue_element = etree.Element("p")
                dialogue_element.text = dialogue.strip()
                text_element.append(dialogue_element)
            chapter_element.append(text_element)
            div.append(chapter_element)


    else:
        file = "".join(lines)
        file = file.replace(".\n", ".[$]")
        ptext = file.split("[$]")
        for line in ptext:
            paragraph = etree.Element("p")
            paragraph.text = line.strip()
            div.append(paragraph)
    
    return div

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
    availability.append(licence)
    publicationStmt.append(availability)

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
    content_div = encode_text(filename, is_text_standard=(fields["divtype"] == "text"), is_poem=(fields["divtype"] == "poem"), is_play=(fields["divtype"] == "play"), is_book=(fields["divtype"] == "book"))
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
            
            output_name = filename + '.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                for token in doc:
                    out.write(f"Token: {token.text} --> POS: {token.pos_}\n")

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#------------------ NER ---------------------------

@app.route('/named_entity_recognition', methods=["POST"])
@stream_with_context
def named_entity_recognition():
    from tei_ner import tei_ner_params
    from lxml import etree

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
            # Case 1 : xml -> Spacy or Flair
            #-------------------------------------
            if input_format == 'xml':
                #print("XML détecté")
                output_name = os.path.join(result_path, f.filename)
                xmlnamespace = request.form['xmlnamespace']
                balise_racine = request.form['balise_racine']
                balise_parcours = request.form['balise_parcours']
                encodage = request.form['encodage']

                try:
                    etree.fromstring(contenu)
                    #print("Le XML est valide")
                except etree.ParseError as err:
                    erreur = "Le fichier XML est invalide. \n {}".format(err)

                root = tei_ner_params(contenu, xmlnamespace, balise_racine, balise_parcours, moteur_REN, modele_REN, encodage=encodage)
                
                root.write(output_name, pretty_print=True, xml_declaration=True, encoding="utf-8")

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

    render_template('entites_nommees.html', erreur=erreur)
        
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
                return render_template('outils/extraction_mots_cles.html', 
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
                import torch
                import gc
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
                return render_template('outils/extraction_mots_cles.html',
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
            return render_template('outils/extraction_mots_cles.html', 
                                form=form, 
                                res=res)

        except Exception as e:
            print(f"General error: {e}")
            return render_template('outils/extraction_mots_cles.html', 
                                form=form, 
                                res={}, 
                                error=str(e))
    
    return render_template('outils/extraction_mots_cles.html', 
                         form=form, 
                         res={})

# Test route to verify template loading
@app.route('/test_template')
def test_template():
    form = FlaskForm()
    return render_template('outils/extraction_mots_cles.html', 
                         form=form, 
                         res={}, 
                         error=None)


#----------------- Topic Modelling -----------------------------

@app.route('/topic_extraction', methods=["POST"])
@stream_with_context
def topic_extraction():
    form = FlaskForm()
    msg = ""
    res = {}

    
    if request.method == 'POST':
        try:
            uploaded_files = request.files.getlist("topic_model")
            if not uploaded_files:
                return render_template('outils/topic_modelling.html', 
                                    form=form, 
                                    res=res, 
                                    msg="Please upload at least one file.")
                
            # Process single file case
            if len(uploaded_files) == 1:
                text = uploaded_files[0].read().decode("utf-8")
                if len(text) < 4500:
                    return render_template('outils/topic_modelling.html', 
                                        form=form, 
                                        res=res, 
                                        msg="Le texte est trop court, merci de charger un corpus plus grand pour des résultats significatifs. A défaut, vous pouvez utiliser l'outil d'extraction de mot-clés.")

            print("Loading required libraries...")
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            from sklearn.decomposition import NMF, LatentDirichletAllocation
            from pathlib import Path
            import numpy as np

            # Loading stop words
            stop_words_path = ROOT_FOLDER / os.path.join(app.config['UTILS_FOLDER'], "stop_words_fr.txt")
            print(f"Loading stop words from {stop_words_path}")
            try:
                with open(stop_words_path, 'r', encoding="utf-8") as sw:
                    stop_words_fr = sw.read().splitlines()
            except Exception as e:
                print(f"Error loading stop words: {e}")
                return render_template('outils/topic_modelling.html', 
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
            return render_template('outils/topic_modelling.html', 
                                form=form, 
                                res=res, 
                                msg=msg)

        except Exception as e:
            print(f"General error in topic extraction: {e}")
            return render_template('outils/topic_modelling.html', 
                                form=form, 
                                res={}, 
                                msg=f"Error processing request: {e}")

    return render_template('outils/topic_modelling.html', 
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
                r'"(.*?)"',           # Double quotes
                r'«\s*(.*?)\s*»'      # French quotes with optional spaces
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
    words = input_text.lower().replace(',', '').replace('?', '').split()
    word_counts = Counter(words)
    hapaxes = [word for word, count in word_counts.items() if count == 1]
    return hapaxes

def generate_ngrams(input_text, n, r):
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

def analyze_dependency(filename, result_path, input_text, nlp_eng):
    doc = nlp_eng(input_text)
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

def analyze_combined(filename, result_path, analysis_type, hapaxes_list, detected_languages_str, input_text, n, r, nlp_eng):
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
        doc = nlp_eng(input_text)
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
    output_name = filename + f'_{analysis_type}.txt'
    write_to_file(os.path.join(result_path, output_name), content)


@app.route('/analyze_linguistic', methods=['POST'])
def analyze_linguistic():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    analysis_type = request.form['analysis_type']

    n = int(request.form.get('n', 2))  # Default n-gram length to 2 if not provided
    r = int(request.form.get('r', 5)) 

    nlp_eng = spacy.load('en_core_web_sm')

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

            if analysis_type == 'hapax':
                analyze_hapax(filename, result_path, hapaxes_list)
            elif analysis_type == 'n_gram':
                analyze_ngrams(filename, result_path, input_text, n, r)
            elif analysis_type == 'dependency':
                analyze_dependency(filename, result_path, input_text, nlp_eng)
            else:
                analyze_combined(filename, result_path, analysis_type, hapaxes_list, detected_languages_str, input_text, n, r, nlp_eng)


        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#--------------- Analyse statistique --------------------------
@app.route('/analyze_statistic', methods=['POST'])
def analyze_statistic():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    analysis_type = request.form['analysis_type']
    context_window = int(request.form.get('context_window', 2)) 
    target_word = str(request.form.get('target_word'))

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

            # Co-occurrences
            stop_words = set(stopwords.words('english'))
            context_pairs = [(target_word, tokens[i + j].lower()) for i, word in enumerate(tokens)
                             for j in range(-context_window, context_window + 1)
                             if i + j >= 0 and i + j < len(tokens) and j != 0
                             if word.lower() == target_word.lower()]
            co_occurrences = FreqDist(context_pairs)

            filename, file_extension = os.path.splitext(f.filename)

            if analysis_type == 'sentence_length_average':
                """output_name = filename + '_length.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Total Words: " + str(total_words) + "\n\nTotal Sentences: " + str(total_sentences) + "\n\nAverage Words per Sentence: " + str(average_words_per_sentence))
                """
                # Generate sentence length visualization
                fig, ax = plt.subplots()
                ax.bar(range(1, total_sentences + 1), sentence_lengths, color='blue', alpha=0.7)
                ax.axhline(average_words_per_sentence, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlabel(f'Sentence Number (Total: {str(total_sentences)})')
                ax.set_ylabel(f'Number of Words  (Total: {str(total_words)})')
                ax.set_title('Number of Words per Sentence')
                ax.legend([f'Average Words per Sentence\n({str(average_words_per_sentence_rounded)})', 'Words per Sentence'])
                
                # Save visualization to a file
                vis_name = filename + '_sentence_lengths.png'
                vis_path = os.path.join(result_path, vis_name)
                plt.savefig(vis_path, format='png')
                plt.close()

            elif analysis_type == 'words_frequency':
                output_name = filename + '_wordsfrequency.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Absolute frequency of words: " + str(abs_frequency) + "\n\nRelative frequency of words: " + str(rel_frequency) + "\n\nTotal number of words:" + str(total_tokens))

                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                # Save word cloud to a file
                wordcloud_name = filename + '_wordcloud.png'
                wordcloud_path = os.path.join(result_path, wordcloud_name)
                plt.savefig(wordcloud_path, format='png')
                plt.close()

            elif analysis_type == 'cooccurrences':
                output_name = filename + '_cooccurrences.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    for pair, count in co_occurrences.items():
                        out.write(f"Co-occurrence of '{pair[0]}' & '{pair[1]}' --> {count}\n")

            elif analysis_type == 'sla_wf':
                output_name = filename + '_sla_wf.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Absolute frequency of words: " + str(abs_frequency) + "\n\nRelative frequency of words: " + str(rel_frequency) + "\n\nTotal number of words:" + str(total_tokens))
                    out.write("\n\nTotal Sentences: " + str(total_sentences) + "\n\nAverage Words per Sentence: " + str(average_words_per_sentence_rounded))
                
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                # Save word cloud to a file
                wordcloud_name = filename + '_wordcloud.png'
                wordcloud_path = os.path.join(result_path, wordcloud_name)
                plt.savefig(wordcloud_path, format='png')
                plt.close()

                # Generate sentence length visualization
                fig, ax = plt.subplots()
                ax.bar(range(1, total_sentences + 1), sentence_lengths, color='blue', alpha=0.7)
                ax.axhline(average_words_per_sentence, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlabel(f'Sentence Number (Total: {str(total_sentences)})')
                ax.set_ylabel(f'Number of Words  (Total: {str(total_words)})')
                ax.set_title('Number of Words per Sentence')
                ax.legend([f'Average Words per Sentence\n({str(average_words_per_sentence_rounded)})', 'Words per Sentence'])
                
                # Save visualization to a file
                vis_name = filename + '_sentence_lengths.png'
                vis_path = os.path.join(result_path, vis_name)
                plt.savefig(vis_path, format='png')
                plt.close()
        
            elif analysis_type == 'sla_coocc':
                output_name = filename + 'sla_coocc.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Total Words: " + str(total_words) + "\n\nTotal Sentences: " + str(total_sentences) + "\n\nAverage Words per Sentence: " + str(average_words_per_sentence) + "\n\n")
                    for pair, count in co_occurrences.items():
                        out.write(f"Co-occurrence of '{pair[0]}' & '{pair[1]}' --> {count}\n")

                # Generate sentence length visualization
                fig, ax = plt.subplots()
                ax.bar(range(1, total_sentences + 1), sentence_lengths, color='blue', alpha=0.7)
                ax.axhline(average_words_per_sentence, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlabel(f'Sentence Number (Total: {str(total_sentences)})')
                ax.set_ylabel(f'Number of Words  (Total: {str(total_words)})')
                ax.set_title('Number of Words per Sentence')
                ax.legend([f'Average Words per Sentence\n({str(average_words_per_sentence_rounded)})', 'Words per Sentence'])
                
                # Save visualization to a file
                vis_name = filename + '_sentence_lengths.png'
                vis_path = os.path.join(result_path, vis_name)
                plt.savefig(vis_path, format='png')
                plt.close()

            elif analysis_type == 'wf_coocc':
                output_name = filename + 'wf_coocc.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Absolute frequency of words: " + str(abs_frequency) + "\n\nRelative frequency of words: " + str(rel_frequency) + "\n\nTotal number of words:" + str(total_tokens) + "\n\n")
                    for pair, count in co_occurrences.items():
                        out.write(f"Co-occurrence of '{pair[0]}' & '{pair[1]}' --> {count}\n")

                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                # Save word cloud to a file
                wordcloud_name = filename + '_wordcloud.png'
                wordcloud_path = os.path.join(result_path, wordcloud_name)
                plt.savefig(wordcloud_path, format='png')
                plt.close()

            elif analysis_type == 'sla_wf_coocc':
                output_name = filename + 'sla_wf_coocc.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Total Words: " + str(total_words) + "\n\nTotal Sentences: " + str(total_sentences) + "\n\nAverage Words per Sentence: " + str(average_words_per_sentence) + "\n\n")
                    out.write("Absolute frequency of words: " + str(abs_frequency) + "\n\nRelative frequency of words: " + str(rel_frequency) + "\n\nTotal number of words:" + str(total_tokens) + "\n\n")
                    for pair, count in co_occurrences.items():
                        out.write(f"Co-occurrence of '{pair[0]}' & '{pair[1]}' --> {count}\n")

                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                
                # Save word cloud to a file
                wordcloud_name = filename + '_wordcloud.png'
                wordcloud_path = os.path.join(result_path, wordcloud_name)
                plt.savefig(wordcloud_path, format='png')
                plt.close()

                # Generate sentence length visualization
                fig, ax = plt.subplots()
                ax.bar(range(1, total_sentences + 1), sentence_lengths, color='blue', alpha=0.7)
                ax.axhline(average_words_per_sentence, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlabel(f'Sentence Number (Total: {str(total_sentences)})')
                ax.set_ylabel(f'Number of Words  (Total: {str(total_words)})')
                ax.set_title('Number of Words per Sentence')
                ax.legend([f'Average Words per Sentence\n({str(average_words_per_sentence_rounded)})', 'Words per Sentence'])
                
                # Save visualization to a file
                vis_name = filename + '_sentence_lengths.png'
                vis_path = os.path.join(result_path, vis_name)
                plt.savefig(vis_path, format='png')
                plt.close()

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')

#--------------- Analyse lexicale --------------------------

@app.route('/analyze_lexicale', methods=['POST'])
def analyze_lexicale():
    if 'files' not in request.files:
        return Response(json.dumps({"error": "No files part"}), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return Response(json.dumps({"error": "No selected files"}), status=400, mimetype='application/json')

    # Get analysis parameters with proper error handling
    analysis_type = request.form.get('analysis_type')
    if not analysis_type:
        return Response(json.dumps({"error": "Analysis type not specified"}), status=400, mimetype='application/json')

    # Initialize parameters with proper error handling
    analyzed_words = []
    word = ''
    words_list = ''

    # Only get parameters needed for the specific analysis type
    if analysis_type == 'lexical_dispersion':
        words_to_analyze = request.form.get('words_to_analyze')
        if not words_to_analyze:
            return Response(json.dumps({"error": "Words to analyze not specified"}), status=400, mimetype='application/json')
        analyzed_words = words_to_analyze.split(';')
    elif analysis_type == 'lexical_relationships':
        word = request.form.get('word')
        if not word:
            return Response(json.dumps({"error": "Word not specified"}), status=400, mimetype='application/json')
    elif analysis_type == 'lexical_specificity':
        words_list = request.form.get('words_list')
        if not words_list:
            return Response(json.dumps({"error": "Words list not specified"}), status=400, mimetype='application/json')

    # Create result directory with error handling
    try:
        rand_name = generate_rand_name('lexicale_')
        result_path = create_named_directory(rand_name)
    except Exception as e:
        return Response(json.dumps({"error": f"Failed to create result directory: {str(e)}"}), 
                       status=500, mimetype='application/json')

    try:
        for f in files:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            unique_words = set(tokens)
            number_unique_words = len(unique_words)
            total_number_words = len(tokens)
            TTR = number_unique_words / total_number_words if total_number_words > 0 else 0
            filename, _ = os.path.splitext(f.filename)

            if analysis_type == 'lexical_dispersion':
                plt.figure(figsize=(10, 6))
                Text(tokens).dispersion_plot(analyzed_words)
                plt.xlabel('Word Offset')
                plt.ylabel('Frequency')
                plt.title('Lexical Dispersion Plot')
                plt.tight_layout()
                vis_path = os.path.join(result_path, f'{filename}_dispersion_plot.png')
                plt.savefig(vis_path, format='png')
                plt.close()

            elif analysis_type == 'lexical_diversity':
                # Save diversity metrics
                with open(os.path.join(result_path, f'{filename}_diversity.txt'), 'w', encoding='utf-8') as out:
                    out.write(f"The lexical richness of the text is: {round(TTR, 2)}")

                # Create visualization
                plt.figure(figsize=(10, 6))
                plt.bar(['Total Words', 'Unique Words'], 
                       [total_number_words, number_unique_words], 
                       color=['blue', 'green'])
                plt.ylabel('Count')
                plt.title('Total Words vs Unique Words')
                plt.text(0.5, max(total_number_words, number_unique_words)/2, 
                        f'Type-Token Ratio (TTR): {round(TTR, 2)}', 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        fontsize=12, 
                        color='black')
                plt.savefig(os.path.join(result_path, f'{filename}_words_comparison.png'))
                plt.close()

            elif analysis_type == 'lexical_relationships':
                synonyms, antonyms, hyponyms, hypernyms = set(), set(), set(), set()
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                        if lemma.antonyms():
                            antonyms.add(lemma.antonyms()[0].name())
                    hyponyms.update(lemma.name() for hypo in syn.hyponyms() for lemma in hypo.lemmas())
                    hypernyms.update(lemma.name() for hyper in syn.hypernyms() for lemma in hyper.lemmas())
                
                with open(os.path.join(result_path, f'{filename}_relationships.txt'), 'w', encoding='utf-8') as out:
                    out.write(f"Synonyms: {list(synonyms)}\nAntonyms: {list(antonyms)}\n"
                            f"Hyponyms: {list(hyponyms)}\nHypernyms: {list(hypernyms)}")

            elif analysis_type == 'lexical_specificity':
                if not words_list:
                    continue
                    
                text_without_stopwords = [w for w in words_list.lower().split(";") if w not in stop_words]
                if not text_without_stopwords:
                    continue

                # Calculate TF-IDF
                tf = Counter(text_without_stopwords)
                total_words = len(text_without_stopwords)
                tf = {word: count / total_words for word, count in tf.items()}
                word_doc_counts = {word: sum(1 for doc in [input_text] if word in doc.lower().split()) 
                                 for word in tf}
                idf = {word: math.log(1 / max(count, 1)) for word, count in word_doc_counts.items()}
                tf_idf = {word: tf[word] * idf[word] for word in tf}
                sorted_tf_idf = sorted(tf_idf.items(), key=lambda item: item[1], reverse=True)

                # Save results
                with open(os.path.join(result_path, f'{filename}_specificity.txt'), 'w', encoding='utf-8') as out:
                    out.write(f"Results: {sorted_tf_idf}")

                # Create visualization
                words, scores = zip(*sorted_tf_idf[:10])  # Only show top 10 words
                plt.figure(figsize=(10, 6))
                plt.bar(words, scores, color='grey')
                plt.xticks(rotation=45)
                plt.xlabel('Words')
                plt.ylabel('TF-IDF Score')
                plt.title('TF-IDF Scores for Words in Text')
                plt.tight_layout()
                plt.savefig(os.path.join(result_path, f'{filename}_specificity.png'))
                plt.close()

    except Exception as e:
        shutil.rmtree(result_path, ignore_errors=True)
        return Response(json.dumps({"error": f"Analysis failed: {str(e)}"}), 
                       status=500, mimetype='application/json')

    # Create and return zip file
    try:
        if not os.listdir(result_path):
            shutil.rmtree(result_path)
            return Response(json.dumps({"error": "No output files were generated"}), 
                          status=500, mimetype='application/json')

        zip_path = f"{result_path}.zip"
        shutil.make_archive(result_path, 'zip', result_path)
        
        with open(zip_path, 'rb') as zip_file:
            response = Response(zip_file.read(), 
                              mimetype='application/zip',
                              headers={"Content-disposition": f"attachment; filename={rand_name}.zip"})
        
        # Cleanup
        shutil.rmtree(result_path)
        os.remove(zip_path)
        
        return response

    except Exception as e:
        shutil.rmtree(result_path, ignore_errors=True)
        if os.path.exists(f"{result_path}.zip"):
            os.remove(f"{result_path}.zip")
        return Response(json.dumps({"error": f"Failed to create zip file: {str(e)}"}), 
                       status=500, mimetype='application/json')


#--------------- Analyse de texte --------------------------

# Cache des pipelines
pipeline_cache = {}

def get_pipeline(task, model, **kwargs):
    key = f"{task}_{model}"
    if key not in pipeline_cache:
        pipeline_cache[key] = pipeline(task, model=model, **kwargs)
    return pipeline_cache[key]


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    if 'input_text' not in request.form:
        response = {"error": "No text part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    input_text = request.form.get('input_text', '').splitlines()

    analysis_type = request.form['analysis_type']
    emotion_type = request.form['emotion_type']


    rand_name = generate_rand_name('textanalysis_')
    result_path = create_named_directory(rand_name)

    
    if analysis_type == 'subjectivity_detection':
        output_name = 'subjectivity_detection.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier1 = get_pipeline('text-classification', model='GroNLP/mdebertav3-subjectivity-multilingual')
                result = classifier1(text)[0]
                label = "objective" if result['label'] == "LABEL_0" else "subjective"
                out.write(f"The sentence: [{text}] is {label} (Score: {result['score']:.2f})\n\n")
    elif analysis_type == 'sentiment_analysis':
        output_name = 'sentiment_analysis.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier2 = get_pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                results = classifier2(text)
                star_rating = int(results[0]['label'].split()[0])
                sentiment = "negative" if star_rating in [1, 2] else "neutral" if star_rating == 3 else "positive"
                out.write(f"Sentence: {text}\n Star Rating: {star_rating} \n Sentiment: {sentiment} \n Score: {results[0]['score']:.2f}\n\n")
    elif analysis_type == 'subjectivity_sentiment':
        output_name = 'subjectivity_sentiment.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier1 = get_pipeline('text-classification', model='GroNLP/mdebertav3-subjectivity-multilingual')
                result = classifier1(text)[0]
                label = "objective" if result['label'] == "LABEL_0" else "subjective"
                classifier2 = get_pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                results = classifier2(text)
                star_rating = int(results[0]['label'].split()[0])
                sentiment = "negative" if star_rating in [1, 2] else "neutral" if star_rating == 3 else "positive"
                out.write(f"The sentence: [{text}] is {label} (Score: {result['score']:.2f})")
                out.write(f"\nStar Rating: {star_rating} \n Sentiment: {sentiment} \n Score: {results[0]['score']:.2f}\n\n")
    elif analysis_type == 'subjectivity_sentiment_emotion':
        output_name = 'subjectivity_sentiment_emotion.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier1 = get_pipeline('text-classification', model='GroNLP/mdebertav3-subjectivity-multilingual')
                result = classifier1(text)[0]
                label = "objective" if result['label'] == "LABEL_0" else "subjective"
                classifier2 = get_pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                results = classifier2(text)
                star_rating = int(results[0]['label'].split()[0])
                sentiment = "negative" if star_rating in [1, 2] else "neutral" if star_rating == 3 else "positive"
                classifier_distilbert = get_pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, top_k=None)
                vis1 = classifier_distilbert(text)
                emotions = [result['label'] for result in vis1[0]]
                scores = [result['score'] for result in vis1[0]]
                out.write(f"The sentence: [{text}] is {label} (Score: {result['score']:.2f})")
                out.write(f"\nStar Rating: {star_rating} \n Sentiment: {sentiment} \n Score: {results[0]['score']:.2f}\n\n")
                out.write(f"The emotions for the text are : \n")
                for emotion, score in zip(emotions, scores):
                    out.write(f"{emotion}: {score:.4f}\n")
                out.write("\n\n")
    elif analysis_type == 'subjectivity_emotion':
        output_name = 'subjectivity_emotion.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier1 = get_pipeline('text-classification', model='GroNLP/mdebertav3-subjectivity-multilingual')
                result = classifier1(text)[0]
                label = "objective" if result['label'] == "LABEL_0" else "subjective"
                classifier_distilbert = get_pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, top_k=None)
                vis1 = classifier_distilbert(text)
                emotions = [result['label'] for result in vis1[0]]
                scores = [result['score'] for result in vis1[0]]
                out.write(f"The sentence: [{text}] is {label} (Score: {result['score']:.2f})")
                out.write(f"\nThe emotions for the text are : \n")
                for emotion, score in zip(emotions, scores):
                    out.write(f"{emotion}: {score:.4f}\n")
                out.write("\n\n")
    elif analysis_type == 'sentiment_emotion':
        output_name = 'sentiment_emotion.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier2 = get_pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                results = classifier2(text)
                star_rating = int(results[0]['label'].split()[0])
                sentiment = "negative" if star_rating in [1, 2] else "neutral" if star_rating == 3 else "positive"
                classifier_distilbert = get_pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, top_k=None)
                vis1 = classifier_distilbert(text)
                emotions = [result['label'] for result in vis1[0]]
                scores = [result['score'] for result in vis1[0]]
                out.write(f"Sentence: {text}\n : {star_rating} \n Sentiment: {sentiment} \n Score: {results[0]['score']:.2f}\n\n")
                out.write(f"The emotions for the text are : \n")
                for emotion, score in zip(emotions, scores):
                    out.write(f"{emotion}: {score:.4f}\n")
                out.write("\n\n")
    elif analysis_type == 'emotion_analysis':
        if emotion_type == "analyse1":
            output_name = 'emotion_analysis.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                for text in input_text:
                    #Sortie
                    classifier_distilbert = get_pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, top_k=None)
                    vis1 = classifier_distilbert(text)
                    emotions = [result['label'] for result in vis1[0]]
                    scores = [result['score'] for result in vis1[0]]
                    out.write(f"The emotions for [{text}] are : \n")
                    for emotion, score in zip(emotions, scores):
                        out.write(f"{emotion}: {score:.4f}\n")
                    out.write("\n\n")

            text_base = "emotion_viz1_"
            for i, text in enumerate(input_text):
                #Visualisation
                source = [0] * len(emotions)
                target = list(range(1, len(emotions) + 1))
                value = scores
                node_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#bc80bd", "#ccebc5", "#ffed6f", "#8dd3c7", "#fb8072"]
                fig = go.Figure(data=[go.Sankey(
                    node=dict(pad=55, thickness=55, line=dict(color="black", width=0.5), label=["Input Text"] + emotions, color=node_colors),
                    link=dict(source=source, target=target, value=value)
                )])
                fig.update_layout(title_text="Emotion Classification", font_size=15)
                vis1_name = text_base + str(i) + '.png'
                vis1_path = os.path.join(result_path, vis1_name)
                fig.write_image(vis1_path, format='png')

        elif emotion_type == "analyse2":
            #Visualisation2
            text_base = "emotion_viz2_"  # Chaîne de caractères de base
            for i, text in enumerate(input_text):  # input_text est une liste de textes à analyser
                classifier_roberta = get_pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=True, top_k=None)
                vis2 = classifier_roberta(text)
                labels = [emotion['label'] for emotion in vis2[0]]
                scores = [emotion['score'] for emotion in vis2[0]]
                combined = list(zip(labels, scores))
                random.shuffle(combined)
                labels, scores = zip(*combined)
                scores = list(scores) + [scores[0]]
                angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))] + [0]
                fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                ax.fill(angles, scores, color='blue', alpha=0.5)
                ax.plot(angles, scores, color='blue', linewidth=1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels)
                plt.title('Emotion Classification')
                
                vis2_name = text_base + str(i) + '.png'
                vis2_path = os.path.join(result_path, vis2_name)
                plt.savefig(vis2_path, format='png')
                plt.close()


    elif analysis_type == 'readibility_scoring':
        output_name = 'readibility_scoring.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                flesch_reading_ease = textstat.flesch_reading_ease(text)
                out.write(f"Flesch Reading Ease Score: {flesch_reading_ease}\n\n")

    elif analysis_type == 'subjectivity_sentiment_emotion_readability':
        output_name = 'subjectivity_sentiment_emotion_readability.txt'
        with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
            for text in input_text:
                classifier1 = get_pipeline('text-classification', model='GroNLP/mdebertav3-subjectivity-multilingual')
                result = classifier1(text)[0]
                label = "objective" if result['label'] == "LABEL_0" else "subjective"
                classifier2 = get_pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                results = classifier2(text)
                star_rating = int(results[0]['label'].split()[0])
                sentiment = "negative" if star_rating in [1, 2] else "neutral" if star_rating == 3 else "positive"
                classifier_distilbert = get_pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, top_k=None)
                vis1 = classifier_distilbert(text)
                emotions = [result['label'] for result in vis1[0]]
                scores = [result['score'] for result in vis1[0]]
                flesch_reading_ease = textstat.flesch_reading_ease(text)
                out.write(f"The sentence: [{text}] is {label} (Score: {result['score']:.2f})")
                out.write(f"\nStar Rating: {star_rating} \n Sentiment: {sentiment} \n Score: {results[0]['score']:.2f}\n\n")
                out.write(f"The emotions for the text are : \n")
                for emotion, score in zip(emotions, scores):
                    out.write(f"{emotion}: {score:.4f}\n")
                out.write(f"\nFlesch Reading Ease Score: {flesch_reading_ease}\n\n")


    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#--------------- Comparison --------------------------

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
    with open(file_path, 'r') as file:
        return file.read()

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
            shutil.rmtree(result_path)
            os.remove(str(result_path) + '.zip')
            return response

        return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')
    
    else:
        return jsonify({"error": "Please upload at least two text files"}), 400


#--------------- Embeddings --------------------------
model_glove_cache = None # Cache pour le modèle

def get_glove_model():
    global model_glove_cache
    if model_glove_cache is None:
        model_glove_cache = api.load("glove-wiki-gigaword-100")  # Chargement unique
    return model_glove_cache


@app.route('/embedding_tool', methods=['POST'])
def embedding_tool():
    analysis_type = request.form['analysis_type']
    inputText = str(request.form.get('inputText'))
    input1 = str(request.form.get('input1'))
    input2 = str(request.form.get('input2'))
    input3 = str(request.form.get('input3'))
    words_list = str(request.form.get('words_list'))
    analyzed_words = words_list.split(';')

    model_glove = get_glove_model()

    rand_name = generate_rand_name('embedding_')
    result_path = create_named_directory(rand_name)

    try:
        if analysis_type == 'similarity':
            similar_words = model_glove.most_similar(inputText, topn=5)
            output_name = 'similarity.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                for term, similarity in similar_words:
                    out.write(f"{term}: {similarity:.4f}\n")
        elif analysis_type == 'relations':
            result = model_glove.most_similar(positive=[input1, input2], negative=[input3], topn=1)
            output_name = 'relations.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write(f"'{input1}' is to '{input2}' as '{input3}' is to '{result[0][0]}'")
        elif analysis_type == 'clustering':
            word_vectors = [model_glove[word] for word in analyzed_words]
            kmeans = KMeans(n_clusters=3, random_state=0).fit(word_vectors)
            clusters = kmeans.labels_

            #Sortie
            output_name = 'clustering.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                for word, cluster in zip(analyzed_words, clusters):
                    out.write(f"{word} is in cluster {cluster}\n")

            #Visualisation
            pca = PCA(n_components=2)
            word_vecs_2d = pca.fit_transform(word_vectors)
            plt.figure(figsize=(5, 5))
            for i, word in enumerate(analyzed_words):
                plt.scatter(word_vecs_2d[i, 0], word_vecs_2d[i, 1])
                plt.annotate(word, xy=(word_vecs_2d[i, 0], word_vecs_2d[i, 1]), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom')
            cluster_name = 'clustering.png'
            cluster_path = os.path.join(result_path, cluster_name)
            plt.savefig(cluster_path, format='png')
            plt.close()

    except Exception as e:
        response = {"error": str(e)}
        return Response(json.dumps(response), status=500, mimetype='application/json')

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')



#---------------------------------------------------------
# Visualisation
#---------------------------------------------------------
"""
@app.route("/run_renard",  methods=["GET", "POST"])
@stream_with_context
def run_renard():

    try: # For debugging  
        from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
        print("Available parameters for CoOccurrencesGraphExtractor:")
        print(help(CoOccurrencesGraphExtractor))
    except Exception as e:
        print(f"Error importing renard: {str(e)}")

    form = FlaskForm()
    if request.method == 'POST':
        try:
            min_appearances = int(request.form['min_appearances'])
            lang = request.form.get('toollang')
        min_appearances = int(request.form['min_appearances'])
        lang = request.form.get('toollang')

            if request.files['renard_upload'].filename != '':
                f = request.files['renard_upload']
                text = f.read()
                text = text.decode('utf-8')
            else:
                text = request.form['renard_txt_input']
            
            rand_name = 'renard_graph_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8))) + '.gexf'
            result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
            
            from renard.pipeline import Pipeline
            from renard.pipeline.tokenization import NLTKTokenizer
            from renard.pipeline.ner import BertNamedEntityRecognizer
            from renard.pipeline.character_unification import GraphRulesCharacterUnifier
            from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
            from renard.graph_utils import graph_with_names
            from renard.plot_utils import plot_nx_graph_reasonably
            import matplotlib.pyplot as plt
            import networkx as nx
            import base64
        if request.files['renard_upload'].filename != '':
            f = request.files['renard_upload']

            text = f.read()
            text = text.decode('utf-8')
        else:
            text = request.form['renard_txt_input']
        
        rand_name =  'renard_graph_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8))) + '.gexf'
        result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
        
        from renard.pipeline import Pipeline
        from renard.pipeline.tokenization import NLTKTokenizer
        from renard.pipeline.ner import NLTKNamedEntityRecognizer, BertNamedEntityRecognizer
        from renard.pipeline.character_unification import GraphRulesCharacterUnifier
        from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
        from renard.graph_utils import graph_with_names
        from renard.plot_utils import plot_nx_graph_reasonably
        import matplotlib.pyplot as plt
        import networkx as nx
        import base64

            BERT_MODELS = {
                "fra" : "Jean-Baptiste/camembert-ner",
                "eng" : "dslim/bert-base-NER",
                "spa" : "mrm8488/bert-spanish-cased-finetuned-ner"
            }
            
            pipeline = Pipeline(
            [
                NLTKTokenizer(),
                BertNamedEntityRecognizer(model=BERT_MODELS[lang]),
                GraphRulesCharacterUnifier(min_appearances=min_appearances),
                CoOccurrencesGraphExtractor(co_occurences_dist=35) 
            ], lang=lang)
        BERT_MODELS = {
            "fra" : "Jean-Baptiste/camembert-ner",
            "eng" : "dslim/bert-base-NER",
            "spa" : "mrm8488/bert-spanish-cased-finetuned-ner"
        }
        

        pipeline = Pipeline(
        [
            NLTKTokenizer(),
            BertNamedEntityRecognizer(model=BERT_MODELS[lang]), #NLTKNamedEntityRecognizer(),
            GraphRulesCharacterUnifier(min_appearances=min_appearances),
            CoOccurrencesGraphExtractor(co_occurrences_dist=35)
        ], lang = lang)

            out = pipeline(text)
            out.export_graph_to_gexf(result_path)
            G = graph_with_names(out.characters_graph)
            plot_nx_graph_reasonably(G)
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.clf()
            figdata_png = base64.b64encode(img.getvalue()).decode('ascii')
        out = pipeline(text)

        # Save GEXF network
        out.export_graph_to_gexf(result_path)

        # Networkx to plot
        G = graph_with_names(out.characters_graph)
        plot_nx_graph_reasonably(G)
        img = BytesIO() # file-like object for the image
        plt.savefig(img, format='png') # save the image to the stream
        img.seek(0) # writing moved the cursor to the end of the file, reset
        plt.clf() # clear pyplot
        figdata_png = base64.b64encode(img.getvalue()).decode('ascii')

            return render_template('outils/renard.html', form=form, graph=figdata_png, fname=str(rand_name))
        return render_template('renard.html', form=form, graph=figdata_png, fname=str(rand_name))

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return render_template('outils/renard.html', form=form, 
                                error=f"Pipeline error: {str(e)}")

    return render_template('outils/renard.html', form=form, graph="", fname="")
"""
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
            os.remove(result_path)
                
    return render_template('/collecter_corpus.html')

@app.route('/corpus_from_url',  methods=["GET","POST"])
@stream_with_context

#Modifiée pour travail local + corrections
def corpus_from_url():
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

    return render_template('collecter_corpus.html')

#----------------------- Wikisource -------------------

def generate_random_corpus(nb):

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
            page = urllib.request.urlopen(location)
        except Exception as e:
            with open('pb_url.log', 'a') as err_log:
                err_log.write("No server is associated with the following page:" + location + '\n')
                err_log.write(e)
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
    form = FlaskForm()
    input_format = request.form['input_format']
    res_ok = ""
    res_err = ""
    res = ""

    # Arks dans fichier
    if request.files['ark_upload'].filename != '':
        f = request.files['ark_upload']
        text = f.read()
        arks_list = re.split(r"[~\r\n]+", text)
    # Arks dans textarea
    else:
        arks_list = re.split(r"[~\r\n]+", request.form['ark_input'])

    # Prépare le dossier résultat
    rand_name =  generate_rand_name('corpus_gallica_')
    result_path = create_named_directory(rand_name)

    
    for arkEntry in arks_list:
        # Vérifie si une plage de pages est indiquée
        arkEntry = arkEntry.strip()
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
                with urllib.request.urlopen(url) as response, open(path_file, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                res_ok += url + '\n'
            except Exception as exc:
                res_err += url + '\n'
                continue
        
        elif input_format == 'img':
            # Nb de pages à télécharger : si tout le document, aller chercher l'info dans le service pagination de l'API
            if nb_p == 0:
                url_pagination = "https://gallica.bnf.fr/services/Pagination?ark={}".format(arkName)
                try:
                    with urllib.request.urlopen(url_pagination) as response:
                        soup = BeautifulSoup(response.read().decode('utf-8'), features="xml")
                        nb_p = soup.find('nbVueImages').get_text()
                except Exception as exc:
                    print(exc)

            # Parcours des pages à télécharger
            for i in range(int(debut), int(debut) + int(nb_p) + 1):
                taille = get_size(arkName, i)
                largeur = taille["width"]
                hauteur = taille["height"]
                url = "https://gallica.bnf.fr/iiif/ark:/12148/{}/f{}/{},{},{},{}/full/0/native.jpg".format(arkName, i, 0, 0, largeur, hauteur)
                print(url)
                outfile = "{}_{:04}.jpg".format(arkName, i)
                path_file = os.path.join(result_path, outfile)
                try:
                    with urllib.request.urlopen(url) as response, open(path_file, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                    res_ok += url + '\n'
                except Exception as exc:
                    res_err += url + '\n'

        else:
            print("Erreur de paramètre")
            abort(400)
    

    
    with open(os.path.join(result_path, 'download_report.txt'), 'w') as report:
        if res_err != "":
            report.write("Erreur de téléchargement pour : \n {}".format(res_err))
        else:
            res = len(arks_list)
            report.write("{} documents ont bien été téléchargés.\n".format(res))
            report.write(res_ok)

    response = create_zip_and_response(result_path, rand_name)
    return response

    return render_template('extraction_gallica', form=form)

def get_size(ark, i):
    url = "https://gallica.bnf.fr/iiif/ark:/12148/{}/f{}/info.json".format(ark, i)
    try:
        with urllib.request.urlopen(url) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as exc:
        print("Erreur {} en récupérant {}".format(exc, url))
        return None

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
# Correction de corpus
#-----------------------------------------------------------------

#------------- Normalisation de graphie ---------------------
@app.route('/normalisation_graphies', methods=['POST'])
def normalisation_graphies():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    rand_name = generate_rand_name('normgraph_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            doc = nlp_eng(input_text)
            filename, file_extension = os.path.splitext(f.filename)
            
            output_name = filename + '.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                for token in doc:
                    out.write(f"Token: {token.text} --> POS: {token.pos_}\n")

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')


#------------- Correction Erreurs ---------------------

@app.route('/autocorrect', methods=["GET", "POST"])
def autocorrect():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    selected_language = request.form['selected_language']
    nlp = get_nlp(selected_language)
    if 'contextual spellchecker' not in nlp.pipe_names:
        contextualSpellCheck.add_to_pipe(nlp)

    rand_name = generate_rand_name('autocorrected_')
    result_path = create_named_directory(rand_name)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            doc = nlp(input_text)
            filename, file_extension = os.path.splitext(f.filename)
            output_name = filename + '.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write("The input text was : \n" + str(input_text) + "\n\nThe corrected text is: \n" + str(doc._.outcome_spellCheck))

        finally:
            f.close()

    response = create_zip_and_response(result_path, rand_name)
    return response

    return Response(json.dumps({"error": "Une erreur est survenue dans le traitement des fichiers."}), status=500, mimetype='application/json')

#-----------------------------------------------------------------
# Génération de texte
#-----------------------------------------------------------------

#------------- Complétion de texte ---------------------

#------------- Q/A ---------------------

#------------- Traduction ---------------------

#------------- Ajustement du niveau de lecture ---------------------


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

def spacy_lemmatizer(text):
# Input : raw text
# Ouput : lemmatized text
    import spacy
    nlp = spacy.load('fr_core_news_md')
    doc = nlp(text)
    result = []
    for d in doc:
        result.append(d.lemma_)
    return " ".join(result)

def createRandomDir(prefix, length):
    rand_name =  prefix + ''.join((random.choice(string.ascii_lowercase) for x in range(length)))
    result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
    os.mkdir(result_path)
    return (result_path, rand_name)

def getWikiPage(url):
# Renvoie le contenu d'un texte wikisource à partir de son url, -1 en cas d'erreur
    page = urllib.request.urlopen(url)
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
    from txt_ner import txt_ner_params
    from ocr import tesseract_to_txt
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


@app.route("/run_ocr_map", methods=["POST"])
def run_ocr_map():
    from txt_ner import txt_ner_params
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="http")

    # paramètres globaux
    uploaded_files = request.files.getlist("inputfiles")
    # paramètres OCR
    ocr_model = request.form['tessmodel']
    # paramètres NER
    up_folder = app.config['UPLOAD_FOLDER']
    encodage = request.form['encodage']
    moteur_REN = request.form['moteur_REN']
    modele_REN = request.form['modele_REN']

    rand_name =  generate_rand_name('ocr_ner_')
    if ocr_model != "raw_text":
        contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, '', rand_name, ROOT_FOLDER, up_folder)
    else:
        liste_contenus = []
        for uploaded_file in uploaded_files:
            try:
                liste_contenus.append(uploaded_file.read().decode(encodage))
            finally: # ensure file is closed
                uploaded_file.close()
        contenu = "\n\n".join(liste_contenus)

        del liste_contenus

    entities = txt_ner_params(contenu, moteur_REN, modele_REN, encodage=encodage)
    ensemble_mentions = set(text for label, start, end, text in entities if label == "LOC")
    coordonnees = []
    for texte in ensemble_mentions:
        location = geolocator.geocode(texte, timeout=30)
        if location:
            coordonnees.append(to_geoJSON_point(location, texte))

    return {"points": coordonnees}

#---------------------------------------------------------
#AFFICHAGE MAP des résultats pour plusieurs outils de NER
#---------------------------------------------------------

@app.route("/run_ocr_map_intersection", methods=["GET", "POST"])
def run_ocr_map_intersection():
    from txt_ner import txt_ner_params
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="http")
    # paramètres globaux
    uploaded_files = request.files.getlist("inputfiles")
    #print(uploaded_files)
    lang = request.form.get('toollang')
    # paramètres OCR
    #ocr_model = request.form['tessmodel']

    # paramètres NER
    up_folder = app.config['UPLOAD_FOLDER']
    encodage = request.form['encodage']
    moteur_REN1 = request.form['moteur_REN1']
    modele_REN1 = request.form['modele_REN1']
    moteur_REN2 = request.form['moteur_REN2']
    modele_REN2 = request.form['modele_REN2']
    frequences_1 = collections.Counter()
    frequences_2 = collections.Counter()
    frequences = collections.Counter()
    outil_1 = f"{moteur_REN1}/{modele_REN1}"
    outil_2 = (f"{moteur_REN2}/{modele_REN2}" if moteur_REN2 != "aucun" else "aucun")

    # print(moteur_REN1, moteur_REN2)

    if request.form.get("do_ocr"):
        rand_name =  generate_rand_name('ocr_ner_')
        contenu = ocr.tesseract_to_txt(uploaded_files, lang, '', rand_name, ROOT_FOLDER, up_folder)
        print("Numérisation en cours...")
    else:
        liste_contenus = []
        for uploaded_file in uploaded_files:
            #print(uploaded_file)
            try:
                f = uploaded_file.read()
                liste_contenus.append(f.decode(encodage))
                #print(liste_contenus)
            finally: # ensure file is closed
                uploaded_file.close()
        contenu = "\n\n".join(liste_contenus)

        del liste_contenus

    # TODO: ajout cumul
    entities_1 = txt_ner_params(contenu, moteur_REN1, modele_REN1, encodage=encodage)
    ensemble_mentions_1 = set(text for label, start, end, text in entities_1 if label == "LOC")
    ensemble_positions_1 = set((text, start, end) for label, start, end, text in entities_1 if label == "LOC")
    ensemble_positions = set((text, start, end) for label, start, end, text in entities_1 if label == "LOC")

    # TODO: ajout cumul
    if moteur_REN2 != "aucun":
        entities_2 = txt_ner_params(contenu, moteur_REN2, modele_REN2, encodage=encodage)
        ensemble_mentions_2 = set(text for label, start, end, text in entities_2 if label == "LOC")
        ensemble_positions_2 = set((text, start, end) for label, start, end, text in entities_2 if label == "LOC")
        ensemble_positions |= set((text, start, end) for label, start, end, text in entities_2 if label == "LOC")
    else:
        entities_2 = ()
        ensemble_positions_2 = set()
        ensemble_mentions_2 = set()

    ensemble_mentions_commun = ensemble_mentions_1 & ensemble_mentions_2
    ensemble_mentions_1 -= ensemble_mentions_commun
    ensemble_mentions_2 -= ensemble_mentions_commun

    for text, start, end in ensemble_positions_1:
        frequences_1[text] += 1
    for text, start, end in ensemble_positions_2:
        frequences_2[text] += 1
    for text, start, end in ensemble_positions:
        frequences[text] += 1

    # print("TEST1")

    text2coord = {}
    for text in set(p[0] for p in ensemble_positions):
        text2coord[text] = geolocator.geocode(text, timeout=30) # check for everyone

    # TODO: faire clustering pour cumul + outil 1 / outil 2 / commun
    clusters_1 = freqs2clustering(frequences_1)
    clusters_2 = freqs2clustering(frequences_2)
    clusters = freqs2clustering(frequences)

    # print("TEST2")
    frequences_cumul_1 = {}
    for centroid in clusters_1:
        frequences_cumul_1[centroid] = 0
        for forme_equivalente in clusters_1[centroid]["Termes"]:
            frequences_cumul_1[centroid] += frequences_1[forme_equivalente]
    frequences_cumul_2 = {}
    for centroid in clusters_2:
        frequences_cumul_2[centroid] = 0
        for forme_equivalente in clusters_2[centroid]["Termes"]:
            frequences_cumul_2[centroid] += frequences_2[forme_equivalente]
    frequences_cumul = {}
    for centroid in clusters:
        frequences_cumul[centroid] = 0
        for forme_equivalente in clusters[centroid]["Termes"]:
            frequences_cumul[centroid] += frequences[forme_equivalente]

    # print("TEST3")

    # TODO: ajout cumul
    liste_keys = ["commun", outil_1, outil_2]
    liste_ensemble_mention = [ensemble_mentions_commun, ensemble_mentions_1, ensemble_mentions_2]
    dico_mention_marker = {key: [] for key in liste_keys}
    for key, ensemble in zip(liste_keys, liste_ensemble_mention):
        if key == "commun":
            my_clusters = clusters
            my_frequences = frequences_cumul
        elif key == outil_1:
            my_clusters = clusters_1
            my_frequences = frequences_cumul_1
        elif key == outil_2:
            my_clusters = clusters_2
            my_frequences = frequences_cumul_2
        sous_ensemble = [texte for texte in my_frequences if texte in ensemble]
        for texte in sous_ensemble:
            # forms = (" / ".join(my_clusters[texte]["Termes"]) if my_clusters else "")
            #SAVE forms = [(form, [0, 0]) for form in my_clusters[texte]["Termes"]]
            forms = []
            for form in my_clusters[texte]["Termes"]:
                coords = text2coord[form]
                if coords:
                    coords = [text2coord[form].latitude, text2coord[form].longitude]
                else:
                    coords = [0.0, 0.0]
                forms.append([form, coords])
            # location = geolocator.geocode(texte, timeout=30) # déjà fait avant
            location = text2coord[texte]
            # print(location, file=sys.stderr)
            if location:
                dico_mention_marker[key].append((
                    location.latitude,
                    location.longitude,
                    texte,
                    my_frequences[texte],
                    forms
                ))

    # for key, value in dico_mention_marker.items():
    #     print(key, value, file=sys.stderr)

    return dico_mention_marker


@app.route("/nermap_to_csv", methods=['GET', "POST"])
@stream_with_context
def nermap_to_csv():
    input_json_str = request.data
    print(input_json_str)
    input_json = json.loads(input_json_str)
    print(input_json)
    keys = ["nom", "latitude", "longitude", "outil", "fréquence", "cluster"]
    output_stream = StringIO()
    writer = csv.DictWriter(output_stream, fieldnames=keys, delimiter="\t")
    writer.writeheader()
    for point in input_json["data"]:
        row = {
            "latitude" : point[0],
            "longitude" : point[1],
            "nom" : point[2],
            "outil" : point[3],
            "fréquence" : point[4],
            "cluster" : point[5],
        }
        writer.writerow(row)
    # name not useful, will be handled in javascript
    response = Response(output_stream.getvalue(), mimetype='text/csv', headers={"Content-disposition": "attachment; filename=export.csv"})
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