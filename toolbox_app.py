#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from flask import Flask, abort, request, render_template, render_template_string, url_for, redirect, send_from_directory, Response, stream_with_context, session, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from forms import ContactForm, SearchForm
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_babel import Babel, get_locale
from langdetect import detect_langs
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
from nltk import FreqDist
from nltk import Text
from collections import Counter
from wordcloud import WordCloud
import zipfile
import os
from io import StringIO, BytesIO
import string
import random
from bs4 import BeautifulSoup
import urllib
import urllib.request
from urllib.parse import urlparse
import re
from lxml import etree
import csv
import contextualSpellCheck
import spacy
from spacy import displacy
import shutil
from pathlib import Path
import json
import collections
#from transformers import pipeline
#from txt_ner import txt_ner_params

#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('punkt')

import pandas as pd

import sem
#import sem.storage
#import sem.exporters

import ocr

from cluster import freqs2clustering

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'static/models'
UTILS_FOLDER = 'static/utils'
ROOT_FOLDER = Path(__file__).parent.absolute()

csrf = CSRFProtect()
SECRET_KEY = os.urandom(32)

app = Flask(__name__)

# Babel config
#def get_locale():
#    return request.accept_languages.best_match(['fr', 'en'])
#babel = Babel(app, locale_selector=get_locale)
babel = Babel(app)

# App config
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = SECRET_KEY

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # Limit file upload to 35MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['UTILS_FOLDER'] = UTILS_FOLDER
app.config['LANGUAGES'] = {
    'fr': 'FR',
    'en': 'EN',
}
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)
csrf.init_app(app)


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

@app.route('/tutoriel_recognition')
def tutoriel_recognition():
    return render_template('tutoriel/tutoriel_recognition.html')

@app.route('/tutoriel_preprocessing')
def tutoriel_preprocessing():
    return render_template('tutoriel/tutoriel_preprocessing.html')

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

@app.route('/outils_corpus')
def outils_corpus():
    return render_template('taches/corpus.html')

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
# Numérisation
#-----------------------------------------------------------------

#   NUMERISATION TESSERACT
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
        rand_name =  'ocr_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))


        text = ocr.tesseract_to_txt(uploaded_files, model, model_bis, rand_name, ROOT_FOLDER, up_folder)
        response = Response(text, mimetype='text/plain',
                            headers={"Content-disposition": "attachment; filename=" + rand_name + '.txt'})

        return response
    return render_template('numeriser.html', erreur=erreur)

#-----------------------------------------------------------------
# Prétraitement
#-----------------------------------------------------------------

#-------------- Nettoyage de texte -------------------------

# Importer les stopwords pour chaque langue
stop_words_english = set(stopwords.words('english'))
stop_words_french = set(stopwords.words('french'))
stop_words_spanish = set(stopwords.words('spanish'))
stop_words_german = set(stopwords.words('german'))
stop_words_danish = set(stopwords.words('danish'))
stop_words_finnish = set(stopwords.words('finnish'))
stop_words_greek = set(stopwords.words('greek'))
stop_words_italian = set(stopwords.words('italian'))
stop_words_dutch = set(stopwords.words('dutch'))
stop_words_polish = set(stopwords.words('polish'))
stop_words_portuguese = set(stopwords.words('portuguese'))
stop_words_russian = set(stopwords.words('russian'))

# Fonction pour obtenir les stopwords en fonction de la langue
def get_stopwords(language):
    if language == 'english':
        return stop_words_english
    elif language == 'french':
        return stop_words_french
    elif language == 'spanish':
        return stop_words_spanish
    elif language == 'german':
        return stop_words_german
    elif language == 'danish':
        return stop_words_danish
    elif language == 'finnish':
        return stop_words_finnish
    elif language == 'greek':
        return stop_words_greek
    elif language == 'italian':
        return stop_words_italian
    elif language == 'dutch':
        return stop_words_dutch
    elif language == 'polish':
        return stop_words_polish
    elif language == 'portuguese':
        return stop_words_portuguese
    elif language == 'russian':
        return stop_words_russian
    else:
        return set()

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

    rand_name = 'removing_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            removing_punctuation = [token for token in tokens if token.isalpha()]
            stop_words = get_stopwords(selected_language)
            removing_stopwords = [token for token in tokens if token.lower() not in stop_words]
            filename, file_extension = os.path.splitext(f.filename)

            if removing_type == 'punctuation':
                output_name = filename + '_punctuation.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The original text was :\n"' + input_text + '"\n\nThe text without punctuation is :\n"' + " ".join(removing_punctuation) + '"')
            elif removing_type == 'stopwords':
                output_name = filename + '_stopwords.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write('The original text was :\n"' + input_text + '"\n\nThe text without stopwords is :\n"' + " ".join(removing_stopwords) + '"')



        finally:
            f.close()

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


#-------------- Normalisation de texte -------------------------


nlp_eng = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')
nlp_es = spacy.load('es_core_news_sm')
nlp_de = spacy.load('de_core_news_sm')
nlp_it = spacy.load('it_core_news_sm')
nlp_da = spacy.load("da_core_news_sm")
nlp_nl = spacy.load("nl_core_news_sm")
nlp_fi = spacy.load("fi_core_news_sm")
nlp_pl = spacy.load("pl_core_news_sm")
nlp_pt = spacy.load("pt_core_news_sm")
nlp_el = spacy.load("el_core_news_sm")
nlp_ru = spacy.load("ru_core_news_sm")

def get_nlp(language):
    if language == 'english':
        return nlp_eng
    elif language == 'french':
        return nlp_fr
    elif language == 'spanish':
        return nlp_es
    elif language == 'german':
        return nlp_de
    elif language == 'italian':
        return nlp_it
    elif language == 'danish':
        return nlp_da
    elif language == 'dutch':
        return nlp_nl
    elif language == 'finnish':
        return nlp_fi
    elif language == 'polish':
        return nlp_pl
    elif language == 'portuguese':
        return nlp_pt
    elif language == 'greek':
        return nlp_el
    elif language == 'russian':
        return nlp_ru
    else:
        return set()

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

    rand_name = 'normalized_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            lowers = [token.lower() for token in tokens]
            nlp = get_nlp(selected_language)
            lemmas = [token.lemma_ for token in nlp(input_text)]
            filename, file_extension = os.path.splitext(f.filename)

            if normalisation_type == 'tokens':
                output_name = filename + '_tokens.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The tokens of the text are: " + ", ".join(tokens))
            elif normalisation_type == 'lowercases':
                output_name = filename + '_lower.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The lowercase version of the text is: " + ", ".join(lowers))
            elif normalisation_type == 'lemmas':
                output_name = filename + '_lemmas.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The lemmas of the text are: " + ", ".join(lemmas))


        finally:
            f.close()

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

#-------------- Séparation de texte -------------------------

@app.route('/split_sentences', methods=['POST'])
def split_sentences():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    rand_name = 'splitsentences_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            sentences = nltk.sent_tokenize(input_text)
            splitsentence = [sentence.strip() for sentence in sentences]
            filename, file_extension = os.path.splitext(f.filename)
            
            output_name = filename + '.txt'
            with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                out.write("The sentences of the text are: " + ",\n".join(splitsentence))

        finally:
            f.close()

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

    rand_name = 'postagging_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

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
    rand_name =  'ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
    result_path = ROOT_FOLDER / os.path.join(UPLOAD_FOLDER, rand_name)
    os.mkdir(result_path)

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
                    output_name = os.path.join(result_path, filename + ".ann")
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

    render_template('entites_nommees.html', erreur=erreur)

#-----------------------------------------------------------------
# Extraction d'informations
#-----------------------------------------------------------------

#--------------- Mots-clés -----------------------


@app.route('/keyword_extraction', methods=['POST'])

@stream_with_context
def keyword_extraction():
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

            # Import required libraries
            from keybert import KeyBERT
            from sentence_transformers import SentenceTransformer
            
            # Load model
            print("Loading models...")
            sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
            kw_model = KeyBERT(model=sentence_model)
            print("Models loaded successfully")
            
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

                    if 'default' in methods:
                        print("Running default extraction...")
                        keywords_def = kw_model.extract_keywords(text)
                        res[fname]['default'] = keywords_def

                    if 'mmr' in methods:
                        print("Running MMR extraction...")
                        diversity = float(request.form.get('diversity', '7')) / 10
                        keywords_mmr = kw_model.extract_keywords(text, use_mmr=True, diversity=diversity)
                        res[fname]['mmr'] = keywords_mmr

                    if 'mss' in methods:
                        print("Running MSS extraction...")
                        keywords_mss = kw_model.extract_keywords(
                            text, 
                            keyphrase_ngram_range=(1, 3),
                            use_maxsum=True,
                            nr_candidates=10,
                            top_n=3
                        )
                        res[fname]['mss'] = keywords_mss
                    
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

    rand_name = 'quotation_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

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

    rand_name = 'linguistic_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            hapaxes_list = find_hapaxes(input_text)
            langues_detectees = detect_langs(input_text)
            langues_probabilites = [f"Langue : {lang.lang}, Probabilité : {lang.prob * 100:.2f}%" for lang in langues_detectees]
            detected_languages_str = "\n".join(langues_probabilites)
            filename, file_extension = os.path.splitext(f.filename)

            if analysis_type == 'hapax':
                output_name = filename + '_hapaxes.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The hapaxes are: " + ", ".join(hapaxes_list))
            elif analysis_type == 'n_gram':
                most_frequent_ngrams = generate_ngrams(input_text, n, r)
                output_name = filename + '_ngrams.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    for n_gram, count in most_frequent_ngrams:
                        out.write(f"{n}-gram: {' '.join(n_gram)} --> Count: {count}\n")
            elif analysis_type == 'detect_lang':
                output_name = filename + '_detected_languages.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(f"Detected languages:\n{detected_languages_str}\n")
            elif analysis_type == 'dependency':
                # Dependency parsing
                doc = nlp_eng(input_text)
                syntax_info = "\n".join([f"{token.text} ({token.pos_}) <--{token.dep_} ({spacy.explain(token.dep_)})-- {token.head.text} ({token.head.pos_})" for token in doc])
                output_name_text = filename + '_syntax.txt'
                with open(os.path.join(result_path, output_name_text), 'w', encoding='utf-8') as out:
                    out.write(syntax_info)

                # Visualization with displacy
                svg = displacy.render(doc, style='dep', jupyter=False)
                output_name_svg = filename + '_syntax.svg'
                with open(os.path.join(result_path, output_name_svg), 'w', encoding='utf-8') as out:
                    out.write(svg)
        finally:
            f.close()

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

    rand_name = 'statistics_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            # Sentence length average
            sentences = sent_tokenize(input_text)
            total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
            total_sentences = len(sentences)
            average_words_per_sentence = total_words / total_sentences

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
                output_name = filename + '_length.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Total Words: " + str(total_words) + "\nTotal Sentences: " + str(total_sentences) + "\nAverage Words per Sentence: " + str(average_words_per_sentence))
                
                # Generate sentence length visualization
                fig, ax = plt.subplots()
                ax.bar(range(1, total_sentences + 1), sentence_lengths, color='blue', alpha=0.7)
                ax.axhline(average_words_per_sentence, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlabel('Sentence Number')
                ax.set_ylabel('Number of Words')
                ax.set_title('Number of Words per Sentence')
                ax.legend(['Average Words per Sentence', 'Words per Sentence'])
                
                # Save visualization to a file
                vis_name = filename + '_sentence_lengths.png'
                vis_path = os.path.join(result_path, vis_name)
                plt.savefig(vis_path, format='png')
                plt.close()

            elif analysis_type == 'words_frequency':
                output_name = filename + '_wordsfrequency.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("Absolute frequency of words: " + str(abs_frequency) + "\nRelative frequency of words: " + str(rel_frequency) + "\nTotal number of words:" + str(total_tokens))
                
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
        
        finally:
            f.close()

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


#--------------- Analyse lexicale --------------------------

@app.route('/analyze_lexicale', methods=['POST'])
def analyze_lexicale():
    if 'files' not in request.files:
        response = {"error": "No files part"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        response = {"error": "No selected files"}
        return Response(json.dumps(response), status=400, mimetype='application/json')

    analysis_type = request.form['analysis_type']

    words_to_analyze = str(request.form.get('words_to_analyze'))

    rand_name = 'lexicale_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            tokens = word_tokenize(input_text)
            unique_words = set(tokens)
            number_unique_words = len(unique_words)
            total_number_words = len(tokens)
            TTR = number_unique_words / total_number_words
            filename, file_extension = os.path.splitext(f.filename)

            if analysis_type == 'lexical_dispersion':
                text_nltk = Text(tokens) 
                fig = plt.figure(figsize=(10, 6)) 
                text_nltk.dispersion_plot(words_to_analyze) 
                # Personnalisation du plot 
                plt.xlabel('Word Offset') 
                plt.ylabel('Frequency') 
                plt.title('Lexical Dispersion Plot') 
                plt.tight_layout() 
                # Sauvegarder la visualisation dans un fichier 
                vis_name = filename + '_dispersion_plot.png' 
                vis_path = os.path.join(result_path, vis_name) 
                plt.savefig(vis_path, format='png') 
                plt.close()
            elif analysis_type == 'lexical_diversity':
                #Sortie
                output_name = filename + '_diversity.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The lexical richness of the text is:" + str(round(TTR, 2)))

                #Visualisation
                labels = ['Total Words', 'Unique Words']
                values = [total_number_words, number_unique_words]

                # Création du graphique à barres
                fig, ax = plt.subplots()
                plt.bar(labels, values, color=['blue', 'green'])
                plt.ylabel('Count')
                plt.title('Total Words vs Unique Words')
                plt.text(0.5, max(values)/2, f'TTR: {round(TTR, 2)}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')

                # Sauvegarder la visualisation dans un fichier
                vis_name = 'words_comparison.png'
                vis_path = os.path.join(result_path, vis_name)
                plt.savefig(vis_path, format='png')
                plt.close()
            elif analysis_type == 'lexical_relationships':
                output_name = filename + '_relationships.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(f"Detected languages:\n{detected_languages_str}\n")
            elif analysis_type == 'lexical_specificity':
                # Dependency parsing
                doc = nlp_eng(input_text)
                syntax_info = "\n".join([f"{token.text} ({token.pos_}) <--{token.dep_} ({spacy.explain(token.dep_)})-- {token.head.text} ({token.head.pos_})" for token in doc])
                output_name_text = filename + '_specifity.txt'
                with open(os.path.join(result_path, output_name_text), 'w', encoding='utf-8') as out:
                    out.write(syntax_info)

                # Visualization with displacy
                svg = displacy.render(doc, style='dep', jupyter=False)
                output_name_svg = filename + '_specifity.svg'
                with open(os.path.join(result_path, output_name_svg), 'w', encoding='utf-8') as out:
                    out.write(svg)
        finally:
            f.close()

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



#--------------- Analyse de texte --------------------------

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
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

    rand_name = 'textanalysis_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

    for f in files:
        try:
            input_text = f.read().decode('utf-8')
            filename, file_extension = os.path.splitext(f.filename)

            if analysis_type == 'subjectivity_detection':
                output_name = filename + '_subjectivity_detection.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write("The hapaxes are: " + ", ".join(hapaxes_list))
            elif analysis_type == 'sentiment_analysis':
                most_frequent_ngrams = generate_ngrams(input_text, n, r)
                output_name = filename + '_sentiment_analysis.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    for n_gram, count in most_frequent_ngrams:
                        out.write(f"{n}-gram: {' '.join(n_gram)} --> Count: {count}\n")
            elif analysis_type == 'emotion_analysis':
                most_frequent_ngrams = generate_ngrams(input_text, n, r)
                output_name = filename + '_emotion_analysis.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    for n_gram, count in most_frequent_ngrams:
                        out.write(f"{n}-gram: {' '.join(n_gram)} --> Count: {count}\n")
            elif analysis_type == 'readibility_scoring':
                output_name = filename + '_readibility_scoring.txt'
                with open(os.path.join(result_path, output_name), 'w', encoding='utf-8') as out:
                    out.write(f"Detected languages:\n{detected_languages_str}\n")
            elif analysis_type == 'comparison':
                # Dependency parsing
                doc = nlp_eng(input_text)
                syntax_info = "\n".join([f"{token.text} ({token.pos_}) <--{token.dep_} ({spacy.explain(token.dep_)})-- {token.head.text} ({token.head.pos_})" for token in doc])
                output_name_text = filename + '_comparison.txt'
                with open(os.path.join(result_path, output_name_text), 'w', encoding='utf-8') as out:
                    out.write(syntax_info)

                # Visualization with displacy
                svg = displacy.render(doc, style='dep', jupyter=False)
                output_name_svg = filename + '_comparison.svg'
                with open(os.path.join(result_path, output_name_svg), 'w', encoding='utf-8') as out:
                    out.write(svg)
        finally:
            f.close()

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
    rand_name =  'corpus_gallica_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
    result_path = ROOT_FOLDER / os.path.join(UPLOAD_FOLDER, rand_name)
    os.mkdir(result_path)

    
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

    return render_template('extraction_gallica', form=form)

def get_size(ark, i):
    url = "https://gallica.bnf.fr/iiif/ark:/12148/{}/f{}/info.json".format(ark, i)
    try:
        with urllib.request.urlopen(url) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as exc:
        print("Erreur {} en récupérant {}".format(exc, url))
        return None

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

    rand_name = 'normgraph_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

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
    contextualSpellCheck.add_to_pipe(nlp)

    rand_name = 'autocorrected_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    result_path = os.path.join(os.getcwd(), rand_name)
    os.makedirs(result_path, exist_ok=True)

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


"""
def autocorrect():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("uploaded_files")

        # Initialisation du correcteur
        modelpath = str(ROOT_FOLDER / os.path.join(app.config['MODEL_FOLDER'], 'fr.bin'))
        jsp = jamspell.TSpellCorrector()
        assert jsp.LoadLangModel(modelpath) # modèle de langue

        # Nom de dossier aléatoire pour le résultat de la requête
        rand_name =  'autocorrect_' + ''.join((random.choice(string.ascii_lowercase) for x in range(5)))
        result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
        os.mkdir(result_path)
        print(result_path)

        for f in uploaded_files:
            filename, file_extension = os.path.splitext(f.filename)
            output_name = filename + '_OK.txt'             # Texte corrigé
            #tabfile_name = filename + '_corrections.tsv'   # Liste des corrections

            byte_str = f.read()
            f.close()

            # Prétraitements du texte d'entrée
            texte = byte_str.decode('UTF-8')
            texte = re.sub(" +", " ", texte) # Suppression espaces multiples
            texte = texte.replace("'", "’") # Guillemet français
            phrases = sentencizer(texte)

            with open(ROOT_FOLDER / os.path.join(result_path, output_name), 'a+', encoding="utf-8") as out:
                for sent in phrases:
                    correction = jsp.FixFragment(sent)
                    out.write(correction)

        # ZIP le dossier résultat
        shutil.make_archive(result_path, 'zip', result_path)
        output_stream = BytesIO()
        with open(str(result_path) + '.zip', 'rb') as res:
            content = res.read()
        output_stream.write(content)
        response = Response(output_stream.getvalue(), mimetype='application/zip',
                                headers={"Content-disposition": "attachment; filename=" + rand_name + '.zip'})
        output_stream.seek(0)
        output_stream.truncate(0)

        # Nettoie le dossier de travail
        shutil.rmtree(result_path)

        return response

    return render_template('/correction_erreur.html')

"""

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

    # paramètres globaux
    uploaded_files = request.files.getlist("inputfiles")
    # paramètres OCR
    ocr_model = request.form['tessmodel']
    # paramètres NER
    up_folder = app.config['UPLOAD_FOLDER']
    encodage = request.form['encodage']
    moteur_REN = request.form['moteur_REN']
    modele_REN = request.form['modele_REN']

    rand_name =  'ocr_ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
    if ocr_model != "raw_text":
        contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, '', rand_name, ROOT_FOLDER, up_folder)
    else:
        print(uploaded_files)
        liste_contenus = []
        for uploaded_file in uploaded_files:
            try:
                liste_contenus.append(uploaded_file.read().decode(encodage))

            finally: # ensure file is closed
                uploaded_file.close()
        contenu = "\n\n".join(liste_contenus)

        del liste_contenus

    entities = txt_ner_params(contenu, moteur_REN, modele_REN, encodage=encodage)
    # Writing in stream
    output_stream = StringIO()
    output = rand_name + ".ann"
    writer = csv.writer(output_stream, delimiter="\t")
    for nth, entity in enumerate(entities, 1):
        ne_type, start, end, text = entity
        row = [f"T{nth}", f"{ne_type} {start} {end}", f"{text}"]
        writer.writerow(row)
    response = Response(
        output_stream.getvalue(),
        mimetype='text/plain',
        headers={"Content-disposition": "attachment; filename=" + output}
    )
    output_stream.seek(0)
    output_stream.truncate(0)

    return response

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

    rand_name =  'ocr_ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
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
        rand_name =  'ocr_ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
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

