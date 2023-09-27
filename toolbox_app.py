#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from flask import Flask, abort, request, render_template, url_for, redirect, send_file, Response, stream_with_context, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from forms import ContactForm, SearchForm
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_babel import Babel, get_locale
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
import shutil
from pathlib import Path
import jamspell
import json
import collections
#from txt_ner import txt_ner_params

import pandas as pd

import sem
import sem.storage
import sem.exporters

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

app.config['MAX_CONTENT_LENGTH'] = 35 * 1024 * 1024 # Limit file upload to 35MB
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

@app.route('/outils')
def outils():
	form = SearchForm()
	return render_template('outils.html', form=form)

@app.route('/documentation')
def documentation():
	return render_template('documentation.html')

@app.route('/contact')
def contact():
	form = ContactForm()
	return render_template('contact.html', form=form)

@app.route('/outils_corpus')
def outils_corpus():
	return render_template('corpus.html')

@app.route('/outils_fouille')
def outils_fouille():
	return render_template('fouille_de_texte.html')

@app.route('/outils_visualisation')
def outils_visualisation():
	return render_template('visualisation.html')

@app.route('/numeriser')
def numeriser():
	form = FlaskForm()
	return render_template('numeriser.html', form=form)

@app.route('/normalisation')
def normalisation():
	return render_template('normalisation.html')

@app.route('/categories_semantiques')
def categories_semantiques():
	return render_template('categories_semantiques.html')

@app.route('/resume_automatique')
def resume_automatique():
	return render_template('resume_automatique.html')

@app.route('/extraction_mots_cles')
def extraction_mots_cles():
	form = FlaskForm()
	return render_template('extraction_mots_cles.html', form=form, res={})

@app.route('/topic_modelling')
def topic_modelling():
	form = FlaskForm()
	return render_template('topic_modelling.html', form=form, res={})

@app.route('/outils_pipeline')
def outils_pipeline():
	return render_template('pipeline.html')

@app.route('/ocr_ner')
def ocr_ner():
	form = FlaskForm()
	return render_template('ocr_ner.html', form=form)

@app.route('/ocr_map')
def ocr_map():
	form = FlaskForm()
	return render_template('ocr_map.html', form=form)

@app.route('/extraction_gallica')
def extraction_gallica():
	form = FlaskForm()
	return render_template('extraction_gallica.html', form=form)

@app.route('/extraction_wikisource')
def extraction_wikisource():
	form = FlaskForm()
	return render_template('extraction_wikisource.html', form=form)

@app.route('/tanagra')
def tanagra():
	return render_template('tanagra.html')

@app.route('/renard')
def renard():
	form = FlaskForm()
	return render_template('renard.html', form=form, graph="")
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


#   NUMERISATION TESSERACT
@app.route('/run_tesseract',  methods=["GET","POST"])
@stream_with_context
def run_tesseract():
	if request.method == 'POST':
		uploaded_files = request.files.getlist("tessfiles")
		model = request.form['tessmodel']

		up_folder = app.config['UPLOAD_FOLDER']
		rand_name =  'ocr_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))

		text = ocr.tesseract_to_txt(uploaded_files, model, rand_name, ROOT_FOLDER, up_folder)
		response = Response(text, mimetype='text/plain',
							headers={"Content-disposition": "attachment; filename=" + rand_name + '.txt'})

		return response
	return render_template('numeriser.html', erreur=erreur)

@app.route('/collecter_corpus')
def collecter_corpus():
	return render_template('collecter_corpus.html')

@app.route('/correction_erreur')
def correction_erreur():
	form = FlaskForm()
	return render_template('correction_erreur.html', form=form)

@app.route('/entites_nommees')
def entites_nommees():
	form = FlaskForm()
	return render_template('entites_nommees.html', form=form)

@app.route('/etiquetage_morphosyntaxique')
def etiquetage_morphosyntaxique():
	form = FlaskForm()
	err = ""
	return render_template('etiquetage_morphosyntaxique.html', form=form, err=err)

@app.route('/generate_corpus',  methods=["GET","POST"])
@stream_with_context
def generate_corpus():
	if request.method == 'POST':
		nb = int(request.form['nbtext'])
		all_texts = generate_random_corpus(nb)
		output_stream = StringIO()
		output_stream.write('\n\n\n'.join(all_texts))
		response = Response(output_stream.getvalue(), mimetype='text/plain',
							headers={"Content-disposition": "attachment; filename=corpus_wikisource.txt"})
		output_stream.seek(0)
		output_stream.truncate(0)
		return response
	return render_template('/collecter_corpus.html')

@app.route('/corpus_from_url',  methods=["GET","POST"])
@stream_with_context
def corpus_from_url():
	if request.method == 'POST':
		keys = request.form.keys()
		urls = [k for k in keys if k.startswith('url')]
		urls = sorted(urls)

		result_path, rand_name = createRandomDir('wiki_', 8)

		# PARCOURS DES URLS UTILISATEUR
		for url_name in urls:
			url = request.form.get(url_name)
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
						text = getWikiPage(link)
						if text != -1:
							if not name:
								name = path_elems[-1]
							with open(os.path.join(result_path, name), 'w') as output:
								output.write(text)

				except urllib.error.HTTPError:
					print(" ".join(["The page", url, "cannot be opened."]))
					continue

				filename = urllib.parse.unquote(path_elems[-1])

			# URL vers texte intégral
			else:
				try:
					clean_text = getWikiPage(url)
					if clean_text == -1:
						print("Erreur lors de la lecture de la page {}".format(url))

					else:
						if path_elems[-1] != 'Texte_entier':
							filename = urllib.parse.unquote(path_elems[-1])
						else:
							filename = urllib.parse.unquote(path_elems[-2])

						with open(os.path.join(result_path, filename), 'w') as output:
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


@app.route('/conversion_xml')
def conversion_xml():
	form = FlaskForm()
	return render_template('conversion_xml.html', form=form)

@app.route('/xmlconverter', methods=["GET", "POST"])
@stream_with_context
def xmlconverter():
	if request.method == 'POST':
		fields = {}

		f = request.files['file']
		fields['title'] = request.form['title'] # required
		fields['author'] = request.form.get('author')
		fields['respStmt_name'] = request.form.get('nameresp')
		fields['respStmt_resp'] = request.form.get('resp')
		fields['pubStmt'] = request.form['pubStmt'] # required
		fields['sourceDesc'] = request.form['sourceDesc'] # required

		filename = secure_filename(f.filename)
		path_to_file = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], filename)
		f.save(path_to_file)
		# Validating file format
		try:
			with open(path_to_file, "r") as f:
				for l in f:
					break;

			# Returning xml string
			root = txt_to_xml(path_to_file, fields)

			# Writing in stream
			output_stream = BytesIO()
			output = os.path.splitext(filename)[0] + '.xml'
			etree.ElementTree(root).write(output_stream, pretty_print=True, xml_declaration=True, encoding="utf-8")
			response = Response(output_stream.getvalue(), mimetype='application/xml',
								headers={"Content-disposition": "attachment; filename=" + output})
			output_stream.seek(0)
			output_stream.truncate(0)

		except UnicodeDecodeError:
			return 'format de fichier incorrect'

		return response

	return render_template("/conversion_xml")

@app.route('/autocorrect', methods=["GET", "POST"])
@stream_with_context
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

#-----------------------------------------------------------------
# FONCTIONS de traitement
#-----------------------------------------------------------------

# CONVERSION XML-TEI
# Construit un fichier TEI à partir des métadonnées renseignées dans le formulaire.
# Renvoie le chemin du fichier ainsi créé
# Paramètres :
# - filename : emplacement du fichier uploadé par l'utilisateur
# - fields : dictionnaire des champs présents dans le form metadata
def txt_to_xml(filename, fields):
	# Initialise TEI
	root = etree.Element("TEI")

	# TEI header
	teiHeader = etree.Element("teiHeader")
	fileDesc = etree.Element("fileDesc")
	titleStmt = etree.Element("titleStmt")
	editionStmt = etree.Element("editionStmt")
	publicationStmt = etree.Element("publicationStmt")
	sourceDesc = etree.Element("sourceDesc")

	#- TitleStmt
	#-- Title
	title = etree.Element("title")
	title.text = fields['title']
	titleStmt.append(title)

	#-- Author
	if fields['author']:
		author = etree.Element("author")
		author.text = fields['author']
		titleStmt.append(author)

	#- EditionStmt
	#-- respStmt
	if fields['respStmt_name']:
		respStmt = etree.Element("respStmt")
		name = etree.Element("name")
		name.text = fields['respStmt_name']
		respStmt.append(name)

		if fields['respStmt_resp']:
			resp = etree.Element("resp")
			resp.text = fields['respStmt_resp']
			respStmt.append(resp)

		editionStmt.append(respStmt)

	#- PublicationStmt
	publishers_list = fields['pubStmt'].split('\n') # Get publishers list
	publishers_list = list(map(str.strip, publishers_list)) # remove trailing characters
	publishers_list = [x for x in publishers_list if x] # remove empty strings
	for pub in publishers_list:
		publisher = etree.Element("publisher")
		publisher.text = pub
		publicationStmt.append(publisher)

	#- SourceDesc
	paragraphs = fields['sourceDesc'].split('\n')
	for elem in paragraphs:
		p = etree.Element('p')
		p.text = elem
		sourceDesc.append(p)

	# Header
	fileDesc.append(titleStmt)
	fileDesc.append(editionStmt)
	fileDesc.append(publicationStmt)
	fileDesc.append(sourceDesc)
	teiHeader.append(fileDesc)
	root.append(teiHeader)

	# Text
	text = etree.Element("text")

	with open(filename, "r") as f:
		for line in f:
			ptext = etree.Element('p')
			ptext.text = line
			text.append(ptext)

	root.append(text)
	return root
#-----------------------------------------------------------------
def generate_random_corpus(nb):

	# Read list of urls
	with open(ROOT_FOLDER / 'static/wikisource_bib.txt', 'r') as bib:
		random_texts = bib.read().splitlines()

	# Pick random urls
	urls = random.sample(random_texts, nb)
	all_texts = []

	for text_url in urls:
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
			clean_text = re.sub("[^\.:!?»[A-Z]]\n", ' ', text[0].text)
			all_texts.append(clean_text)

	return all_texts
#-----------------------------------------------------------------


@app.route('/pos_tagging', methods=["POST"])
@stream_with_context
def pos_tagging():
	form = FlaskForm()
	model_path = str(ROOT_FOLDER / os.path.join(app.config['MODEL_FOLDER'], 'sem_pos'))
	pipeline = sem.load(model_path)
	conllexporter = sem.exporters.CoNLLExporter()
	
	uploaded_files = request.files.getlist("uploaded_files")
	rand_name =  'postagging_' + ''.join((random.choice(string.ascii_lowercase) for x in range(5)))
	result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
	os.mkdir(result_path)

	#f = request.files['file']
	for f in uploaded_files:
		try:
			contenu = f.read()
			document = pipeline.process_text(contenu.decode("utf-8"))
			filename, file_extension = os.path.splitext(f.filename)
			output_name = filename + '_tokens.txt'  

			with open(ROOT_FOLDER / os.path.join(result_path, output_name), 'w', encoding="utf-8") as out:
				out.write(conllexporter.document_to_string(document, couples={"pos": "POS"}))
		finally:
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
	
	return render_template('etiquetage_morphosyntaxique.html', form=form, err="Une erreur est survenue dans le traitement des fichiers.")
	# Writing in stream
	#output_stream = BytesIO()
	#output = f.filename
	#output_stream.write(conllexporter.document_to_string(document, couples={"pos": "POS"}).encode("utf-8"))
	#response = Response(output_stream.getvalue(), mimetype='text/plain', headers={"Content-disposition": "attachment; filename=" + output})



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
					writer = csv.writer(output_name, delimiter="\t")
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

@app.route('/keyword_extraction', methods=['POST'])
@stream_with_context
def keyword_extraction():
	form = FlaskForm()
	if request.method == 'POST':
		uploaded_files = request.files.getlist("keywd-extract")
		methods = request.form.getlist('extraction-method')
		res = {}
		
		if uploaded_files == []:
			abort(400)

		from keybert import KeyBERT
		from sentence_transformers import SentenceTransformer
		from pathlib import Path

		# Chargement du modèle
		sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
		kw_model = KeyBERT(model=sentence_model)

		# Le résultat est stocké dans un dictionnaire. 
		# Clé : nom du fichier (string)
		# Valeur : dictionnaire dont la clé est la méthode d'extraction et la valeur une liste de mots-clés
		
		for f in uploaded_files:
			fname = Path(f.filename).stem
			fname = fname.replace(' ', '_')
			fname = fname.strip()
			fname = "".join(x for x in fname if (x.isalnum() or x == '_'))
			
			res[fname] = {}
			text = f.read().decode("utf-8")
			
			if 'default' in methods:
				keywords_def = kw_model.extract_keywords(text)
				res[fname]['default'] = keywords_def

			if 'mmr' in methods:
				diversity = int(request.form.get('diversity')) / 10
				keywords_mmr = kw_model.extract_keywords(text, use_mmr=True, diversity=diversity)
				res[fname]['mmr'] = keywords_mmr

			if 'mss' in methods:
				keywords_mss = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), use_maxsum=True, nr_candidates=10, top_n=3)
				res[fname]['mss'] = keywords_mss
		
		return Response(response=render_template('extraction_mots_cles.html', form=form, res=res))
		
	return render_template('extraction_mots_cles.html', form=form, res=res)


@app.route('/topic_extraction', methods=["POST"])
@stream_with_context
def topic_extraction():
	form = FlaskForm()
	msg = ""
	res = {}
	if request.method == 'POST':
		uploaded_files = request.files.getlist("topic_model")
		if uploaded_files == []:
			abort(400)
		if len(uploaded_files) == 1:
			text = uploaded_files[0].read().decode("utf-8")
			if len(text) < 4500:
				return Response(response=render_template('topic_modelling.html', form=form, res=res, msg="Le texte est trop court, merci de charger un corpus plus grand pour des résultats significatifs. A défaut, vous pouvez utiliser l'outil d'extraction de mot-clés."))

		# Topic modelling
		from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
		from sklearn.decomposition import NMF, LatentDirichletAllocation
		from pathlib import Path
		import numpy as np

		# Loading stop words
		with open(ROOT_FOLDER / os.path.join(app.config['UTILS_FOLDER'], "stop_words_fr.txt"), 'r', encoding="utf-8") as sw :
			stop_words_fr = sw.read().splitlines()
		
		# Form options
		methods = request.form.getlist('modelling-method')
		lemma_state = request.form.getlist('lemma-opt')

		# Loading corpus
		corpus = []
		max_f = 0

		# If one file is uploaded, we split it in 3 chunks to be able to retrieve more than 1 topic cluster.
		if len(uploaded_files) == 1:
			sents = sentencizer(text)
			chunks = [x.tolist() for x in np.array_split(sents, 3)]
			total_tokens = set()
			for l in chunks:
				if lemma_state:
					txt_part = spacy_lemmatizer("\n".join(l))
				else:
					txt_part = "\n".join(l)
				
				corpus.append(txt_part)
				
				# Compute corpus size
				total_tokens.update(set(txt_part.split(' ')))
			
			# Number of topics when corpus xxs
			no_topics = 2
			
		else:
			total_tokens = set()
			for f in uploaded_files:
				text = f.read().decode("utf-8")
				if lemma_state:
					text = spacy_lemmatizer(text)
				
				corpus.append(text)

				# Compute corpus size
				total_tokens.update(set(text.split(' ')))

				# Nb topics = nb fichiers
				if len(uploaded_files) > 8:
					no_topics = 8
				else:
					no_topics = len(uploaded_files)
				
		# Taille du corpus
		max_f = len(total_tokens)
		print("Nb types : ".format(max_f))

		# Number of terms included in the bag of word matrix
		no_features = int(max_f - (10 * max_f / 100))
		
		no_top_words = 5

		# Topic extraction
		if 'nmf' in methods:
			tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, max_features=no_features, stop_words=stop_words_fr)
			tfidf = tfidf_vectorizer.fit_transform(corpus)
			tfidf_feature_names = tfidf_vectorizer.get_feature_names()

			# Parameter: nndsvda for less sparsity ; else nndsvd
			nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda').fit(tfidf)
			
			res_nmf = display_topics(nmf, tfidf_feature_names, no_top_words)
			res['nmf'] = res_nmf


		if 'lda' in methods:
			tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=no_features, stop_words=stop_words_fr)
			tf = tf_vectorizer.fit_transform(corpus)
			tf_feature_names = tf_vectorizer.get_feature_names()

			lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
			
			res_lda = display_topics(lda, tf_feature_names, no_top_words)
			res['lda'] = res_lda
		
		return Response(response=render_template('topic_modelling.html', form=form, res=res, msg=msg))

	return render_template('topic_modelling.html', form=form, res=res, msg=msg)

#-----------------------------------------------------------------
@app.route('/extract_gallica', methods=["GET", "POST"])
@stream_with_context
def extract_gallica():
	form = FlaskForm()
	input_format = request.form['input_format']
	res_ok = ""
	res_err = ""

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

			print("Ark et plages détectés : {}\t{}".format(arkName, suffixe))
		
		# Cas 2 : on télécharge tout le document
		else:
			arkName = elems[0].strip()
			debut = 1
			nb_p = 0
			suffixe = ''
			print(arkName)

		if input_format == 'txt':
			url = 'https://gallica.bnf.fr/ark:/12148/{}{}.texteBrut'.format(arkName, suffixe)
			outfile = arkName + '.txt'
			path_file = os.path.join(result_path, outfile)
			
		
		elif input_format == 'img':
			# Parcours des pages à télécharger
			for i in range(int(debut), int(debut) + int(nb_p)):
				taille = get_size(arkName, i)
				largeur = taille["width"]
				hauteur = taille["height"]
				url = "https://gallica.bnf.fr/iiif/ark:/12148/{}/f{}/{},{},{},{}/full/0/native.jpg".format(arkName, i, 0, 0, largeur, hauteur)
				outfile = "{}_{:04}.jpg".format(arkName, i)
				path_file = os.path.join(result_path, outfile)

		else:
			print("Erreur de paramètre")
			abort(400)
	
	try:
		with urllib.request.urlopen(url) as response, open(os.path.join(result_path, outfile), 'wb') as out_file:
			shutil.copyfileobj(response, out_file)
			
		res_ok += url + '\n'
	except Exception as exc:
		#print('\nFAILURE - download ({}) - Exception raised: {}'.format(url, exc))
		res_err += url + '\n'
	
	with open(os.path.join(result_path, 'download_report.txt'), 'w') as report:
		if res_err != "":
			report.write("Erreur de téléchargement pour : \n {}".format(res_err))
		else:
			report.write("{} documents ont bien été téléchargés.\n".format(len(arks_list)))
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
	split_regex=['\.{1,}','\!+','\?+']
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
		clean_text = re.sub("[^\.:!?»[A-Z]]\n", ' ', text[0].text)
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
# chaînes de traitement
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
		contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
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
		contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
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
	# paramètres OCR
	ocr_model = request.form['tessmodel']
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

	if ocr_model != "raw_text":
		rand_name =  'ocr_ner_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8)))
		contenu = ocr.tesseract_to_txt(uploaded_files, ocr_model, rand_name, ROOT_FOLDER, up_folder)
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
	# 	print(key, value, file=sys.stderr)

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

@app.route("/run_renard",  methods=["GET", "POST"])
@stream_with_context
def run_renard():
	form = FlaskForm()
	if request.method == 'POST':
		if request.files['renard_upload'].filename != '':
			f = request.files['renard_upload']

			text = f.read()
		else:
			text = request.form['renard_txt_input']
		
		rand_name =  'renard_graph_' + ''.join((random.choice(string.ascii_lowercase) for x in range(8))) + '.png'
		result_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], rand_name)
		
		from renard.pipeline import Pipeline
		from renard.pipeline.tokenization import NLTKTokenizer
		from renard.pipeline.ner import NLTKNamedEntityRecognizer
		from renard.pipeline.characters_extraction import NaiveCharactersExtractor
		from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
		import matplotlib.pyplot as plt
		import networkx as nx
		from base64 import b64encode
		from PIL import Image
		import io
		import numpy as np
		import cv2

		pipeline = Pipeline(
		[
			NLTKTokenizer(),
			NLTKNamedEntityRecognizer(),
			NaiveCharactersExtractor(),
			CoOccurrencesGraphExtractor(co_occurences_dist=35)
		])

		out = pipeline(text.decode('utf-8'))
		#out.export_graph_to_gexf(result_path)
		
		out.plot_graph()
		plt.savefig(result_path)
		with open(result_path, 'rb') as local_img:
			f = local_img.read()
		
		npimg = np.fromstring(f,np.uint8)
		img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
		img = Image.fromarray(img.astype("uint8"))
		rawBytes = io.BytesIO()
		img.save(rawBytes, "PNG")
		rawBytes.seek(0)
		img_base64 = b64encode(rawBytes.getvalue()).decode('ascii')
		mime = "image/png"
		uri = "data:%s;base64,%s"%(mime, img_base64)
		
		return render_template('renard.html', form=form, graph=uri)

	return render_template('renard.html', form=form, graph="")

if __name__ == "__main__":
	app.run()
