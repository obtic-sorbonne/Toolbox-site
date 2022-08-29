#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from flask import Flask, request, render_template, url_for, redirect, send_from_directory, Response, stream_with_context, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
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
import sys
import shutil
import subprocess
import glob
from pathlib import Path
import jamspell

import sem
import sem.storage
import sem.exporters

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'static/models'
ROOT_FOLDER = Path(__file__).parent.absolute()

app = Flask(__name__)

# App config
app.config['SESSION_TYPE'] = 'filesystem'
#app.config['SECRET_KEY'] = 'pakisqa'

app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024 # Limit file upload to 8MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)

#-----------------------------------------------------------------
# ROUTES
#-----------------------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/outils_corpus')
def outils_corpus():
	return render_template('layouts/corpus.html')

@app.route('/outils_fouille')
def outils_fouille():
	return render_template('layouts/fouille_de_texte.html')

@app.route('/outils_visualisation')
def outils_visualisation():
	return render_template('layouts/visualisation.html')

@app.route('/numeriser')
def numeriser():
	return render_template('layouts/numeriser.html')

@app.route('/normalisation')
def normalisation():
	return render_template('normalisation.html')

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
# NUMERISATION TESSERACT
#-----------------------------------------------------------------
@app.route('/run_tesseract',  methods=["GET","POST"])
@stream_with_context
def run_tesseract():
	if request.method == 'POST':
		uploaded_files = request.files.getlist("tessfiles")
		model = request.form['tessmodel']

		# Nom de dossier aléatoire pour le résultat de la requête
		result_path, rand_name = createRandomDir('ocr_', 8)

		# Répertoire de travail pour les fichiers pdf
		directory_path = ''

		extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']

		for f in uploaded_files:
			filename, file_extension = os.path.splitext(f.filename)
			output_name = filename + '.txt'
			#------------------------------------------------------
			# Fichier pdf
			#------------------------------------------------------
			# A FAIRE : le traitement des gros fichiers doit être parallélisé
			if file_extension.lower() == ".pdf":
				# Créer un dossier pour stocker l'ensemble des images
				directory = filename + '_temp'
				directory_path = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], directory)
				try:
					os.mkdir(directory_path)
				except FileExistsError:
					pass

				# Sauvegarde du PDF
				path_to_file = ROOT_FOLDER / os.path.join(directory_path, secure_filename(f.filename))
				f.save(path_to_file)

				# Conversion en PNG
				subprocess.run(['pdftoppm', '-r', '180', path_to_file, os.path.join(directory_path, filename), '-png'])	# Bash : pdftoppm -r 180 fichier.pdf fichier -png

				# OCRisation
				png_list = glob.glob(str(directory_path) + '/*.png')
				final_output = ""

				if len(png_list) > 1:
					png_list.sort(key=lambda f: int(re.sub('\D', '', f)))

				for png_file in png_list:
					output_txt = os.path.splitext(png_file)[0]
					try:
						subprocess.run(['tesseract', '-l', model, png_file, output_txt])
					except:
						raise Exception("Tesseract a rencontré un problème lors de la lecture du fichier {}".format(filename))

					with open(output_txt + '.txt', 'r', encoding="utf-8") as ftxt:
						final_output += ftxt.read()
						final_output += '\n\n'

				# Ecriture du résultat
				with open(ROOT_FOLDER / os.path.join(result_path, filename + '.txt'), 'w', encoding="utf-8") as out:
					out.write(final_output)

			#------------------------------------------------------
			# Fichier image
			#------------------------------------------------------
			elif file_extension.lower() in extensions:
				# Sauvegarde de l'image
				path_to_file = ROOT_FOLDER / os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
				f.save(path_to_file)

				# Tesseract
				output_txt = os.path.join(result_path, os.path.splitext(f.filename)[0])
				subprocess.run(['tesseract', '-l', model, path_to_file, output_txt])

				# Nettoyage de l'image
				os.remove(path_to_file)

			else:
				raise Exception("Le fichier {} n'a pas d'extension ou a une extension invalide.".format(filename))


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
		if directory_path != '':
			shutil.rmtree(directory_path)
		shutil.rmtree(result_path)

		return response

	return render_template('layouts/numeriser.html', erreur=erreur)

@app.route('/creer_corpus')
def creer_corpus():
	return render_template('creer_corpus.html')

@app.route('/correction_erreur')
def correction_erreur():
	return render_template('correction_erreur.html')

@app.route('/entites_nommees')
def entites_nommees():
	return render_template('entites_nommees.html')

@app.route('/etiquetage_morphosyntaxique')
def etiquetage_morphosyntaxique():
	return render_template('etiquetage_morphosyntaxique.html')

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
	return render_template('/creer_corpus.html')

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
					page = urllib.request.urlopen(url)
					soup = BeautifulSoup(page, 'html.parser')
					text = soup.findAll("div", attrs={'class': 'prp-pages-output'})
					if len(text) == 0:
						print("This does not appear to be part of the text (no prp-pages-output tag at this location).")
						with open('pb_url.log', 'a') as err_log:
							err_log.write(text_url)
					else:
						# Remove end of line inside sentence
						clean_text = re.sub("[^\.:!?»[A-Z]]\n", ' ', text[0].text)
						if path_elems[-1] != 'Texte_entier':
							filename = urllib.parse.unquote(path_elems[-1])
						else:
							filename = urllib.parse.unquote(path_elems[-2])

						with open(result_path + filename, 'w') as output:
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

	return render_template('creer_corpus.html')


@app.route('/conversion_xml')
def conversion_xml():
	return render_template('conversion_xml.html')

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

		if fields['resp']:
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
	path = str(ROOT_FOLDER / os.path.join(app.config['MODEL_FOLDER'], 'sem_pos'))
	pipeline = sem.load(path)
	conllexporter = sem.exporters.CoNLLExporter()
	f = request.files['file']
	try:
		contenu = f.read()
	finally: # ensure file is closed
		f.close()
	document = pipeline.process_text(contenu.decode("utf-8"))
	# Writing in stream
	output_stream = BytesIO()
	output = f.filename
	output_stream.write(conllexporter.document_to_string(document, couples={"pos": "POS"}).encode("utf-8"))
	response = Response(output_stream.getvalue(), mimetype='text/plain',
						headers={"Content-disposition": "attachment; filename=" + output})
	output_stream.seek(0)
	output_stream.truncate(0)
	return response


@app.route('/named_entity_recognition', methods=["POST"])
@stream_with_context
def named_entity_recognition():
	from tei_ner import tei_ner_params
	from lxml import etree
	f = request.files['file']
	balise_racine = request.form['balise_racine']
	balise_parcours = request.form['balise_parcours']
	encodage = request.form['encodage']
	moteur_REN = request.form['moteur_REN']
	modele_REN = request.form['modele_REN']
	try:
		contenu = f.read()
	finally: # ensure file is closed
		f.close()
	root = tei_ner_params(contenu, balise_racine, balise_parcours, moteur_REN, modele_REN, encodage=encodage)
	# Writing in stream
	output_stream = BytesIO()
	output = f.filename
	root.write(output_stream, pretty_print=True, xml_declaration=True, encoding="utf-8")
	response = Response(output_stream.getvalue(), mimetype='application/xml',
						headers={"Content-disposition": "attachment; filename=" + output})
	output_stream.seek(0)
	output_stream.truncate(0)
	return response

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
		return clean_text

if __name__ == "__main__":
	app.run()
