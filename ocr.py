import os
import random
import string
import subprocess
import glob
from io import StringIO
import shutil
import re

from werkzeug.utils import secure_filename

def tesseract_to_txt(uploaded_files, model, rand_name, ROOT_FOLDER, UPLOAD_FOLDER):
    # Nom de dossier aléatoire pour le résultat de la requête
    result_path = ROOT_FOLDER / os.path.join(UPLOAD_FOLDER, rand_name)
    os.mkdir(result_path)

    # Répertoire de travail pour les fichiers pdf
    directory_path = ''

    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']

    output_stream = StringIO()
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
            directory_path = ROOT_FOLDER / os.path.join(UPLOAD_FOLDER, directory)
            try:
                os.mkdir(directory_path)
            except FileExistsError:
                pass

            # Sauvegarde du PDF
            path_to_file = ROOT_FOLDER / os.path.join(directory_path, secure_filename(f.filename))
            f.save(path_to_file)

            # Conversion en PNG
            subprocess.run(['pdftoppm', '-r', '180', path_to_file, os.path.join(directory_path, filename), '-png'])	# Bash : pdftoppm -r 180 fichier.pdf fichier -png

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
                    output_stream.write(ftxt.read())
                    output_stream.write('\n\n')

        #------------------------------------------------------
        # Fichier image
        #------------------------------------------------------
        elif file_extension.lower() in extensions:
            # Sauvegarde de l'image
            path_to_file = ROOT_FOLDER / os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
            f.save(path_to_file)

            # Tesseract
            output_txt = os.path.join(result_path, os.path.splitext(f.filename)[0])
            subprocess.run(['tesseract', '-l', model, path_to_file, output_txt])

            with open(output_txt + '.txt', 'r', encoding="utf-8") as ftxt:
                output_stream.write(ftxt.read())
                output_stream.write('\n\n')

            # Nettoyage de l'image
            os.remove(path_to_file)
            os.remove(output_txt+".txt")

        else:
            raise Exception("Le fichier {} n'a pas d'extension ou a une extension invalide.".format(filename))

    final_text = output_stream.getvalue()
    output_stream.seek(0)
    output_stream.truncate(0)

    # Nettoie le dossier de travail
    if directory_path != '':
        shutil.rmtree(directory_path)
    shutil.rmtree(result_path)

    return final_text
