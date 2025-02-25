"""description:
    Annote un fichier txt avec un moteur de reconnaissance d'entités nommées.
    La sortie de ce script est au format BRAT

"""

import pathlib
import functools
from io import BytesIO
import csv

import spacy

from flair.data import Sentence as FlairSentence
from flair.models import SequenceTagger


def flair_annotate(sentence, modele):
    try:
        s = FlairSentence(sentence)
        print("Successfully created Flair sentence")
        print(f"Predicting with model: {modele}")
        modele.predict(s)
        print("Successfully made prediction")
        return s
    except Exception as e:
        print(f"Error in flair_annotate: {str(e)}")
        raise



def get_label_function(annotateur_name, annotateur):
    print(f"Getting label function for {annotateur_name}")
    if annotateur_name == "flair":
        try:
            return functools.partial(flair_annotate, modele=annotateur)
        except Exception as e:
            print(f"Error creating Flair label function: {str(e)}")
            raise

    if annotateur_name == "spacy":
        return annotateur.__call__

    raise KeyError(f"{annotateur_name} n'a pas de fonction d'annotation connue")


def spacy_iterate(doc):
    for entity in doc.ents:
        yield (entity.label_, entity.start_char, entity.end_char)


def flair_iterate(doc):
    print(f"Starting Flair iteration on document")
    try:
        for entity in doc.get_spans('ner'):
            try:
                yield (entity.tag, entity.start_position, entity.end_position)
            except AttributeError:
                try:
                    yield (entity.tag, entity.start_pos, entity.end_pos)
                except Exception as e:
                    print(f"Error accessing entity positions: {str(e)}")
                    raise
    except Exception as e:
        print(f"Error in Flair iteration: {str(e)}")
        raise



loaders = {
    "spacy": spacy.load,
    "flair": SequenceTagger.load,
}

entity_iterators = {
    "spacy": spacy_iterate,
    "flair": flair_iterate,
}


def txt_ner_params(texte, moteur, modele, encodage="utf-8"):
    moteur = moteur.lower()
    loader = loaders.get(moteur)
    iterator = entity_iterators.get(moteur)

    if loader is None:
        raise ValueError(f"Pas de chargeur de modèle pour {moteur}")

    if iterator is None:
        raise ValueError(f"Pas d'itérateur d'entités pour {moteur}")

    print(f"Attempting to load {moteur} model: {modele}")
    try:
        pipeline = loader(modele)
        print(f"Successfully loaded model: {pipeline}")
    except Exception as e:
        print(f"Error loading model {modele}: {str(e)}")
        raise

    label_function = get_label_function(moteur, pipeline)
    try:
        contenu = texte.decode(encodage)
    except AttributeError:
        contenu = texte
        print("Erreur dans la spécification de l'encodage.")
    return txt_ner(contenu, label_function, iterator, encodage=encodage)

def txt_ner(texte, annotateur, iterateur, encodage="utf-8"):
    """Annote un fichier TEI avec un moteur de reconnaissance d'entités nommées.
    Renvoie un objet XML (lxml.etree.ElementTree). Tout formattage du texte
    (ex: italique, gras, etc.) sera perdu au cours du processus.

    Parameters
    ----------
    texte : str
        La chaîne d'entrée
    annotateur : function(str) -> object
        Le moteur d'annotation à utiliser
    iterateur : Iterable
        la fonction d'itération pour parcourir les entités nommées
    encodage : str, "utf-8"
        le nom de l'encodage à utiliser pour le texte

    Returns
    -------
    entities : list[[str, int, int, str]]
        La liste des entités trouvées par le moteur. Contient les champs :
        - type
        - offset début
        - offset fin
        - texte
    """

    entities = []
    for label, start, end in iterateur(annotateur(texte)):
        entities.append([label, start, end, texte[start:end]])
    return entities


def main(
    fichier,
    sortie,
    racine="text",
    balise="p",
    annotateur="spacy",
    modele="fr_core_news_lg",
    encodage="utf-8"
):
    inputpath = pathlib.Path(fichier)
    outputpath = pathlib.Path(sortie)

    if outputpath.exists() and inputpath.samefile(outputpath):
        raise ValueError("Les fichiers d'entrée et de sortie sont identiques")

    loader = loaders.get(annotateur)
    iterator = entity_iterators.get(annotateur)

    if loader is None:
        raise ValueError(f"Pas de chargeur de modèle pour {annotateur}")

    if iterator is None:
        raise ValueError(f"Pas d'itérateur d'entités pour {annotateur}")

    pipeline = loader(modele)
    label_function = get_label_function(annotateur, pipeline)
    print(label_function)

    """with open(fichier) as input_stream:
        contenu = input_stream.read()"""
    try:
        input_stream = open(fichier, 'r')
    except IOError:
        print('Erreur en lisant le fichier à analyser.')
    else:
        with input_stream:
            contenu = input_stream.read()

    with open(sortie, "w", encoding="utf-8") as output_stream:
        writer = csv.writer(output_stream, delimiter="\t")
        for nth, entity in enumerate(txt_ner(contenu, label_function, iterator), 1):
            ne_type, start, end, text = entity
            row = [f"T{nth}", f"{ne_type} {start} {end}", f"{text}"]
            writer.writerow(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("fichier", help="Le fichier TEI à annoter")
    parser.add_argument("sortie", help="Le fichier TEI à écrire")
    parser.add_argument(
        "-a",
        "--annotateur",
        choices=("spacy", "flair"),
        default="spacy",
        help="Le moteur d'annotation à utiliser (défaut: spacy)"
    )
    parser.add_argument(
        "-m",
        "--modele",
        default="fr_core_news_sm",
        help="Le modèle à utiliser par l'annotateur (défaut : fr_core_news_lg)"
    )
    parser.add_argument(
        "-e",
        "--encodage",
        default="utf-8",
        help="L'encodage à utiliser (défault: utf-8)"
    )

    args = parser.parse_args()

    main(**vars(args))
