#!/usr/bin/env python3
"""description:
    Annote un fichier TEI avec un moteur de reconnaissance d'entités nommées.
    Ce script utilise la structure du XML pour trouver le contenu textuel du
    fichier. De base, ce script itère sur les balises <p> à l'intérieur de la
    balise <text>. Le script récupère le contenu textuel de chaque balise et
    annote avec un outil au choix pour ensuite ajouter des balise <Entity> sur
    les mentions trouvées dans le texte.

    L'utilisation de cette balise <Entity> rend utilisable la sortie de ce
    script dans la plateforme Ariane.

    ATTENTION : ce script détruit détruit la structuration et le formattage XML
    à l'intérieur des balises parcourues. En effet, lorsque le contenu textuel
    est récupéré, tout formattage présent dans le texte est ignoré.

exemples d'utilisation:
    python ./tei_ner.py -h
    python ./tei_ner.py entree.xml.tei sortie.xml.tei
    python ./tei_ner.py entree.xml.tei sortie.xml.tei -a spacy -m fr_core_news_lg -b s
    python ./tei_ner.py entree.xml.tei sortie.xml.tei -a flair -m "flair/ner-french"
"""

import pathlib
import functools
from lxml import etree
from io import BytesIO

import spacy

from flair.data import Sentence as FlairSentence
from flair.models import SequenceTagger


def flair_annotate(sentence, modele):
    s = FlairSentence(sentence)
    modele.predict(s)
    return s


def get_label_function(annotateur_name, annotateur):
    if annotateur_name == "flair":
        return functools.partial(flair_annotate, modele=annotateur)

    if annotateur_name == "spacy":
        return annotateur.__call__

    raise KeyError(f"{annotateur_name} n'a pas de fonction d'annotation connue")


def spacy_iterate(doc):
    for entity in doc.ents:
        yield (entity.label_, entity.start_char, entity.end_char)


def flair_iterate(doc):
    for entity in doc.get_spans('ner'):
        try:
            yield (entity.tag, entity.start_position, entity.end_position)
        except AttributeError:
            yield (entity.tag, entity.start_pos, entity.end_pos)


loaders = {
    "spacy": spacy.load,
    "flair": SequenceTagger.load,
}

entity_iterators = {
    "spacy": spacy_iterate,
    "flair": flair_iterate,
}


def tei_ner_params(contenu, xmlnamespace, racine, balise, moteur, modele, encodage="utf-8"):
    moteur = moteur.lower()
    loader = loaders.get(moteur)
    iterator = entity_iterators.get(moteur)

    if loader is None:
        raise ValueError(f"Pas de chargeur de modèle pour {moteur}")

    if iterator is None:
        raise ValueError(f"Pas d'itérateur d'entités pour {moteur}")

    pipeline = loader(modele)
    label_function = get_label_function(moteur, pipeline)
    try:
        tree = etree.parse(BytesIO(contenu))
        return tei_ner(tree, xmlnamespace, racine, balise, label_function, iterator, encodage=encodage)
    except etree.XMLSyntaxError:
        print("Erreur de syntaxe dans le fichier XML.")
        pass


def tei_ner(arbre, xmlns, racine, balise, annotateur, iterateur, encodage="utf-8"):
    """Annote un fichier TEI avec un moteur de reconnaissance d'entités nommées.
    Renvoie un objet XML (lxml.etree.ElementTree). Tout formattage du texte
    (ex: italique, gras, etc.) sera perdu au cours du processus.

    Parameters
    ----------
    arbre : lxml.etree.ElementTree
        Le XML d'entrée
    racine : str
        Le type de balise XML à utiliser comme racine pour parcourir le XML
    balise : str
        Le type de balise XML contenant le texte
    annotateur : function(str) -> object
        Le moteur d'annotation à utiliser
    iterateur : Iterable
        la fonction d'itération pour parcourir les entités nommées
    encodage : str, "utf-8"
        le nom de l'encodage à utiliser pour le texte

    Returns
    -------
    xml : lxml.etree.ElementTree
        L'arbre XML enrichi des annotations en entités nommées fournies par le
        moteur.
    """

    #xmlns = "http://www.tei-c.org/ns/1.0"

    textnode = next(arbre.iterfind(f".//{{{xmlns}}}{racine}"))
    textnode.iterfind(f".//{{{xmlns}}}{balise}")

    for node in textnode.iterfind(f".//{{{xmlns}}}{balise}"):
        # get all the text content of node. This will remove any existing XML
        # formatting as we will provide annotations for the text and replace the
        # content of the XML, so beware.
        text = etree.tostring(node, method="text", encoding=encodage).decode(encodage).strip()

        node.clear()

        prev = 0
        previous_node = None
        for label, start, end in iterateur(annotateur(text)):
            if prev == 0:
                node.text = text[prev: start]
            else:
                previous_node.tail = text[prev: start]
            node.append(etree.Element("Entity"))
            previous_node = node[-1]
            previous_node.attrib["annotation"] = label
            previous_node.text = text[start : end]
            prev = end

        # the only way previous_node is None is when no entities are found
        if previous_node is not None:
            previous_node.tail = text[end:]
        else:
            node.text = text

    return arbre


def main(
    fichier,
    sortie,
    racine="text",
    balise="p",
    annotateur="spacy",
    modele="fr_core_news_md",
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
    try:
        tree = etree.parse(fichier)
        tree = tei_ner(tree, racine, balise, label_function, iterator, encodage=encodage)
        with open(sortie, "w", encoding="utf-8") as output_stream:
            output_stream.write(
                etree.tostring(tree, pretty_print=True, encoding="utf-8").decode("utf-8")
            )
    except etree.XMLSyntaxError:
        pass




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("fichier", help="Le fichier TEI à annoter")
    parser.add_argument("sortie", help="Le fichier TEI à écrire")
    parser.add_argument(
        "-r",
        "--racine",
        default="text",
        help="Le type de balise à utiliser comme racine (par défaut: 'text')"
    )
    parser.add_argument(
        "-b",
        "--balise",
        default="p",
        help="Le type de balise à utiliser pour extraire le texte (par défaut: 'p')"
    )
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
        help="Le modèle à utiliser par l'annotateur (défaut : fr_core_news_md)"
    )
    parser.add_argument(
        "-e",
        "--encodage",
        default="utf-8",
        help="L'encodage à utiliser (défault: utf-8)"
    )

    args = parser.parse_args()

    main(**vars(args))
