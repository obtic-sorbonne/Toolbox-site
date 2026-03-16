#!/usr/bin/env python3
"""
Annote un fichier TEI avec un moteur de reconnaissance d'entités nommées.

Deux modes possibles :

tei :
    conserve la structure TEI et ajoute
        persName
        orgName
        placeName

ariane :
    ancien comportement avec
        <Entity annotation="...">
"""

import pathlib
import functools
from lxml import etree

import spacy
from flair.data import Sentence as FlairSentence
from flair.models import SequenceTagger


# ----------------------------
# NER wrappers
# ----------------------------

def flair_annotate(sentence, modele):
    s = FlairSentence(sentence)
    modele.predict(s)
    return s


def get_label_function(annotateur_name, annotateur):

    if annotateur_name == "flair":
        return functools.partial(flair_annotate, modele=annotateur)

    if annotateur_name == "spacy":
        return annotateur.__call__

    raise KeyError(f"{annotateur_name} inconnu")


def spacy_iterate(doc):
    for entity in doc.ents:
        yield (entity.label_, entity.start_char, entity.end_char)


def flair_iterate(doc):

    for entity in doc.get_spans("ner"):

        try:
            yield (entity.tag, entity.start_position, entity.end_position)

        except AttributeError:
            yield (entity.tag, entity.start_pos, entity.end_pos)


loaders = {
    "spacy": spacy.load,
    "flair": SequenceTagger.load
}

entity_iterators = {
    "spacy": spacy_iterate,
    "flair": flair_iterate
}


# ----------------------------
# TEI label mapping
# ----------------------------

LABEL_MAP = {
    "PER": "persName",
    "PERSON": "persName",
    "ORG": "orgName",
    "LOC": "placeName",
    "GPE": "placeName",
    "PLACE": "placeName"
}


# ----------------------------
# Helpers (mode TEI)
# ----------------------------

def annotate_text_fragment(text, annotateur, iterateur):

    entities = list(iterateur(annotateur(text)))

    if not entities:
        return [text]

    result = []
    prev = 0

    for label, start, end in entities:

        if start > prev:
            result.append(text[prev:start])

        tag = LABEL_MAP.get(label)

        if tag:

            el = etree.Element(tag)
            el.text = text[start:end]
            result.append(el)

        else:
            result.append(text[start:end])

        prev = end

    if prev < len(text):
        result.append(text[prev:])

    return result


def inject_fragment(parent, fragments, is_tail=False, ref_node=None):

    previous = None

    for item in fragments:

        if isinstance(item, str):

            if previous is None:

                if is_tail:
                    ref_node.tail = (ref_node.tail or "") + item
                else:
                    parent.text = (parent.text or "") + item

            else:
                previous.tail = (previous.tail or "") + item

        else:

            if is_tail:
                parent.insert(parent.index(ref_node) + 1, item)

            else:
                parent.insert(
                    0 if previous is None else parent.index(previous) + 1,
                    item
                )

            previous = item


# ----------------------------
# Mode TEI (préserve balises)
# ----------------------------

def tei_ner_tei(arbre, xmlns, racine, balise, annotateur, iterateur):

    textnode = next(arbre.iterfind(f".//{{{xmlns}}}{racine}"))

    for node in textnode.iterfind(f".//{{{xmlns}}}{balise}"):

        if node.text:

            fragments = annotate_text_fragment(node.text, annotateur, iterateur)

            node.text = None
            inject_fragment(node, fragments)

        for child in list(node):

            if child.tail:

                fragments = annotate_text_fragment(child.tail, annotateur, iterateur)

                child.tail = None
                inject_fragment(node, fragments, is_tail=True, ref_node=child)

    return arbre


# ----------------------------
# Mode Ariane (ancien script)
# ----------------------------

def tei_ner_ariane(arbre, xmlns, racine, balise, annotateur, iterateur, encodage="utf-8"):

    textnode = next(arbre.iterfind(f".//{{{xmlns}}}{racine}"))

    for node in textnode.iterfind(f".//{{{xmlns}}}{balise}"):

        text = etree.tostring(node, method="text", encoding=encodage).decode(encodage).strip()

        node.clear()

        prev = 0
        previous_node = None

        for label, start, end in iterateur(annotateur(text)):

            if prev == 0:
                node.text = text[prev:start]

            else:
                previous_node.tail = text[prev:start]

            node.append(etree.Element("Entity"))

            previous_node = node[-1]
            previous_node.attrib["annotation"] = label
            previous_node.text = text[start:end]

            prev = end

        if previous_node is not None:
            previous_node.tail = text[end:]

        else:
            node.text = text

    return arbre

def ner_tei_params(
    contenu,
    xmlnamespace,
    balise_racine,
    balise_parcours,
    moteur_REN,
    modele_REN,
    mode="tei",
    encodage="utf-8"
):

    loader = loaders.get(moteur_REN)
    iterator = entity_iterators.get(moteur_REN)

    pipeline = loader(modele_REN)

    label_function = get_label_function(moteur_REN, pipeline)

    tree = etree.ElementTree(etree.fromstring(contenu))

    if mode == "ariane":

        tree = tei_ner_ariane(
            tree,
            xmlnamespace,
            balise_racine,
            balise_parcours,
            label_function,
            iterator,
            encodage
        )

    else:

        tree = tei_ner_tei(
            tree,
            xmlnamespace,
            balise_racine,
            balise_parcours,
            label_function,
            iterator
        )

    return tree

# ----------------------------
# MAIN
# ----------------------------

def main(
    fichier,
    sortie,
    racine="text",
    balise="p",
    annotateur="spacy",
    modele="fr_core_news_md",
    mode="tei",
    encodage="utf-8"
):

    inputpath = pathlib.Path(fichier)
    outputpath = pathlib.Path(sortie)

    if outputpath.exists() and inputpath.samefile(outputpath):
        raise ValueError("Les fichiers d'entrée et de sortie sont identiques")

    loader = loaders.get(annotateur)
    iterator = entity_iterators.get(annotateur)

    pipeline = loader(modele)

    label_function = get_label_function(annotateur, pipeline)

    xmlns = "http://www.tei-c.org/ns/1.0"

    tree = etree.parse(fichier)

    if mode == "ariane":

        tree = tei_ner_ariane(
            tree,
            xmlns,
            racine,
            balise,
            label_function,
            iterator
        )

    else:

        tree = tei_ner_tei(
            tree,
            xmlns,
            racine,
            balise,
            label_function,
            iterator
        )

    with open(sortie, "w", encoding="utf-8") as output_stream:

        output_stream.write(
            etree.tostring(
                tree,
                pretty_print=True,
                encoding="utf-8"
            ).decode("utf-8")
        )


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fichier")
    parser.add_argument("sortie")

    parser.add_argument("-r", "--racine", default="text")

    parser.add_argument("-b", "--balise", default="p")

    parser.add_argument(
        "-a",
        "--annotateur",
        choices=("spacy", "flair"),
        default="spacy"
    )

    parser.add_argument(
        "-m",
        "--modele",
        default="fr_core_news_md"
    )

    parser.add_argument(
        "--mode",
        choices=("tei", "ariane"),
        default="tei"
    )

    parser.add_argument(
        "-e",
        "--encodage",
        default="utf-8"
    )

    args = parser.parse_args()

    main(**vars(args))