import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer
# import distance
import sklearn
from collections import OrderedDict


def freqs2clustering(dic_mots):
    if not dic_mots:
        return {}

    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

    Set_00 = set(dic_mots.keys())
    liste_words = [item for item in Set_00 if len(item) != 1]
    dic_output = {}
    matrice=[]

    words = np.asarray(liste_words) #So that indexing with a list will work
    for w in words:
        liste_vecteur=[]
        for w2 in words:
                V = CountVectorizer(ngram_range=(2,3), analyzer='char')# Vectorisation bigramme et trigramme de caractères 
                X = V.fit_transform([w,w2]).toarray()
                distance_tab1=sklearn.metrics.pairwise.cosine_distances(X) # Distance avec cosinus            
                liste_vecteur.append(distance_tab1[0][1])# stockage de la mesure de similarité
        matrice.append(liste_vecteur)
    matrice_def=-1*np.array(matrice)

    ##### CLUSTER

    affprop = AffinityPropagation(affinity="precomputed", damping= 0.5, random_state = None) 

    # print("="*64)
    # print("="*64)
    # print("="*64)
    # print()
    affprop.fit(matrice_def)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
        dic = new_d.get(exemplar)
        # print(exemplar, " ==> ", list(cluster))
        if dic is not None:
            dic_output[exemplar] = {
                "Freq.centroide": dic,
                "Termes": list(cluster),
            }
    # print()
    # print("="*64)
    # print("="*64)
    # print("="*64)

    return dic_output
