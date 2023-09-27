from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_vectors(document_A, document_B):

    vocabulary = [document_A + ' ' + document_B]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(vocabulary)

    document_A_vector = vectorizer.transform([document_A])
    document_B_vector = vectorizer.transform([document_B])

    return [document_A_vector, document_B_vector]