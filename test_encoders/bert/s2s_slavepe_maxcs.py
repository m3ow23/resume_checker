# Model Used: https://huggingface.co/bert-base-uncased

# UNTESTED

# sentence-to-sentence, sentence-level average pooling embeddings, average cosine similarity

from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import os

import sys
sys.path.append('.')

from utils.tokenizer import sent_tokenize
from dataset.job_description import job_descriptions

# import bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embeddings(document_sentences):
    # Tokenize and pad/trim sentences
    tokenized_document = [tokenizer(sentence, padding=True, truncation=True, return_tensors="pt") for sentence in document_sentences]
    # Compute sentence-level embeddings for each sentence
    document_sentence_embeddings = [model(**tokenized_sentence).last_hidden_state.mean(dim=1) for tokenized_sentence in tokenized_document]

    return document_sentence_embeddings

def get_cosine_similarity(document_A_sentence_embeddings, document_B_sentence_embeddings):
    # Calculate cosine similarity between sentences
    max_similarity_scores = []
    for sentence_embedding_A in document_A_sentence_embeddings:
        max_similarity_score = 0
        for sentence_embedding_B in document_B_sentence_embeddings:
            similarity_score = 1 - cosine(
                sentence_embedding_A.detach().numpy().flatten(),  # Convert to 1-D array
                sentence_embedding_B.detach().numpy().flatten(),  # Convert to 1-D array
            )
            if (similarity_score > max_similarity_score):
                max_similarity_score = similarity_score
        max_similarity_scores += [max_similarity_score]

    return np.mean(max_similarity_scores)


# import dataset 
dataset = pd.read_csv('dataset/resume_dataset.csv')

# drop unecessary columns
dataset = dataset.drop(columns=['Resume_html'])

# noise_words for sent_tokenize()
noise_words = ['n a', 'company name', 'city', 'state', '\[YEAR\]', '\[NUMBER\]']    

# resume to be used
job_desc = sent_tokenize(job_descriptions[0], noise_words=noise_words)

# get sentence embeddings of job description
job_desc_sentence_embeddings = get_sentence_embeddings(job_desc)

i = len(open("test_encoders/bert/similarities/s2s_slavepe_maxcs.txt", "r").readlines())

while i < dataset.shape[0]:
    f = open("test_encoders/bert/similarities/s2s_slavepe_maxcs.txt", "a")

    resume = dataset['Resume_str'][i]
    tokenized_resume = sent_tokenize(resume, noise_words=noise_words)
    resume_sentence_embeddings = get_sentence_embeddings(tokenized_resume)

    similarity = get_cosine_similarity(resume_sentence_embeddings, job_desc_sentence_embeddings)

    string = str(i) + " " + str(dataset['ID'][i]) + " " + str(dataset['Category'][i]) + " " + str(similarity) + "\n"

    print(string)
    f.write(string)
    f.close()

    i += 1 #increment