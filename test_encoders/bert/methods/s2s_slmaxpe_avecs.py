# Model Used: https://huggingface.co/bert-base-uncased

# sentence-to-sentence, sentence-level max pooling embeddings, average cosine similarity

from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import os
from datetime import datetime

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
    document_sentence_embeddings = [model(**tokenized_sentence).last_hidden_state.max(dim=1).values for tokenized_sentence in tokenized_document]

    return document_sentence_embeddings

def get_cosine_similarity(document_A_sentence_embeddings, document_B_sentence_embeddings):
    # Calculate cosine similarity between sentences
    similarity_scores = []
    for sentence_embedding_A in document_A_sentence_embeddings:
        for sentence_embedding_B in document_B_sentence_embeddings:
            similarity_score = 1 - cosine(
                sentence_embedding_A.detach().numpy().flatten(),  # Convert to 1-D array
                sentence_embedding_B.detach().numpy().flatten(),  # Convert to 1-D array
            )
            similarity_scores += [similarity_score]

    return np.mean(similarity_scores)


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

i = len(open("test_encoders/bert/similarities/s2s_slmaxpe_avecs.txt", "r").readlines())

average_processing_time = None

while i < dataset.shape[0]:
    start_time = datetime.now()

    f = open("test_encoders/bert/similarities/s2s_slmaxpe_avecs.txt", "a")

    resume = dataset['Resume_str'][i]
    tokenized_resume = sent_tokenize(resume, noise_words=noise_words)
    resume_sentence_embeddings = get_sentence_embeddings(tokenized_resume)

    similarity = get_cosine_similarity(resume_sentence_embeddings, job_desc_sentence_embeddings)

    string = str(i) + " " + str(dataset['ID'][i]) + " " + str(dataset['Category'][i]) + " " + str(similarity) + "\n"
    
    elapsed_time = datetime.now() - start_time
    if (not average_processing_time):
        average_processing_time = elapsed_time
    else:
        average_processing_time = (average_processing_time + elapsed_time) / 2

    print(string + 'Elapsed Time: ' + str(elapsed_time) + ' ETA: ' + str(average_processing_time * (dataset.shape[0] - i)) + '\n')
    f.write(string)
    f.close()

    i += 1 #increment

# os.system("shutdown /s /t 1")