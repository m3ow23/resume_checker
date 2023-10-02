# Model Used: https://huggingface.co/bert-base-uncased

# document-to-document, average cosine similarity

from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy
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

def get_document_embedding(document):
    
    # Tokenize and pad/trim sentences
    tokenized_document = tokenizer(document, padding=True, truncation=True, return_tensors="pt")
    # Compute sentence-level embeddings for each sentence
    document_embedding = model(**tokenized_document).last_hidden_state.mean(dim=1)

    return document_embedding

def get_cosine_similarity(document_A_embeddings, document_B_embeddings):
    # Calculate cosine similarity between documents
    similarity_score = 1 - cosine(
        document_A_embeddings.detach().numpy().flatten(),  # Convert to 1-D array
        document_B_embeddings.detach().numpy().flatten(),  # Convert to 1-D array
    )

    return similarity_score


# import dataset 
dataset = pd.read_csv('dataset/resume_dataset.csv')

# drop unecessary columns
dataset = dataset.drop(columns=['Resume_html'])

# noise_words for sent_tokenize()
noise_words = ['n a', 'company name', 'city', 'state', '\[YEAR\]', '\[NUMBER\]']    

# resume to be used
job_desc = '. '.join(sent_tokenize(job_descriptions[0], noise_words=noise_words))

# get sentence embeddings of job description
job_desc_embedding = get_document_embedding(job_desc)

i = len(open("test_encoders/bert/similarities/d2d_dlavepe_avecs.txt", "r").readlines())

average_processing_time = None

while i < dataset.shape[0]:
    start_time = datetime.now()

    f = open("test_encoders/bert/similarities/d2d_dlavepe_avecs.txt", "a")

    resume = dataset['Resume_str'][i]
    tokenized_resume = '. '.join(sent_tokenize(resume, noise_words=noise_words))
    resume_embedding = get_document_embedding(tokenized_resume)

    similarity = get_cosine_similarity(resume_embedding, job_desc_embedding)

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