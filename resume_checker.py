import pandas as pd
import os

from test_encoders.bert import s2s_slpe_cs as bert_encoder
from utils.tokenizer import sent_tokenize

from dataset.job_description import job_descriptions

# import dataset 
dataset = pd.read_csv('dataset/resume_dataset.csv')

# drop unecessary columns
dataset = dataset.drop(columns=['Resume_html'])

# noise_words for sent_tokenize()
noise_words = ['n a', 'company name', 'city', 'state', '\[YEAR\]', '\[NUMBER\]']    

# resume to be used
job_desc = sent_tokenize(job_descriptions[0], noise_words=noise_words)

# get sentence embeddings of job description
job_desc_sentence_embeddings = bert_encoder.get_sentence_embeddings(job_desc)

i = len(open("resume_similarities.txt", "r").readlines())

while i < dataset.shape[0]:
    f = open("resume_similarities.txt", "a")

    resume = dataset['Resume_str'][i]
    tokenized_resume = sent_tokenize(resume, noise_words=noise_words)
    resume_sentence_embeddings = bert_encoder.get_sentence_embeddings(tokenized_resume)

    similarity = bert_encoder.get_cosine_similarity(resume_sentence_embeddings, job_desc_sentence_embeddings)

    string = str(i) + " " + str(dataset['ID'][i]) + " " + str(dataset['Category'][i]) + " " + str(similarity) + "\n"

    print(string)
    f.write(string)
    f.close()

    i += 1 #increment

# os.system("shutdown /s /t 1")