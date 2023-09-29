import pandas as pd

import sys
sys.path.append('.')

from test_encoders.bert.s2s_slavepe_avecs import get_cosine_similarity
from test_encoders.bert.s2s_slavepe_avecs import get_sentence_embeddings
from utils.tokenizer import sent_tokenize

from dataset.job_description import job_descriptions

# import dataset 
dataset = pd.read_csv('dataset/resume_dataset.csv')

# drop unecessary columns
dataset = dataset.drop(columns=['Resume_html'])

# resume to be used
job_desc = sent_tokenize(job_descriptions[0])

# get sentence embeddings of job description
job_desc_sentence_embeddings = get_sentence_embeddings(job_desc)

indexes = [67, 997, 16, 7, 33, 90, 635, 1536, 1532, 334]

top_10_resumes = []
for index in indexes:
    top_10_resumes += [[index, dataset['ID'][index], dataset['Resume_str'][index], dataset['Category'][index]]]

noise_words = ['n a', 'company name', 'city', 'state', '\[YEAR\]', '\[NUMBER\]']    

resume_similarities = []
for resume in top_10_resumes:
    tokenized_resume = sent_tokenize(resume[2], noise_words=noise_words)
    resume_sentence_embeddings = get_sentence_embeddings(tokenized_resume)

    similarity = get_cosine_similarity(resume_sentence_embeddings, job_desc_sentence_embeddings)

    resume_similarities += [[str(resume[0]), str(resume[1]), str(resume[3]), str(similarity)]]

sorted_resume_similarities = sorted(resume_similarities, key=lambda resume_similarities: resume_similarities[3], reverse=True)

for resume in sorted_resume_similarities:
    print(str(resume[0]).ljust(6, ' ') + ' ' + str(resume[1]).ljust(10, ' ') + ' ' + str(resume[3]).ljust(20, ' ') + ' ' + str(resume[2]))