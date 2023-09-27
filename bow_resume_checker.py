import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.tokenizer import sent_tokenize
from dataset.job_description import job_descriptions
from bow_vectorizer import d2d_cs as bow_vectorizer

# import dataset 
dataset = pd.read_csv('dataset/resume_dataset.csv')

# drop unecessary columns
dataset = dataset.drop(columns=['Resume_html'])

# noise_words for sent_tokenize()
noise_words = ['n a', 'company name', 'city', 'state', '\[YEAR\]', '\[NUMBER\]']    

# job description to be used
job_desc = " ".join(sent_tokenize(job_descriptions[0], noise_words=noise_words))

i = len(open("bow_resume_similarities.txt", "r").readlines())

while i < dataset.shape[0]:
    f = open("bow_resume_similarities.txt", "a")

    resume = " ".join(sent_tokenize(dataset['Resume_str'][i], noise_words=noise_words))

    job_desc_vector, resume_vector = bow_vectorizer.get_vectors(job_desc, resume)

    similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]

    string = str(i) + " " + str(dataset['ID'][i]) + " " + str(dataset['Category'][i]) + " " + str(similarity) + "\n"

    print(string)
    f.write(string)
    f.close()

    i += 1 #increment