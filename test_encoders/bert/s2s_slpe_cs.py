# Model Used: https://huggingface.co/bert-base-uncased

from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy as np

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
    similarity_scores = []
    for i, sentence_embedding_A in enumerate(document_A_sentence_embeddings):
        for j, sentence_embedding_B in enumerate(document_B_sentence_embeddings):
            similarity_score = 1 - cosine(
                sentence_embedding_A.detach().numpy().flatten(),  # Convert to 1-D array
                sentence_embedding_B.detach().numpy().flatten(),  # Convert to 1-D array
            )
            similarity_scores += [similarity_score]

    return np.mean(similarity_scores)