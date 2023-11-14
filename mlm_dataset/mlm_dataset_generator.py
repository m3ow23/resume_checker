import pandas as pd
import re
import random
import math

class MLMDatasetGenerator:
    def __init__(self, datasetPath) -> None:
        # import dataset 
        dataset = pd.read_csv(datasetPath)

        # drop unecessary columns
        dataset = dataset.drop(columns=['ID', 'Resume_html', 'Category'])

        # set settings for printing
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', 1000)

        # # check for missing values in each row
        # missing_values_per_row = dataset.isnull().any(axis=1)

        # # display rows with missing values
        # rows_with_missing_values = dataset[missing_values_per_row]

        segmented_sentence_dataset = []
        # iterate over all resumes
        for resume in dataset['Resume_str']:
            # split the text when there is two or more spaces between
            segmented_sentence_dataset += (re.split(r'\s\s+', resume))

        cleaned_sentence_dataset = []
        for sentence in segmented_sentence_dataset:
            # convert item names to lowercase
            sentence = sentence.lower()

            # substitute year to [YEAR] token
            sentence = re.sub(r"(20\d\d|19\d\d)", '[YEAR]', sentence)

            # substitute numbers to [NUMBER] token
            sentence = re.sub(r"\d+\.?\d*\+?", '[NUMBER]', sentence)
            
            # remove special characters (e.g., punctuations)
            special_characters = r"[!\"#$%&()*+,/:;<=>?@^_{|}~•·]"
            sentence = re.sub(special_characters, ' ', sentence)
            new_sentence = []
            for word in sentence.split(' '):
                if (re.search(r"[\.\-]", word)):
                    # handler of abbreviations that use '.' (e.g., B.S.)
                    if (re.search(r"^[^\.]+\.$|^.$", word)):
                        word = re.sub(r"\.", ' ', word)
                    # handler of misplaced '-' and avoid removing '-' for hypenated words (e.g., Mother-in-law)
                    if (re.search(r"[^\s]+-[^\s]+", word) == None):
                        word = re.sub(r"-", ' ', word)
                new_sentence.append(word)
            sentence = ' '.join(new_sentence)

            # remove white spaces
            sentence = ' '.join(sentence.split())

            # explicit removal of observed noise words
            noise_words = ['n a', 'company name']
            if (sentence in noise_words):
                sentence = ''

            cleaned_sentence_dataset.append(sentence)

        sentence_dataset = []
        # remove data with less than 2 words
        for sentence in cleaned_sentence_dataset:
            if (sentence.strip() != '' 
                    and len(sentence.split(' ')) > 1
                    and not re.search(r"\[YEAR\]", sentence)):
                sentence_dataset.append(sentence)

        self.preprocessedDataset = sentence_dataset
        
    def generateMLMDataset(self, batch_size) -> [tuple]:
        sentence_dataset = self.preprocessedDataset

        mlm_dataset = []
        batch = []
        tokens_batch = []
        labels_batch = []
        counter = 0
        # create tuple of sentence with masked tokens and the labels of the masked tokens
        for sentence in sentence_dataset:
            tokens = sentence.split(' ')

            # get 15% of indices within the sentence to mask
            random_indices = sorted([random.randint(0, len(tokens) - 1) for _ in range(math.ceil((len(tokens) * .15)))])

            tokens_batch.append(tokens)

            labels = []
            # change the words in the indices to the token [MASK]
            for index in random_indices:
                labels.append(tokens[index])
                tokens[index] = '[MASK]'
            labels_batch.append(labels)
            
            counter += 1
            if (counter >= batch_size):
                batch = (tokens_batch, labels_batch)
                mlm_dataset.append(batch)
                # create new batch
                tokens_batch = []
                labels_batch = []
                batch = [] 
                counter = 0

        return mlm_dataset
    
    def getVocubulary(self):
        return [['[MASK]'], ['[NUMBER]']] + self.preprocessedDataset