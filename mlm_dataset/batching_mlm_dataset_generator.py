import pandas as pd
import re
import random
import math
import tensorflow as tf
import numpy as np
from typing import List, Tuple
from collections import Counter
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

class MLMDatasetGenerator:
    def __init__(self, max_pos=None):
        self.max_pos = max_pos

    def create_dataset(self, datasetPath) -> None:
        # import dataset 
        dataset = pd.read_csv(datasetPath)

        # drop unecessary columns
        dataset = dataset.drop(columns=['ID', 'Resume_html'])

        categories = Counter(dataset['Category'])

        # Dividing data for each category by 60:20:20 ratio
        training_data = []
        validation_data = []
        testing_data = []
        for category, count in categories.items():
            category_subset = dataset[dataset['Category'] == category]

            # Calculate indices for training, validation, and testing
            training_index = round(count * 0.6)
            validation_index = training_index + round(count * 0.2)

            # Split the data into training, validation, and testing sets
            training_data += category_subset.iloc[:training_index]['Resume_str'].tolist()
            validation_data += category_subset.iloc[training_index:validation_index]['Resume_str'].tolist()
            testing_data += category_subset.iloc[validation_index:]['Resume_str'].tolist()
    
        self.training_data = self._preprocess(training_data)
        self.validation_data = self._preprocess(validation_data)
        self.testing_data = self._preprocess(testing_data)

    """
        This method reduces the resumes in a dataset to a list of sentences.
    """

    def _preprocess(self, dataset):
        segmented_sentence_dataset = []
        # iterate over all resumes
        for resume in dataset:
            # split the text when there is two or more spaces between
            segmented_sentence_dataset += (re.split(r'\s\s+', resume))

        cleaned_sentence_dataset = []
        for sentence in segmented_sentence_dataset:
            # convert item names to lowercase
            sentence = sentence.lower()
            
            # remove special characters (e.g., punctuations)
            # special_characters = r"[\[\]!\"#$%&()*+,/:;<=>?@^_{|}~•·◦]"
            not_allowed_characters = r"[^a-z0-9\-\.']"
            sentence = re.sub(not_allowed_characters, ' ', sentence)

            # reduce from possesive form
            # sentence = re.sub(r"'|'s", ' ', sentence) # already handled by stemming

            # substitute year to [YEAR] token
            sentence = re.sub(r"(20\d\d|19\d\d)", '[YEAR]', sentence)

            # substitute numbers to [NUMBER] token
            sentence = re.sub(r"\d+\.?\d*\+?|\d+th|\d+rd|\d+k", '[NUMBER]', sentence)

            new_sentence = ''
            for word in sentence.split(' '):
                if (re.search(r"[\.\-]", word)):
                    # handler of abbreviations that use '.' (e.g., B.S.)
                    if (not re.match(r"([a-z]{1,2}\.)+", word)):
                        word = re.sub(r"\.", ' ', word)
                    # handler of misplaced '-' and avoid removing '-' for hypenated words (e.g., Mother-in-law)
                    if (not re.match(r"[\w]+(-[\w]+)+", word)):
                        word = re.sub(r"-", ' ', word)
                # stemming
                new_sentence += stemmer.stem(word) + ' '

            # remove white spaces
            sentence = ' '.join(new_sentence.split())

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

        return sentence_dataset
        
    def generateMLMDataset(self, batch_size, sample_limit=None) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
        if (sample_limit):
            training_data = self.training_data[:sample_limit]
            validation_data = self.validation_data[:sample_limit]
            testing_data = self.testing_data[:sample_limit]
        else:
            training_data = self.training_data
            validation_data = self.validation_data
            testing_data = self.testing_data

        return (self._generate_masked_data(training_data, batch_size=batch_size),
                     self._generate_masked_data(validation_data, batch_size=batch_size),
                     self._generate_masked_data(testing_data, batch_size=batch_size))

    def _generate_masked_data(self, data, batch_size):
        mlm_dataset = []
        tokens_batch = []
        labels_batch = []
        counter = 0
        # create tuple of sentence with masked tokens and the labels of the masked tokens
        for sentence in data:
            tokens = np.array(sentence.split(' '))

            # get 15% of indices within the sentence to mask
            random_indices = tf.constant(sorted(random.sample(range(len(tokens)), math.ceil(len(tokens) * 0.15))))

            # labels
            labels = tokens[random_indices]
            labels_batch.append(labels.tolist())
            # change masked token indices to [MASK] token
            tokens[random_indices] = '[MASK]'
            tokens_batch.append(tokens.tolist())
            
            counter += 1
            if (counter >= batch_size):
                batch = (tokens_batch, labels_batch)
                mlm_dataset.append(batch)
                # create new batch
                tokens_batch = []
                labels_batch = []
                counter = 0

        return mlm_dataset
    
    
    def read_mlm_dataset_from_file(self, batch_size: int = 1, sample_limit: int = None):
        training_data = []
        validation_data = []
        testing_data = []

        batch_size *= 2
        if sample_limit is not None:
            sample_limit *= 2

        with open('training_data.txt', 'r') as file:
            if sample_limit is not None:
                lines = file.readlines()[:sample_limit]
            else:
                lines = file.readlines()

            for i in range(0, len(lines), batch_size):
                batch = []
                for line in lines[i:i + batch_size]:
                    batch.append(line.strip().split(' '))
                training_data.append(batch)
        
        with open('validation_data.txt', 'r') as file:
            if sample_limit is not None:
                lines = file.readlines()[:sample_limit]
            else:
                lines = file.readlines()

            for i in range(0, len(lines), batch_size):
                batch = []
                for line in lines[i:i + batch_size]:
                    batch.append(line.strip().split(' '))
                validation_data.append(batch)

        with open('testing_data.txt', 'r') as file:
            if sample_limit is not None:
                lines = file.readlines()[:sample_limit]
            else:
                lines = file.readlines()

            for i in range(0, len(lines), batch_size):
                batch = []
                for line in lines[i:i + batch_size]:
                    batch.append(line.strip().split(' '))
                testing_data.append(batch)

        return (training_data, validation_data, testing_data)

    def read_raw_training_data_from_file(self):
        with open('raw_training_data.txt', 'r') as file:
            self.training_data = [line.strip() for line in file.readlines()]

    def getVocubulary(self):
        return ['[pad] [mask]'] + self.training_data