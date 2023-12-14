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

"""
    This is a non-batching implementation of MLM Dataset Generator.\n
    This implementation includes saving to file and loading to file.
"""

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
                        word = re.sub(r"\.", '\n', word)
                    # handler of misplaced '-' and avoid removing '-' for hypenated words (e.g., Mother-in-law)
                    if (not re.match(r"[\w]+(-[\w]+)+", word)):
                        word = re.sub(r"-", ' ', word)

                new_sentence += word + ' '

            sentences = new_sentence.split('\n')

            for index, sentence in enumerate(sentences):
                new_sentence = ''

                # remove white spaces
                sentence = str.strip(re.sub(r"\s+", ' ', sentence))

                for word in sentence.split(' '):
                    # stemming
                    new_sentence += stemmer.stem(word) + ' '

                # explicit removal of observed noise words
                noise_words = ['n a', 'company name']

                new_sentence = new_sentence.strip()
                
                if (new_sentence in noise_words):
                    new_sentence = ''

                sentences[index] = new_sentence

            cleaned_sentence_dataset += [item for item in sentences if item != ""]

        sentence_dataset = []
        # remove data with less than 2 words
        for sentence in cleaned_sentence_dataset:
            if (sentence.strip() != '' 
                    and len(sentence.split(' ')) > 1
                    and not re.search(r"\[year\]", sentence)):
                sentence_dataset.append(sentence)

        # truncate sentence that are longer than max_pos
        truncated_sentence_dataset = []
        if self.max_pos is not None:
            for sentence in sentence_dataset:
                splitted_sentence = sentence.split(' ')
                if (len(splitted_sentence) > self.max_pos):
                    num_split = math.ceil(len(splitted_sentence) / self.max_pos)
                    # Split the list into n parts
                    truncated_sentence_dataset += [' '.join(splitted_sentence[i * self.max_pos: (i + 1) * self.max_pos]) for i in range(num_split)]
                else:
                    truncated_sentence_dataset.append(sentence)

        return truncated_sentence_dataset
    
    def _generate_masked_data(self, data) -> List[str]:
        mlm_dataset = []
        # create tuple of sentence with masked tokens and the labels of the masked tokens
        for sentence in data:
            tokens = sentence.split(' ')

            # get 15% of indices within the sentence to mask
            random_indices = sorted(random.sample(range(len(tokens)), math.ceil(len(tokens) * 0.15)))

            # labels
            labels = []
            # change masked token indices to [MASK] token
            for index in random_indices:
                labels.append(tokens[index])
                tokens[index] = "[mask]"

            mlm_dataset.append(tokens)
            mlm_dataset.append(labels)

        return mlm_dataset
        
    def generateMLMDataset(self, sample_limit=None) -> Tuple[List[str], List[str], List[str]]:
        """
            Returns a tuple of lists of masked tokens and the labels: [tokens, labels, ..., tokens, labels].

            Usage:

                training_data, validation_data, testing_data = mlm_dataset

                training_tokens = training_data[::2] even indices are tokens

                training_labels = training_data[1::2] odd indices are labels
        """

        if (sample_limit):
            training_data = self.training_data[:sample_limit]
            validation_data = self.validation_data[:sample_limit]
            testing_data = self.testing_data[:sample_limit]
        else:
            training_data = self.training_data
            validation_data = self.validation_data
            testing_data = self.testing_data

        return (self._generate_masked_data(training_data),
                     self._generate_masked_data(validation_data),
                     self._generate_masked_data(testing_data))
    
    def generate_mlm_dataset_to_file(self, samle_limit=None):
        mlm_dataset = self.generateMLMDataset(samle_limit)

        training_data, validation_data, testing_data = mlm_dataset

        with open('training_data.txt', 'w') as file:
            for data in training_data:
                file.write(' '.join(data) + '\n')
        
        with open('validation_data.txt', 'w') as file:
            for data in validation_data:
                file.write(' '.join(data) + '\n')

        with open('testing_data.txt', 'w') as file:
            for data in testing_data:
                file.write(' '.join(data) + '\n')

    def read_mlm_dataset_from_file(self):
        training_data = []
        validation_data = []
        testing_data = []

        with open('training_data.txt', 'r') as file:
            training_data = [line.strip().split(' ') for line in file.readlines()]
        
        with open('validation_data.txt', 'r') as file:
            validation_data = [line.strip().split(' ') for line in file.readlines()]

        with open('testing_data.txt', 'r') as file:
            testing_data = [line.strip().split(' ') for line in file.readlines()]

        return (training_data, validation_data, testing_data)
    
    def save_raw_training_data_to_file(self):
        with open('raw_training_data.txt', 'w') as file:
            for data in self.training_data:
                file.write(data + '\n')

    def read_raw_training_data_from_file(self):
        with open('raw_training_data.txt', 'r') as file:
            self.training_data = [line.strip() for line in file.readlines()]
    
    def getVocubulary(self):
        return self.training_data