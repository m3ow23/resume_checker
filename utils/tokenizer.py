import re

def sent_tokenize(document):
    # segment sentences when two or more spaces
    segmented_sentences = re.split(r'\s\s+', document)

    # lowercasing, year to [TOKEN], numbers to [NUMBER], remove special chars, remove white sapces, remove noise words 
    cleaned_sentences = _clean_sentences(segmented_sentences)

    # remove single word sentences, remove dates
    filtered_sentences = _filter_sentences(cleaned_sentences)

    return filtered_sentences

def _clean_sentences(segmented_sentences):
    cleaned_sentences = []

    for sentence in segmented_sentences:
        # convert item names to lowercase
        sentence = sentence.lower()

        # substitute year to [YEAR] token
        sentence = re.sub(r"(20\d\d|19\d\d)", '[YEAR]', sentence)

        # substitute numbers to [NUMBER] token
        sentence = re.sub(r"\d+\.?\d*\+?", '[NUMBER]', sentence)
        
        # remove special characters (e.g., punctuations)
        special_characters = r"[!\"#$%&()*+,/:;<=>?@^_{|}~•·]"
        sentence = re.sub(special_characters, ' ', sentence)
        sentence = _remove_special_char(sentence)

        # remove white spaces
        sentence = ' '.join(sentence.split())

        # explicit removal of observed noise words
        noise_words = ['n a', 'company name']
        if (sentence in noise_words):
            sentence = ''

        cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def _remove_special_char(sentence):
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

    return ' '.join(new_sentence)

def _filter_sentences(cleaned_sentences):
    filtered_sentences = []

    for sentence in cleaned_sentences:
        if (sentence.strip() != '' 
                # remove data with less than 2 words
                and len(sentence.split(' ')) > 1
                # remove dates
                and not re.search(r"\[YEAR\]", sentence)):
            filtered_sentences.append(sentence)

    return filtered_sentences