from bs4 import BeautifulSoup
import re
import numpy as np


def preprocessor(text):
    # remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # regex for matching emoticons, keep emoticons, ex: :), :-P, :-D
    r = '(?::|;|=|X)(?:-)?(?:\)|\(|D|P)'
    emoticons = re.findall(r, text)
    text = re.sub(r, '', text)

    # convert to lowercase and append all emoticons behind (with space in between)
    # replace('-','') removes nose of emoticons
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', '')
    return text


def cleaner(text, to_lower=True, norm_num=False, char_to_strip=' |(|)|,', non_alphanumeric_exceptions=","):
    # if the to_lower flag is true, convert the text to lowercase
    text = text.lower() if to_lower else text
    # Removes unwanted http hyper links in text
    text = re.sub(r"http(s)?://\S*", " ", text)
    # In some cases, one may want to normalize numbers for better visualization
    text = re.sub(r"[0-9]", "1", text) if norm_num else text
    # remove non-alpha numeric characters [!, $, #, or %] and normalize whitespace
    text = re.sub(r"[^A-Za-z0-9-" + non_alphanumeric_exceptions + "]", " ", text)
    # replace leftover unwanted white space
    text = whitespace_normalizer(text)
    # remove trailing or leading white spaces
    text = text.strip(char_to_strip)
    return text


# returns indexes where ground truth and predicted value does not match
query_for_false_predictions = lambda predictions, ground_truth: np.where(ground_truth != predictions)

# whitespace normalizer
whitespace_normalizer = lambda x: re.sub(r"\s+", " ", x)

# Convert string to a words list
generate_word_list = lambda x, token_type: whitespace_normalizer(x).split(token_type)
