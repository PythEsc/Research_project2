from gensim.parsing import PorterStemmer
import logging
import nltk
import re
import string as stringlib

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("Word2VecUtility")


def clean_str(string):
    """
    Cleans a given string using certain regular expression rules:
        - lower-case
        - blanks between punctuation
        - urls replaced by '__URL__'
        - @User replaced with '__AT_USER__'
        - replacing #<word> with <word>
        - whitespaces replaced with blanks
        - replacing 3 or more occurrences of one character with the character itself
        - removing words with less than 3 characters
        - removing sequences containing numbers

    :param string: The string that shall be cleaned
    :return: A cleaned string
    """
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    # Instead of using those lines we are replacing every "'" with " " such that words like "I'm" -> "I m" and get
    # removed by the latest filter which removes words with length <= 2

    # string = re.sub(r"'s", " 's", string)
    # string = re.sub(r"'ve", " 've", string)
    # string = re.sub(r"n't", " n't", string)
    # string = re.sub(r"'re", " 're", string)
    # string = re.sub(r"'d", " 'd", string)
    # string = re.sub(r"'ll", " 'll", string)

    string = re.sub(r"'", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    # Convert www.* or https?://* to __URL__
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '__URL__', string)
    # Convert @username to __AT_USER__
    string = re.sub('@[^\s]+', '__AT_USER__', string)
    # Remove additional white spaces
    string = re.sub('[\s]+', ' ', string)
    # Replace #word with word
    string = re.sub(r'#([^\s]+)', r'\1', string)
    # Replace 3 or more repetitions of character with the character itself
    string = re.sub(r'(.)\1{2,}', r'\1', string)
    # Remove words with 2 or less characters
    string = re.sub(r'\b\w{1,2}\b', '', string)
    # Remove sequences that contain numbers
    string = re.sub(r'\b\w*\d\w*\b', '', string)
    # trim
    return string.strip('\'"').strip()


# --------------------------------------------------------------------------

global_stemmer = PorterStemmer()


class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """

    # This reverse lookup will remember the original forms of the stemmed
    # words
    word_lookup = {}

    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """

        # Stem the word
        stemmed = global_stemmer.stem(word)

        # Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)

        return stemmed

    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """

        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word


def get_stopwords():
    logger.info("Downloading stopwords for pre-processing")
    nltk.download("stopwords")

    punctuation = list(stringlib.punctuation)

    stop = nltk.corpus.stopwords.words('english') + punctuation
    stop.append('__AT_USER__')
    stop.append('__URL__')

    print(stop)

    return stop
