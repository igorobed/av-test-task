from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from string import punctuation


nltk.download("stopwords")
en_stops = set(stopwords.words('english'))
ru_stops = set(stopwords.words('russian'))
stops = en_stops | ru_stops


def simple_tokenizer(in_str: str) -> List[str]:
    """
    Простая токенизация по символам и отдельным словам с удалением символов и стоп-слов
    на русском и английском языках
    """
    temp_tokenize = [item for item in wordpunct_tokenize(in_str) if item not in punctuation]
    return [item for item in temp_tokenize if item not in stops]


if __name__ == "__main__":
    pass