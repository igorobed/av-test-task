from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
import pandas as pd
import sys
sys.path.append("../")
import src.feats_generation as f_g


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


def get_preprocess_data() -> pd.DataFrame:
    """
    Скрипт для быстрой предобработки датасета, на которой я после провожу эксперементы
    """
    df = pd.read_csv("../data/raw/АВСОФТ_тест_ML_приложение.csv")
    
    # удаляем столбец, не несущий полезной инфы
    df.drop(columns=["commit_hash"], inplace=True)

    # кодируем имена репозиториев
    df = pd.concat([df, pd.get_dummies(df.repository_name)], axis=1)
    df.drop(columns=["repository_name"], inplace=True)
    
    # кодируем имена авторов коммитов(внутри происходит разделение на группы)
    df = f_g.ohe(df, "commit_author")
    df["commit_date"] = pd.to_datetime(df.commit_date)

    # преобразуем столбец отвечающий за время в пару закодированных признаков
    df = f_g.encode_work_days(df)
    df = f_g.encode_work_hours(df)
    df.drop(columns=["commit_date"], inplace=True)

    return df


if __name__ == "__main__":
    pass