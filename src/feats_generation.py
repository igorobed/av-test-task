import pandas as pd


def ohe(data: pd.DataFrame, column: str, drop_orig_column=True) -> pd.DataFrame:
    """
    Применение One-Hot Encoder к определенному столбцу
    """
    data = pd.concat([data, pd.get_dummies(data[column])], axis=1)
    
    if drop_orig_column:
        data.drop(columns=[column], inplace=True)

    return data


def encode_work_hours(data: pd.DataFrame, column: str="commit_date", type_enc: str="ohe", drop_orig_column: bool=False) -> pd.DataFrame:
    """
    Разделение часов в течение дня на рабочее время и нерабочее
    type_enc - ["ohe", "binary", "label"]
    """

    # для удобства выделим часы из столбца с временем
    data["hour"] = data[column].apply(lambda x: x.hour)

    data["hour"] = data["hour"].apply(lambda x: "work_h" if x in list(range(8, 19)) else "no_work_h")

    if type_enc == "ohe":
        data = pd.concat([data, pd.get_dummies(data["hour"])], axis=1)
    
    # удаляем вспомогательный столбец
    data.drop(columns=["hour"], inplace=True)
    if drop_orig_column:
        data.drop(columns=[column], inplace=True)

    return data


def encode_work_days(data: pd.DataFrame, column: str="commit_date", type_enc: str="ohe", drop_orig_column: bool=False) -> pd.DataFrame:
    """
    Разделение дней нелели на рабочие и выходные и кодирование по этому признаку
    type_enc - ["ohe", "binary", "label"]
    """

    # для удобства выделим дни недели из столбца с временем
    data["day"] = data[column].apply(lambda x: x.weekday())

    data["day"] = data["day"].apply(lambda x: "work_d" if x in list(range(0, 5)) else "no_work_d")

    if type_enc == "ohe":
        data = pd.concat([data, pd.get_dummies(data["day"])], axis=1)
    
    # удаляем вспомогательный столбец
    data.drop(columns=["day"], inplace=True)
    if drop_orig_column:
        data.drop(columns=[column], inplace=True)

    return data



if __name__ == "__main__":
    pass