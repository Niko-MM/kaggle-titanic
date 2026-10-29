import pandas as pd


def drop_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Удаляем колонки"""
    df = df.copy()

    columns_to_drop = cfg.features.drop
    df = df.drop(columns_to_drop, axis=1)

    return df


def build_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Создаем новые фичи"""
    df = df.copy()
    # Вытаскиваем титул из имен и создаем новую фичу
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.", expand=False)

    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Don": "Rare",
        "Rev": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Lady": "Rare",
        "Sir": "Rare",
        "Col": "Rare",
        "Capt": "Rare",
        "the Countess": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }

    df["Title"] = df["Title"].replace(title_mapping)

    # Создаем фичу размера семьи из родственников
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df = drop_columns(df, cfg)

    return df
