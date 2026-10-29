import pandas as pd


def _apply_median_card(df, card, column, keys, fallback):
    """Прикладываем карточку медиан (Собранная по тренировочному сету)
      и заполняем пропуски ею.
    Если медиана карточки не нашлась, берем среднее по фиче
    """
    # Создаем название колонки для маски
    median_column = f"{column}_median"
    # Соединяем датасет с новой колонкой из кард
    filled = df.merge(card, on=keys, how="left")
    # Заполняем пропуски пол логике карда или простой медианой по тренировочному сету
    filled[column] = filled[column].fillna(filled[median_column]).fillna(fallback)
    return filled.drop(columns=median_column)


def fill_train_and_test(train: pd.DataFrame, test: pd.DataFrame):
    """Заполняем пропуски в train и test

    Медианы считаются только по train и ими же закрываются дыры  в test.
    Вызывать после build_features: нужны Title, Sex, Pclass, Embarked, Age, Fare.
    """
    train = train.copy()
    test = test.copy()

    # Порт заполняем константой. В анализе всего два пропуск
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Embarked"] = test["Embarked"].fillna("S")

    # Запасное число, если в группе медиана не нашлась.
    age_fallback = train["Age"].median()
    fare_fallback = train["Fare"].median()

    # Делаем карту на группу. Создаем новую колонку, сбрасываем индкесы
    age_card = (
        train.groupby(["Title", "Pclass", "Sex"])["Age"]
        .median()
        .rename("Age_median")
        .reset_index()
    )
    fare_card = (
        train.groupby(["Pclass", "Embarked"])["Fare"]
        .median()
        .rename("Fare_median")
        .reset_index()
    )
    # Списки для группировок фич
    age_keys = ["Title", "Pclass", "Sex"]
    fare_keys = ["Pclass", "Embarked"]

    train = _apply_median_card(train, age_card, "Age", age_keys, age_fallback)
    test = _apply_median_card(test, age_card, "Age", age_keys, age_fallback)
    train = _apply_median_card(train, fare_card, "Fare", fare_keys, fare_fallback)
    test = _apply_median_card(test, fare_card, "Fare", fare_keys, fare_fallback)

    return train, test


def encode_categorical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Кодируем категориальные признаки через One-Hot Encoding"""

    df_encoded = pd.get_dummies(
        df, columns=list(cfg.features.categorical), drop_first=True
    )

    return df_encoded
