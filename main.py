from pathlib import Path

import pandas as pd

from config import config
from src.data import load_data
from src.ensemble import ensemble
from src.features import build_features
from src.preprocessing import encode_categorical_features, fill_train_and_test
from src.train import run
from src.utils import log_experiment
from src.validation import create_folds


def run_cv_txt(df_train, features, cfg, model):
    """
    Считаем accuracy по фолдам и пишем журнал.
      Возвращаем среднюю точность и разброс
    """
    # Делим датасет на фолды
    df_folded = create_folds(df_train, cfg)
    # Запускаем тренировку моделей
    mean_accuracy, std_accuracy, model_name = run(df_folded, cfg, features, model=model)
    # Пишем логи cv, txt
    log_experiment(
        experiment_name=cfg.general.experiment_name,
        model_name=model_name,
        features=features,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        seed=cfg.general.seed,
    )
    return mean_accuracy, std_accuracy


def prepare(train, test, cfg):
    """Подготовка данных"""

    test_ids = test[cfg.columns.id].copy()
    # Создаем фичи и удаляем не нужные
    df_train = build_features(train, cfg)
    df_test = build_features(test, cfg)
    # Заполняем пропуски
    df_train, df_test = fill_train_and_test(df_train, df_test)
    # Кодируем катигориальные фичи
    df_train = encode_categorical_features(df_train, cfg)
    df_test = encode_categorical_features(df_test, cfg)
    # Кладем в список все фичи, кроме таргета
    features = [col for col in df_train.columns if col != cfg.columns.target]
    # Подгоняем колонки теста под features
    df_test = df_test.reindex(columns=features)

    return df_train, df_test, test_ids, features


def write_submission(df_train, df_test, test_ids, features, cfg, model):
    """Один fit на всех строках и файл для Kaggle"""
    # Обучаем модель на тренировочном сете
    model.fit(df_train[features], df_train[cfg.columns.target])
    predictions = model.predict(df_test)

    submission = pd.DataFrame(
        {
            cfg.columns.id: test_ids.values,
            cfg.columns.target: predictions.astype(int),
        }
    )
    output_path = Path(__file__).parent / "outputs" / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Файл ответа: {output_path}, строк: {len(submission)}")
    return output_path


if __name__ == "__main__":
    train_raw, test_raw = load_data(config)
    df_train, df_test, test_ids, features = prepare(train_raw, test_raw, config)
    model = ensemble(config)

    # Прогоняем модель, пишем логи csv, txt
    # run_cv_txt(df_train, features, config, model)

    # Пишем submission.csv
    write_submission(df_train, df_test, test_ids, features, config, model)
