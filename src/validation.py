import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(df: pd.DataFrame, cfg):
    """Нарезаем фолды"""
    df = df.copy()

    df["kfold"] = -1

    cv = StratifiedKFold(
        n_splits=cfg.split.n_splits, shuffle=True, random_state=cfg.general.seed
    )

    # присваиваем номера фолдам строкам val
    y = df[cfg.columns.target]
    for fold, (_, val_idx) in enumerate(cv.split(X=df, y=y)):
        df.loc[val_idx, "kfold"] = fold

    return df
