from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


def ensemble(cfg):
    """Собираем 1 случайный лес и 2 бустинга.
    кладем в конструктор
    """
    # Беру лучшие параметры леса, строка 22
    forest = RandomForestClassifier(
        n_estimators=100, max_depth=7, random_state=cfg.general.seed
    )

    # Лучшие параметры, строка 31
    catboost = CatBoostClassifier(
        iterations=100,
        depth=3,
        random_seed=cfg.general.seed,
        verbose=0,
        allow_writing_files=False,
    )

    # Лучшие параметры, строка 37
    xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=cfg.general.seed)

    # Собираем ансабль
    committee = VotingClassifier(
        estimators=[("catboost", catboost), ("forest", forest), ("xgb", xgb)],  # type: ignore
        voting="soft",
    )
    return committee
