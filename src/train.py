import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score


def run(df, cfg, features, model):

    fold_accuracies = []
    model_name = model.__class__.__name__

    for fold in range(cfg.split.n_splits):
        # Получаем индекс фолдов
        train_fold = df[df["kfold"] != fold]
        val_fold = df[df["kfold"] == fold]

        # Делим тренировочный дата сет на X, y
        X_train = train_fold[features]
        y_train = train_fold[cfg.columns.target]

        # Делаем валидацию X, y
        X_val = val_fold[features]
        y_val = val_fold[cfg.columns.target]

        # Свежая копия на фолда
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        # Делаем предсказание, подсчет метрик
        preds = fold_model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        fold_accuracies.append(acc)
        print(f"Фолд {fold}: Accuracy = {acc:.4f}")

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    return mean_accuracy, std_accuracy, model_name
