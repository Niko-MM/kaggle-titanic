import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score


def run(df, cfg, features, model):
    """
    На каждом фолде учится своя копия модели.
    Считаем среднее и разброс accuracy
    """

    fold_accuracies = []
    model_name = model.__class__.__name__

    for fold in range(cfg.split.n_splits):
        train_fold = df[df["kfold"] != fold]
        val_fold = df[df["kfold"] == fold]

        X_train = train_fold[features]
        y_train = train_fold[cfg.columns.target]

        X_val = val_fold[features]
        y_val = val_fold[cfg.columns.target]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        fold_accuracies.append(acc)
        print(f"Фолд {fold}: Accuracy = {acc:.4f}")

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    return mean_accuracy, std_accuracy, model_name
