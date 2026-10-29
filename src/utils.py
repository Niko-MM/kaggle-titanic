from pathlib import Path

import pandas as pd


def log_experiment(
    experiment_name: str,
    model_name: str,
    features: list,
    mean_accuracy: float,
    std_accuracy: float,
    seed: int,
    log_path_csv: str = "outputs/results.csv",
    log_path_txt: str = "outputs/results.txt",
):
    """Запись логов в csv, txt"""
    # заемтки для csv и txt файла
    row = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "features": " + ".join(features),
        "mean_accuracy": round(mean_accuracy, 4),
        "std_accuracy": round(std_accuracy, 4),
        "seed": seed,
    }

    df_row = pd.DataFrame([row])

    path_to_file = Path(log_path_csv)
    file_exists = path_to_file.exists()

    df_row.to_csv(
        path_to_file, mode="a", index=False, header=not file_exists, encoding="utf-8"
    )

    print(f"Результаты эксперимента '{experiment_name}' успешно сохранены в CSV")

    # заметки для txt файла
    features_str = ", ".join(features)

    txt_line = (
        f"{experiment_name} | {model_name} | "
        f"среднее {mean_accuracy:.4f} | разброс {std_accuracy:.4f} | "
        f"seed {seed} | f: {features_str}"
    )
    with open(log_path_txt, "a", encoding="utf-8") as file:
        file.write(txt_line + "\n")

    print(f"Результаты эксперимента '{experiment_name}' успешно сохранены в TXT.")
