# Titanic

Задача: угадать, выжил пассажир или нет. На Kaggle считают accuracy.

Модель — ансамбль: случайный лес, XGBoost и CatBoost, мягкое голосование.
На Kaggle получилось 0.7822.
Все прогоны лежат в `outputs/results.txt`.

## Запуск

Скачать `train.csv` и `test.csv` в папку `data/` со страницы соревнования:
https://www.kaggle.com/competitions/titanic

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Команда из корня проекта. Она прогоняет кросс-валидацию, дописывает строку в журнал и заново пишет `outputs/submission.csv`.

Ноутбук `titanic.ipynb` — разведка и кривая потерь.
