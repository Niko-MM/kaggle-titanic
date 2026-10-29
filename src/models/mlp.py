import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TitanicModel(nn.Module):
    """Архитектура моделли, 3 слоя"""

    def __init__(self, input_dim) -> None:
        super().__init__()
        self.line1 = nn.Linear(input_dim, 16)
        self.line2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.line1(x)
        x = self.relu(x)
        x = self.line2(x)
        return x


# Наследуемся, чтобы работал run
class ModelEngine(BaseEstimator, ClassifierMixin):
    """
    Интерфейс модели, адаптирован под pipline sklearn
    """

    def __init__(self, input_dim, lr, epochs, batch_size) -> None:
        # Параметры модели
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Создаем модель
        self.model = TitanicModel(input_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Под историю лоса
        self.loss_history = None
        # Нормализация данных
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Обучаем модель"""
        # Нормализуем данные
        X_scaled = self.scaler.fit_transform(X)
        # Переводим в тензор
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        # Упаковываем данные
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Переводим модель в режим обучения
        self.model.train()
        # Список для хранения лоса по эпохам
        self.loss_history = []

        for epoch in range(self.epochs):
            # лосы батча внутри эпохи
            epoch_losses = []
            for batch_X, batch_y in dataloader:
                # Обнуляем градиенты
                self.optimizer.zero_grad()
                # Прямой проход
                preds = self.model(batch_X)
                # Считаем ошибку
                loss = self.criterion(preds, batch_y)
                # Делаем обратный проход
                loss.backward()
                # Настраиваем веса
                self.optimizer.step()
                # Чистый лосс батча
                epoch_losses.append(loss.item())
            # Средний лосс за всю эпоху. Добавляем в общую
            mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.loss_history.append(mean_epoch_loss)
        return self

    def predict(self, X):
        """Делаем предсказание"""
        # Используем трансформ, чтобы не переобучать
        X_scaled = self.scaler.transform(X)
        # Переводим в тензор
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Переводим в режим предсказания
        self.model.eval()

        # Отсоединяем граф
        with torch.no_grad():
            # Делаем предсказание
            preds = self.model(X_tensor)
            # Переводим результат в бинарные (0/1)
            preds_binary = (preds >= 0.0).int()
            # Переводим обратно в np массив с выпрямлением
            preds_numpy = preds_binary.numpy().flatten()
        return preds_numpy
