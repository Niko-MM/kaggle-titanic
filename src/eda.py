import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, title="Кривая обучения (Loss Curve)"):
    """График падения лосса по эпохам для анализа нейросети"""
    plt.figure(figsize=(9, 4.5))

    # Линия изменения ошибки
    plt.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="crimson",
        linewidth=2,
        label="Train Loss",
    )

    # Оформление графика
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Эпоха", fontsize=11)
    plt.ylabel("Loss (BCEWithLogitsLoss)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()
