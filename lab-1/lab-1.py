from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, binom, poisson, uniform, norm, expon

# Загрузка данных из файла
data_set_1 = pd.read_csv("set_1.csv", header=None)
data_set_1 = data_set_1.values.flatten()

# Построение гистограммы
plt.hist(data_set_1, bins=10, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма данных из set_1.csv")
plt.show()

# Вычисление математического ожидания и дисперсии
mean_set_1 = np.mean(data_set_1)
variance_set_1 = np.var(data_set_1)

# Генерация случайных величин с использованием теоретических распределений
random_geom = geom.rvs(0.5, size=len(data_set_1))
random_binom = binom.rvs(10, 0.3, size=len(data_set_1))
random_poisson = poisson.rvs(5, size=len(data_set_1))
random_uniform = uniform.rvs(0, 10, size=len(data_set_1))
random_norm = norm.rvs(5, 2, size=len(data_set_1))
random_expon = expon.rvs(1, size=len(data_set_1))

# Построение гистограммы для сгенерированных данных
plt.hist(random_geom, bins=15, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма для геометрического распределения")
plt.show()

plt.hist(random_binom, bins=15, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма для биномиального распределения")
plt.show()

plt.hist(random_poisson, bins=15, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма для распределения Пуассона")
plt.show()

plt.hist(random_uniform, bins=15, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма для равномерного распределения")
plt.show()

plt.hist(random_norm, bins=15, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма для нормального распределения")
plt.show()

plt.hist(random_expon, bins=15, density=True, alpha=0.5)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.title("Гистограмма для экспоненциального распределения")
plt.show()


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("The data should contain at least two elements.")

    squared_diffs = [(x - sum(data) / n) ** 2 for x in data]
    return sum(squared_diffs) / (n - 1)


mean_set_1 = mean(data_set_1.tolist())
variance_set_1 = variance(data_set_1.tolist())

print(mean_set_1, variance_set_1)


data_sets = [
    pd.read_csv(filename, header=None).values.flatten() for filename in data_sets_files
]
hist = plt.hist(data_set, bins=15, density=True, alpha=0.5, edgecolor="black")


for i, (data_set, set_name) in enumerate(zip(data_sets, data_sets_files)):

    # Настройка цвета столбцов
    cmap = plt.get_cmap("viridis")
    bin_colors = [cmap(bin_val) for bin_val in bins[:-1]]
    for patch, color in zip(patches, bin_colors):
        patch.set_facecolor(color)

    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма данных из {set_name}")


plt.tight_layout()
plt.show()


def plotting_dataset(dataset: List[float] | List[int], bins_count: int, data_percent: float):
    
    N = len(dataset)
    n = math.ceil(1 + 1.14 * math.log(N))
    
    min_val = min(dataset)
    max_val = max(dataset)
    
    step = (max_val - min_val) / n

    print(n, min_val, max_val, step)

    intervals = {}
    
    for val in dataset:
        index = math.ceil((val - min_val) / step)
        intervals[index] += 1

    keys = sorted(intervals.keys())
    intervals = {key: intervals[key] for key in sorted(intervals)}
    
    for val in keys:
        if intervals[val] / N < data_percent:
            intervals.pop(val)

    keys = sorted(intervals.keys())

    max_val = keys[len(keys) - 1] * step + min_val
    min_val = keys[0] * step + min_val

    plt.hist([val for val in dataset if val <= max_val and val >= min_val], bins=bins_count, color="skyblue", edgecolor="black")

    plt.title("Гистограмма")
    plt.xlabel("Значения")
    plt.ylabel("Частота")

    plt.show()
