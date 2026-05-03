import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import os

def kernel(z):
    z = np.asarray(z)
    return np.where(np.abs(z) <= 1, 0.75 * (1 - z**2), 0.0)

def worker_numerator(weights, y_train, chis_shared):
    chis_shared.value = float(np.sum(weights * y_train))


def worker_denominator(weights, znam_shared):
    znam_shared.value = float(np.sum(weights))


# 1 процесс
def sequential_num_den(weights, y_train):
    start = time.perf_counter()

    chis = 0.0
    for i in range(len(weights)):
        chis += weights[i] * y_train[i]

    znam = 0.0
    for i in range(len(weights)):
        znam += weights[i]

    elapsed = time.perf_counter() - start

    return float(chis), float(znam), elapsed


# 2 процесса
def parallel_2proc(weights, y_train):
    chis_shared = mp.Value('d', 0.0)
    znam_shared = mp.Value('d', 0.0)

    p1 = mp.Process(target=worker_numerator, args=(weights, y_train, chis_shared))
    p2 = mp.Process(target=worker_denominator, args=(weights, znam_shared))

    start = time.perf_counter()

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    elapsed = time.perf_counter() - start

    return chis_shared.value, znam_shared.value, elapsed


# 4 процесса
def parallel_4proc(weights, y_train):
    n = len(weights)
    mid = n // 2

    # shared для частей
    chis1 = mp.Value('d', 0.0)
    chis2 = mp.Value('d', 0.0)
    znam1 = mp.Value('d', 0.0)
    znam2 = mp.Value('d', 0.0)

    # процессы
    p1 = mp.Process(target=worker_numerator,
                    args=(weights[:mid], y_train[:mid], chis1))

    p2 = mp.Process(target=worker_numerator,
                    args=(weights[mid:], y_train[mid:], chis2))

    p3 = mp.Process(target=worker_denominator,
                    args=(weights[:mid], znam1))

    p4 = mp.Process(target=worker_denominator,
                    args=(weights[mid:], znam2))

    start = time.perf_counter()

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    elapsed = time.perf_counter() - start

    chis = chis1.value + chis2.value
    znam = znam1.value + znam2.value

    return chis, znam, elapsed


# --- измерение ---
def measure(x, y, beta, x_grid):
    times_1, times_2, times_4 = [], [], []

    for x0 in x_grid:
        n = len(x)
        h = beta * n ** (-1 / 5)

        z = (x0 - x) / h
        weights = kernel(z)

        if np.sum(weights) == 0:
            continue

        _, _, t1 = sequential_num_den(weights, y)
        _, _, t2 = parallel_2proc(weights, y)
        _, _, t4 = parallel_4proc(weights, y)

        times_1.append(t1)
        times_2.append(t2)
        times_4.append(t4)

    return np.mean(times_1), np.mean(times_2), np.mean(times_4)


if __name__ == "__main__":
    print("CPU:", mp.cpu_count())
    print("PID:", os.getpid())

    np.random.seed(42)

    n = 5000000
    x = np.random.uniform(-3, 7, n)
    noise = np.random.normal(0, 0.3, n)
    y = 0.1*(x-4)*np.cos(x) + 0.5*x + noise

    beta = 1.0
    x_grid = np.linspace(-3, 7, 20)

    t1, t2, t4 = measure(x, y, beta, x_grid)

    # ускорение
    s2 = t1 / t2
    s4 = t1 / t4

    print("T1:", t1)
    print("T2:", t2)
    print("T4:", t4)
    print("S2:", s2)
    print("S4:", s4)

    # график времени
    plt.figure()
    plt.bar(["1 процесс", "2 процесса", "4 процесса"], [t1, t2, t4])
    plt.ylabel("Время (сек)")
    plt.title("Сравнение времени (только num+den)")
    plt.grid(axis="y")
    plt.show()

    # график ускорения
    plt.figure()
    plt.bar(["2 процесса", "4 процесса"], [s2, s4])
    plt.ylabel("Ускорение")
    plt.title("Ускорение относительно 1 процесса")
    plt.grid(axis="y")
    plt.show()
