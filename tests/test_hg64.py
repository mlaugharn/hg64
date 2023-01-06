import math
import random
import threading
import time
import unittest

import hg64

MAX = 10_000_000
THREAD_SAMPLES = 10_000_000
N_THREADS = 2
SIGBITS = 9


class TestHg64(unittest.TestCase):
    def test_hg64(self):
        hg = hg64.Histogram(SIGBITS)
        data = [
            [random.randint(0, MAX) for _ in range(THREAD_SAMPLES)]
            for _ in range(N_THREADS)
        ]  # generate some lists of random ints
        threads = [
            threading.Thread(target=load_data, args=(hg, d)) for d in data
        ]  # create N_THREADS threads to load the data into the histogram
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()  # wait for all threads to finish
        t1 = time.perf_counter()
        print(f"{t1 - t0:.3f}s")
        relative_error = summarize(hg)

        self.assertLess(relative_error, 0.1)

        # performance: megabytes per second
        print(f"{(N_THREADS * THREAD_SAMPLES) / (t1 - t0) / 1e6:.3f} MB/s")


def load_data(hg, data):
    for i in range(THREAD_SAMPLES):
        hg.add(data[i], 1)


def summarize(hg):
    max = 0
    population = 0
    key = 0

    while True:
        continuing, pmin, pmax, pcount = hg.get(key)
        if not continuing:
            break

        max = max if max > pcount else pcount
        population += pcount
        key = hg.next(key)  # update key to the next key value
    print(f"{hg.sigbits()} sigbits")
    print(f"{hg.size()} bytes")
    print(f"{max=} largest")
    print(f"{population=} samples")
    mean, var = hg.mean_variance()

    expected_mean = (0 + MAX) / 2  # expected mean of the samples
    print(f"expected_mean mean={expected_mean}")
    relative_error = 100 * abs(expected_mean - mean) / mean  # calculate relative error

    print(
        f"mean {mean} +/- {math.sqrt(var)} ({relative_error:.2f}% relative error)"
    )  # print relative error as percentage
    return relative_error


if __name__ == "__main__":
    unittest.main()
