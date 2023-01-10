import math
import random
import threading
import time
import unittest

import hg64

MAX = 1000 * 1000 * 1000
THREAD_SAMPLES = 1000 * 1000


class TestHg64(unittest.TestCase):
    def test_hg64(self):
        for SIGBITS in (2, 3, 5):
            for N_THREADS in (
                1,
                2,
                4,
                12,
            ):
                print("Testing with %d threads and %d bits" % (N_THREADS, SIGBITS))
                hg = hg64.Histogram(SIGBITS)
                increments = list(range(THREAD_SAMPLES))
                data = [
                    [random.randint(0, MAX) for _ in range(THREAD_SAMPLES)]
                    for _ in range(N_THREADS)
                ]  # generate some lists of random ints
                threads = [
                    threading.Thread(target=load_data, args=(hg, d, increments))
                    for d in data
                ]  # create N_THREADS threads to load the data into the histogram
                t0 = time.perf_counter()
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()  # wait for all threads to finish
                t1 = time.perf_counter()
                print(f"{t1 - t0:.3f}s")
                summarize(hg)

                # self.assertLess(relative_error, 0.3)

                # performance: megabytes per second
                print(f"{(N_THREADS * THREAD_SAMPLES) / (t1 - t0) / 1e6:.3f} MB/s")

                # nanoseconds per item:
                print(f"{(t1 - t0) / (N_THREADS * THREAD_SAMPLES) * 1e9:.3f} ns/item")

                # time batch sampling:
                # ranks = [random.randint(0, THREAD_SAMPLES) for _ in range(THREAD_SAMPLES)]
                # t0 = time.perf_counter()
                # s = batch_sample_from_snapshot(hg, ranks)
                # print(max(s))
                # t1 = time.perf_counter()
                # print(f"{(t1 - t0) / (THREAD_SAMPLES) * 1e9:.3f} ns/item (batch sampling)")

                print("------------------------")


def load_data(hg, data, increments):
    hg.batch_add(data, increments)


# def batch_sample_from_snapshot(hg, ranks):
# snapshot = hg.snapshot()
# print(snapshot.population)
# vals = snapshot.sample_values(ranks)
# print(len(vals))
# del snapshot
# return vals
# return snapshot.sample_values(ranks)


def summarize(hg):
    cur_max = 0
    population = 0
    key = 0

    while True:
        continuing, pmin, pmax, pcount = hg.get(key)
        if not continuing:
            break

        cur_max = cur_max if cur_max > pcount else pcount
        population += pcount
        key = hg.next(key)  # update key to the next key value
    print(f"{hg.sigbits()} sigbits")
    print(f"{hg.size()} bytes")
    print(f"{cur_max=} largest")
    print(f"{population=} samples")
    mean, var = hg.mean_variance()

    expected_mean = (0 + MAX) / 2  # expected mean of the samples
    print(f"expected_mean mean={expected_mean}")
    relative_error = (
        100 * abs(expected_mean - mean) / (mean if mean != 0 else 1)
    )  # calculate relative error

    print(
        f"mean {mean} +/- {math.sqrt(var)} ({relative_error:.2f}% relative error)"
    )  # print relative error as percentage
    return relative_error


if __name__ == "__main__":
    unittest.main()
