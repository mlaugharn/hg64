import unittest

import numpy as np

from src import hg64

"""
from hg64.h:

    /*
    * Add an arbitrary increment to the value's counter
    */
    void hg64_add(hg64 *hg, uint64_t value, uint64_t inc);

    /*
    * Get information about a counter. This can be used as an iterator,
    * by initializing `key` to zero and incrementing by one or using
    * `hg64_next()` until `hg64_get()` returns `false`. The number of
    * iterations is a little less than `1 << (6 + sigbits)`.
    *
    * If `pmin` is non-NULL it is set to the minimum inclusive value
    * that maps to this counter.
    *
    * If `pmax` is non-NULL it is set to the maximum inclusive value
    * that maps to this counter.
    *
    * If `pcount` is non-NULL it is set to the contents of the counter,
    * which can be zero.
    */
    bool hg64_get(hg64 *hg, unsigned key,
            uint64_t *pmin, uint64_t *pmax, uint64_t *pcount);

    /*
    * Skip to the next key, omitting "bins" of nonexistent counters.
    * This function does not skip counters that exist but are zero.
    * A bin contains `1 << sigbits` counters, and counters are created
    * in bulk one whole bin at a time.
    */
    unsigned hg64_next(hg64 *hg, unsigned key);

pybind11 interface:

    /*
    * Add an arbitrary increment to the value's counter
    */
    void hg64_add(hg64 *hg, uint64_t value, uint64_t inc);

    /*
    * Get information about a counter. This can be used as an iterator,
    * by initializing `key` to zero and incrementing by one or using
    * `hg64_next()` until `hg64_get()` returns `false`. The number of
    * iterations is a little less than `1 << (6 + sigbits)`.
    *
    * If `pmin` is non-NULL it is set to the minimum inclusive value
    * that maps to this counter.
    *
    * If `pmax` is non-NULL it is set to the maximum inclusive value
    * that maps to this counter.
    *
    * If `pcount` is non-NULL it is set to the contents of the counter,
    * which can be zero.
    */
    bool hg64_get(hg64 *hg, unsigned key,
            uint64_t *pmin, uint64_t *pmax, uint64_t *pcount);

    /*
    * Skip to the next key, omitting "bins" of nonexistent counters.
    * This function does not skip counters that exist but are zero.
    * A bin contains `1 << sigbits` counters, and counters are created
    * in bulk one whole bin at a time.
    */
    unsigned hg64_next(hg64 *hg, unsigned key);

"""


class NdHg64:
    def __init__(self, shape, approximated_axes, sigbits):
        self.shape = np.array(shape)
        self.approximated_axes = approximated_axes
        print(f"{self.approximated_axes=}")
        self.nonapproximated_axes = tuple(
            ix for ix in range(len(shape)) if ix not in approximated_axes
        )
        print(f"{self.nonapproximated_axes=}")
        self.sigbits = sigbits
        self.histograms = [
            hg64.Histogram(self.sigbits)
            for _ in range(np.prod(self.shape[self.nonapproximated_axes]))
        ]
        self.approximated_strides = np.array(
            np.cumprod(
                np.array(
                    [
                        size
                        for ax, size in enumerate(self.shape)
                        if ax in self.approximated_axes
                    ]
                )[::-1]
            )[::-1]
        )  # compute strides for indexing the flat histogram
        print(f"{self.approximated_strides=}")

    def __getitem__(self, indices):
        where_in_histogram, which_histogram = self.non_and_approx_indices(indices)
        print(f"{which_histogram=}")
        print(f"{where_in_histogram=}")
        histogram = self.histograms[which_histogram]
        success, pmin, pmax, pcount = histogram.get(where_in_histogram)
        print(pcount)
        if success:
            return pcount
        else:
            return 0

    def __setitem__(self, indices, value):
        where_in_histogram, which_histogram = self.non_and_approx_indices(indices)
        print(f"{which_histogram=}")
        print(f"{where_in_histogram=}")
        histogram = self.histograms[which_histogram]
        histogram.add(where_in_histogram, value)

    def non_and_approx_indices(self, indices):
        which_histogram = np.sum(
            [
                self.nonapproximated_axes[ax] * ix
                for ax, ix in enumerate(indices)
                if ax in self.nonapproximated_axes
            ]
        )
        where_in_histogram = 0
        approx_stride_idx = 0
        for ax, ix in enumerate(indices):
            if ax in self.approximated_axes:
                where_in_histogram += self.approximated_strides[approx_stride_idx] * ix
                approx_stride_idx += 1
        return where_in_histogram, which_histogram

    def __array__(self):
        arr = np.empty(self.shape)
        print(self.shape)
        for indices in np.ndindex(tuple(self.shape)):
            print(f"__array__: {indices=}")
            arr[indices] = self[indices]
        return arr

    def next(self, index):
        where_in_hist, which_hist = self.non_and_approx_indices(index)
        hist = self.histograms[which_hist]
        return which_hist, hist.next(where_in_hist)

    def from_non_and_approx_indices(self, non_approximated_index, approximated_index):
        out_idx = []
        for ax, size in enumerate(self.shape):
            if ax in self.nonapproximated_axes:
                out_idx.append(non_approximated_index // size)
                non_approximated_index %= size
        for ax, size in enumerate(self.shape):
            if ax in self.approximated_axes:
                out_idx.append(approximated_index // size)
                approximated_index %= size
        return tuple(out_idx)


class TestNdHg64(unittest.TestCase):
    shape = (5, 6, 7)
    approximated_axes = (
        1,
        2,
    )
    sigbits = 15
    ndhg = NdHg64(shape, approximated_axes, sigbits)

    print(ndhg[0, 0, 0])
    print(ndhg)

    np.all(ndhg)

    # test the histogram is empty
    assert not np.any(ndhg), np.nonzero(ndhg.__array__())

    # test the histogram can be assigned to
    ndhg[2, 3, 4] += 1
    assert ndhg[2, 3, 4] == 1, ndhg[2, 3, 4]

    # test that the histogram can be iterated through
    which_hist, key = ndhg.next((0, 0, 0))
    print(f"{which_hist=} {key=}")
    assert key == 1 and which_hist == 0
    dst = ndhg.from_non_and_approx_indices(which_hist, key)
    print(f"{dst=}")
    assert ndhg[dst] == 1

    # test that the histogram can be iterated through until the end
    last_key = key
    while key != 0:
        last_key = key
        key = ndhg.histogram.next(key)
    assert last_key == ndhg.histogram.size - 1

    # test the histogram can be indexed
    assert hg[(2, 3, 4)] == 1

    # test the histogram can be indexed after being assigned to
    hg[(2, 3, 4)] = 2
    assert hg[(2, 3, 4)] == 2

    # test that the histogram can be iterated through after being assigned to
    key = hg.histogram.next(0)
    assert key == 1
    assert hg.histogram[key] == 2

    # test that the histogram can be iterated through until the end after being assigned to
    last_key = key
    while key != 0:
        last_key = key
        key = hg.histogram.next(key)
    assert last_key == hg.histogram.size - 1

    # test the histogram can be indexed after being assigned to
    assert hg[(2, 3, 4)] == 2

    # test the histogram can be converted to a numpy array
    assert np.all(hg.histogram[:] == hg[:])

    # test that the histogram can be iterated through after being converted to a numpy array
    key = hg.histogram.next(0)
    assert key == 1
    assert hg.histogram[key] == 2

    # test that the histogram can be iterated through until the end after being converted to a numpy array
    last_key = key
    while key != 0:
        last_key = key
        key = hg.histogram.next(key)
    assert last_key == hg.histogram.size - 1

    # test the histogram can be indexed after being converted to a numpy array
    assert hg[(2, 3, 4)] == 2


if __name__ == "__main__":
    unittest.main()
