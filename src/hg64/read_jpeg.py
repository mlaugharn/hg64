import os
import threading
import time

import datasets
import jpeglib
import numpy as np
import tqdm

import hg64


def read_jpeg(path):
    """
    Read JPEG image from file.
    :param path:
    :return: Y, CbCr, quantization tables
    """
    im = jpeglib.read_dct(path)
    Y = im.Y
    CbCr = np.stack((im.Cb, im.Cr), axis=0)
    qtables = im.qt
    return Y, CbCr, qtables


def read_jpeg_and_add_histogram(hgY, hgCb, hgCr, path):
    Y, CbCr, qtables = read_jpeg(path)
    hgY.batch_add(
        (1024 + Y).flatten().tolist(),
        (np.ones(np.prod(Y.shape), dtype=np.uint64)).tolist(),
    )
    hgCb.batch_add(
        (1024 + CbCr[0]).flatten().tolist(),
        (np.ones(np.prod(CbCr[0].shape), dtype=np.uint64)).tolist(),
    )
    hgCr.batch_add(
        (1024 + CbCr[1]).flatten().tolist(),
        (np.ones(np.prod(CbCr[1].shape), dtype=np.uint64)).tolist(),
    )


def test_against_dataset(dataset, sigbits):
    hgY = hg64.Histogram(sigbits)
    hgCb = hg64.Histogram(sigbits)
    hgCr = hg64.Histogram(sigbits)

    for path in tqdm.tqdm(dataset):
        read_jpeg_and_add_histogram(hgY, hgCb, hgCr, path)

    return hgY, hgCb, hgCr


def test_against_dataset_parallel(dataset, sigbits, n_threads):
    def run_many(args):
        for (hgY, hgCb, hgCr, path) in args:
            read_jpeg_and_add_histogram(hgY, hgCb, hgCr, path)

    hgY = hg64.Histogram(sigbits)
    hgCb = hg64.Histogram(sigbits)
    hgCr = hg64.Histogram(sigbits)
    print("Testing with %d threads and %d bits" % (n_threads, sigbits))
    args = [(hgY, hgCb, hgCr, path) for path in dataset]
    allotment = len(args) // n_threads
    args = [args[i * allotment : (i + 1) * allotment] for i in range(n_threads)]
    threads = [
        threading.Thread(target=run_many, args=[args[i]]) for i in range(n_threads)
    ]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"{dt:.3f}s")

    tot_bytes = sum(os.path.getsize(path) for path in dataset)
    print(f"{tot_bytes / dt / 1e6:.3f} MB/s")

    return hgY, hgCb, hgCr, dt


if __name__ == "__main__":
    dataset = datasets.load_dataset("beans", split="train")["image_file_path"]
    print(len(dataset))
    hgY, hgCb, hgCr, dt = test_against_dataset_parallel(dataset, 6, 2)
    tot_sizes = hgY.size() + hgCb.size() + hgCr.size()
    print(f"Total size: {tot_sizes / 1e6:.3f} MB")
    print(f"information generated speed: {tot_sizes / dt :.3f} B/s")

    print("Done")
