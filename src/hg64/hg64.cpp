/*
 * hg64 - 64-bit histograms
 *
 * Written by Tony Finch <dot@dotat.at> <fanf@isc.org>
 *
 * Copyright (C) Internet Systems Consortium, Inc. ("ISC")
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <cassert>
#include <cerrno>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pybind11/pybind11.h>
#include <atomic>
#include "hg64.h"

// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0, https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0?permalink_comment_id=3931912#gistcomment-3931912

#ifdef _MSC_VER
#include <intrin.h>

static inline int __builtin_ctz(unsigned x) {
    unsigned long ret;
    _BitScanForward(&ret, x);
    return (int)ret;
}

static inline int __builtin_ctzll(unsigned long long x) {
    unsigned long ret;
    _BitScanForward64(&ret, x);
    return (int)ret;
}

static inline int __builtin_ctzl(unsigned long x) {
    return sizeof(x) == 8 ? __builtin_ctzll(x) : __builtin_ctz((uint32_t)x);
}

static inline int __builtin_clz(unsigned x) {
    //unsigned long ret;
    //_BitScanReverse(&ret, x);
    //return (int)(31 ^ ret);
    return (int)__lzcnt(x);
}

static inline int __builtin_clzll(unsigned long long x) {
    //unsigned long ret;
    //_BitScanReverse64(&ret, x);
    //return (int)(63 ^ ret);
    return (int)__lzcnt64(x);
}

static inline int __builtin_clzl(unsigned long x) {
    return sizeof(x) == 8 ? __builtin_clzll(x) : __builtin_clz((uint32_t)x);
}

#ifdef __cplusplus
static inline int __builtin_ctzl(unsigned long long x) {
    return __builtin_ctzll(x);
}

static inline int __builtin_clzl(unsigned long long x) {
    return __builtin_clzll(x);
}
#endif
#endif

/* number of bins is same as number of bits in a value */
#define BINS 64

typedef std::atomic<uint_fast64_t> counter;
typedef std::atomic<std::atomic<uint_fast64_t>*> bin_ptr;

struct hg64 {
    unsigned sigbits;
    bin_ptr bin[BINS];

    hg64& operator=(const hg64& other)
    {
        sigbits = other.sigbits;
        for (unsigned i = 0; i < BINS; i++)
        {
            std::atomic_store(&bin[i], other.bin[i]);
        }
        return *this;
    }
};

static inline counter *
get_bin(hg64 *hg, unsigned b) {
    /* key_to_new_counter() below has the matching store / release */
    counter *result;
    result = std::atomic_load_explicit(&hg->bin[b], std::memory_order_acquire);
    return result;
}

/*
 * static snapshot of a histogram extented with summary data
 */
struct hg64s {
    unsigned sigbits;
    uint64_t binmap;
    uint64_t population;
    uint64_t total[BINS];
    uint64_t *bin[BINS];
    uint64_t counters[];
};

/*
 * when we only care about the histogram precision
 */
struct hg64p {
    unsigned sigbits;
};




typedef union hg64u {
    hg64 *hg;
    const hg64s *hc;
    const struct hg64p *hp;
} hg64u;

#define hg64p(hu) ((const struct hg64p *)(hu))

/*
 * The bins arrays have a static size for simplicity, but that means We
 * waste a little extra space that could be saved by omitting the
 * exponents that land in the denormal number bin. The following macros
 * calculate (at run time) the exact number of keys when we need to do
 * accurate bounds checks.
 */
#define DENORMALS(hp) ((hp)->sigbits - 1)
#define EXPONENTS(hp) (BINS - DENORMALS(hp))
#define MANTISSAS(hp) (1 << (hp)->sigbits)
#define KEYS(hp) (EXPONENTS(hp) * MANTISSAS(hp))

#define MAXBIN(hp) EXPONENTS(hp)
#define BINSIZE(hp) MANTISSAS(hp)

/**********************************************************************/

#define OUTARG(ptr, val) (void)(((ptr) != NULL) && (bool)(*(ptr) = (val)))

static inline uint64_t
interpolate(uint64_t span, uint64_t mul, uint64_t div) {
    double frac = (div == 0) ? 1 : (double)mul / (double)div;
    return((uint64_t)(span * frac));
}

/**********************************************************************/

hg64 *
hg64_create(unsigned sigbits) {
    if(sigbits < 1 || 15 < sigbits) {
        return(NULL);
    }
    hg64 *hg = static_cast<hg64 *>(malloc(sizeof(*hg)));
    hg->sigbits = sigbits;
    /*
     * it is probably portable to zero-initialize atomics but the
     * C standard says we shouldn't rely on it; but this loop
     * should optimize to memset() on most target systems
     */
    for (unsigned b = 0; b < BINS; b++) {
        atomic_init(&hg->bin[b], NULL);
    }
    return(hg);
}

void
hg64_destroy(hg64 *hg) {
    for (unsigned i = 0; i < BINS; i++) {
        std::atomic_store(&hg->bin[i], nullptr);
    }
    free(hg);
}

unsigned
hg64_sigbits(hg64 *hg) {
    return(hg->sigbits);
}

size_t
hg64_size(hg64 *hg) {
    size_t bin_bytes = 0;
    for(unsigned b = 0; b < BINS; b++) {
        if(get_bin(hg, b) != NULL) {
            bin_bytes += sizeof(counter) * BINSIZE(hg);
        }
    }
    return(sizeof(hg64) + bin_bytes);
}

/**********************************************************************/

static inline uint64_t
key_to_minval(void* hu, unsigned key) {
    unsigned binsize = BINSIZE(hg64p(hu));
    unsigned exponent = (key / binsize) - 1;
    uint64_t mantissa = (key % binsize) + binsize;
    return(key < binsize ? key : mantissa << exponent);
}

/*
 * don't shift by 64, and don't underflow exponent; instead,
 * reduce shift by 1 for each hazard and pre-shift UINT64_MAX
 */
static inline uint64_t
key_to_maxval(void* hu, unsigned key) {
    unsigned binsize = BINSIZE(hg64p(hu));
    unsigned shift = 63 - (key / binsize);
    uint64_t range = UINT64_MAX/4 >> shift;
    return(key_to_minval(hu, key) + range);
}

/*
 * This branchless conversion is due to Paul Khuong: see bin_down_of() in
 * https://pvk.ca/Blog/2015/06/27/linear-log-bucketing-fast-versatile-simple/
 */
static inline unsigned
value_to_key(void* hu, uint64_t value) {
    /* fast path */
    const struct hg64p *hp = hg64p(hu);
    /* ensure that denormal numbers are all in the same bin */
    uint64_t binned = value | BINSIZE(hp);
    int clz = __builtin_clzll((unsigned long long)(binned));
    /* actually 1 less than the exponent except for denormals */
    unsigned exponent = 63 - hp->sigbits - clz;
    /* mantissa has leading bit set except for denormals */
    unsigned mantissa = value >> exponent;
    /* leading bit of mantissa adds one to exponent */
    return((exponent << hp->sigbits) + mantissa);
}

static counter *
key_to_new_counter(hg64 *hg, unsigned key) {
    /* slow path */
    unsigned binsize = BINSIZE(hg);
    unsigned b = key / binsize;
    unsigned c = key % binsize;
    counter *old_bp = NULL;
    counter *new_bp = static_cast<counter *>(malloc(sizeof(counter) * binsize));
    /* see comment in hg64_create() above */
    for (unsigned i = 0; i < binsize; i++) {
        atomic_init(new_bp + i, 0);
    }
    bin_ptr *bpp = &hg->bin[b];
    if(atomic_compare_exchange_strong_explicit(bpp, &old_bp, new_bp,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return(new_bp + c);
    } else {
        /* lost the race, so use the winner's counters */
        free(new_bp);
        return(old_bp + c);
    }
}

static inline counter *
key_to_counter(hg64 *hg, unsigned key) {
    /* fast path */
    unsigned binsize = BINSIZE(hg);
    unsigned b = key / binsize;
    unsigned c = key % binsize;
    counter *bp = get_bin(hg, b);
    return(bp == NULL ? NULL : bp + c);
}

static inline uint64_t
get_key_count(hg64 *hg, unsigned key) {
    counter *ctr = key_to_counter(hg, key);
    return(ctr == NULL ? 0 :
           atomic_load_explicit(ctr, std::memory_order_relaxed));
}

static inline void
add_key_count(hg64 *hg, unsigned key, uint64_t inc) {
    if(inc == 0) return;
    counter *ctr = key_to_counter(hg, key);
    ctr = ctr ? ctr : key_to_new_counter(hg, key);
    atomic_fetch_add_explicit(ctr, inc, std::memory_order_relaxed);
}


/**********************************************************************/

void
hg64_add(hg64 *hg, uint64_t value, uint64_t inc) {
    add_key_count(hg, value_to_key(hg, value), inc);
}

void
hg64_inc(hg64 *hg, uint64_t value) {
    add_key_count(hg, value_to_key(hg, value), 1);
}

bool
hg64_get(hg64 *hg, unsigned key,
        uint64_t *pmin, uint64_t *pmax, uint64_t *pcount) {
    if(key < KEYS(hg)) {
        OUTARG(pmin, key_to_minval(hg, key));
        OUTARG(pmax, key_to_maxval(hg, key));
        OUTARG(pcount, get_key_count(hg, key));
        return(true);
    } else {
        return(false);
    }
}

unsigned
hg64_next(hg64 *hg, unsigned key) {
    key++;
    while(key < KEYS(hg) &&
          (key & (BINSIZE(hg) - 1)) == 0 &&
          key_to_counter(hg, key) == NULL) {
        key += BINSIZE(hg);
    }
    return(key);
}

/*
 * https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
 */
void
hg64_mean_variance(hg64 *hg, double *pmean, double *pvar) {
    double pop = 0.0;
    double mean = 0.0;
    double sigma = 0.0;
    uint64_t min, max, count;
    for(unsigned key = 0;
        hg64_get(hg, key, &min, &max, &count);
        key = hg64_next(hg, key)) {
        double delta = (double)min / 2.0 + (double)max / 2.0 - mean;
        if(count != 0) { /* avoid division by zero */
            pop += count;
            mean += count * delta / pop;
            sigma += count * delta * (min + max - mean);
        }
    }
    OUTARG(pmean, mean);
    OUTARG(pvar, sigma / pop);
}

/**********************************************************************/

void
hg64_merge(hg64 *target, hg64 *source) {
    uint64_t count;
    for(unsigned skey = 0;
        hg64_get(source, skey, NULL, NULL, &count);
        skey = hg64_next(source, skey)) {
        uint64_t svmin = key_to_minval(source, skey);
        uint64_t svmax = key_to_maxval(source, skey);
        unsigned tkmin = value_to_key(target, svmin);
        unsigned tkmax = value_to_key(target, svmax);
        unsigned keys = tkmax - tkmin + 1;
        /* is there a more cunning way to spread out the remainder? */
        uint64_t div = count / keys;
        uint64_t rem = count % keys;
        for(unsigned tkey = tkmin; tkey <= tkmax; tkey++) {
            uint64_t inc = div + (uint64_t)(tkey < rem);
            add_key_count(target, tkey, inc);
        }
    }
}

hg64s *
hg64_snapshot(hg64 *hg) {
    unsigned binsize = BINSIZE(hg);
    uint64_t binmap = 0;
    size_t bytes = 0;
    /*
     * first find out which bins we will copy across
     * (as a bitmap) and how much space they need
     */
    for(unsigned b = 0; b < BINS; b++) {
        if(get_bin(hg, b) != NULL) {
            binmap |= 1 << b;
            bytes += binsize * sizeof(uint64_t);
        }
    }
    hg64s *hs = static_cast<hg64s *>(malloc(sizeof(hg64s) + bytes));
    memset(hs, 0, sizeof(hg64s) + bytes);
    hs->sigbits = hg->sigbits;
    hs->binmap = binmap;
    /*
     * second, copy the data, using the bin bitmap not get_bin()
     * because concurrent threads may have added new bins
     */
    for(unsigned b = 0; b < BINS; b++) {
        if(((1 << b) & binmap) == 0) {
            continue;
        }
        hs->bin[b] = &hs->counters[binsize * b];
        for(unsigned c = 0; c < binsize; c++) {
            unsigned key = binsize * b + c;
            uint64_t count = get_key_count(hg, key);
            hs->bin[b][c] = count;
            hs->total[b] += count;
            hs->population += count;
        }
    }
    return(hs);
}

/**********************************************************************/

uint64_t
hg64s_value_at_rank(const hg64s *hs, uint64_t rank) {
    unsigned maxbin = MAXBIN(hs);
    unsigned binsize = BINSIZE(hs);
    unsigned b, c;

    for(b = 0; b < maxbin; b++) {
        uint64_t count = hs->total[b];
        if(rank < count) {
            break;
        }
        rank -= count;
    }
    if(b == maxbin) {
        return(UINT64_MAX);
    }

    for(c = 0; c < binsize; c++) {
        uint64_t count = hs->bin[b][c];
        if(rank < count) {
            break;
        }
        rank -= count;
    }
    if(c == binsize) {
        return(UINT64_MAX);
    }

    unsigned key = binsize * b + c;
    uint64_t min = key_to_minval(&hs, key);
    uint64_t max = key_to_maxval(&hs, key);
    uint64_t count = hs->bin[b][c];
    return(min + interpolate(max - min, rank, count));
}

uint64_t
hg64s_rank_of_value(const hg64s *hs, uint64_t value) {
    unsigned key = value_to_key(&hs, value);
    unsigned binsize = BINSIZE(hs);
    unsigned kb = key / binsize;
    unsigned kc = key % binsize;
    uint64_t rank = 0;

    for(unsigned b = 0; b < kb; b++) {
        rank += hs->total[b];
    }
    for(unsigned c = 0; c < kc; c++) {
        rank += hs->bin[kb][c];
    }

    uint64_t count = hs->bin[kb][kc];
    uint64_t min = key_to_minval(&hs, key);
    uint64_t max = key_to_maxval(&hs, key);
    return(rank + interpolate(count, value - min, max - min));
}

uint64_t
hg64s_value_at_quantile(const hg64s *hs, double q) {
    double pop = hs->population;
    double rank = q < 0.0 ? 0.0 : q > 1.0 ? 1.0 : q;
    return(hg64s_value_at_rank(hs, (uint64_t)(rank * pop)));
}

double
hg64s_quantile_of_value(const hg64s *hs, uint64_t value) {
    uint64_t rank = hg64s_rank_of_value(hs, value);
    return((double)rank / (double)hs->population);
}

/**********************************************************************/

void
hg64_validate(void) {
    for(unsigned sigbits = 1; sigbits < 12; sigbits++) {
        struct hg64p hp = { sigbits };
        const struct hg64p *hpp = &hp;
        unsigned maxbin = MAXBIN(hpp);
        unsigned binsize = BINSIZE(hpp);
        unsigned maxkey = KEYS(hpp) - 1;
        uint64_t prev = 0;
        for(unsigned b = 0; b < maxbin; b++) {
            for(unsigned c = 0; c < binsize; c++) {
                unsigned key = binsize * b + c;
                uint64_t min = key_to_minval((void *) hpp, key);
                uint64_t max = key_to_maxval((void *) hpp, key);
                assert(value_to_key((void *) hpp, min) == key);
                assert(value_to_key((void *) hpp, max) == key);
                assert(b == 0 ? min == max : true);
                assert((key == 0) == (min == 0 && max == 0));
                assert((key == maxkey) == (max == UINT64_MAX));
                assert((b > 0 || c > 0) == (prev + 1 == min));
                prev = max;
            }
        }
    }
}
/**********************************************************************/


namespace py = pybind11;

PYBIND11_MODULE(_hg64, m) {
    m.doc() = "64-bit histograms";

    py::class_<hg64>(m, "Histogram")
            .def(py::init<unsigned>())
            .def("sigbits", &hg64_sigbits)
            .def("size", &hg64_size)
            .def("inc", &hg64_inc, py::arg("value"))
            .def("add", &hg64_add, py::arg("value"), py::arg("inc"))
            .def("get", [](hg64 &hg, unsigned key) -> py::tuple {
                uint64_t min, max, count;
                bool ret = hg64_get(&hg, key, &min, &max, &count);
                return py::make_tuple(ret, min, max, count);
            }, py::arg("key"))
            .def("next", &hg64_next, py::arg("key"))
            .def("mean_variance", [](hg64 &hg) -> py::tuple {
                double mean, variance;
                hg64_mean_variance(&hg, &mean, &variance);
                return py::make_tuple(mean, variance);
            })
            .def("snapshot", &hg64_snapshot, py::return_value_policy::reference)
            .def("__del__", &hg64_destroy);

    py::class_<hg64s>(m, "Snapshot")
            .def_readonly("sigbits", &hg64s::sigbits)
            .def_readonly("binmap", &hg64s::binmap)
            .def_readonly("population", &hg64s::population)
            .def_readonly("total", &hg64s::total)
            .def_property_readonly("bin", [](const hg64s &hs) -> py::list {
                py::list result;
                for (unsigned i = 0; i < 64; i++) {
                    result.append(py::cast(hs.bin[i]));
                }
                return result;
            })
            .def_readonly("counters", &hg64s::counters)
            .def("value_at_rank", &hg64s_value_at_rank, py::arg("rank"))
            .def("value_at_quantile", &hg64s_value_at_quantile, py::arg("quantile"));
}
