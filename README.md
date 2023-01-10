# hg64

example code:

```
hg = hg64.Histogram(2)  # create histogram with 2 significant bits
hg.add(3, 1)
finished, pmin, pmax, pcount = hg.get(3)
print(hg.size())
```

install:  clone and run `pip install .`

license: mpl-2.0 - fork of https://github.com/fanf2/hg64
