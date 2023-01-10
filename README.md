# hg64

example code:

```
hg = hg64.Histogram(2)  # create histogram with 2 significant bits
hg.add(3, 1)
finished, pmin, pmax, pcount = hg.get(3)
print(hg.size())
```

license: mozilla public license 2.0 - this repo derived from https://github.com/fanf2/hg64
