## Lossy Counting

Lossy counting [1] is a memory efficient algorithm for approximate counting. It is used to find the frequent items (or "heavy-hitters") in a dataset. This repository provides an efficient implementation of the algorithm.

### Basic Usage

Suppose we are to find items with relative frequency higher than `s = 1e-4` from a dataset, we can do

```python
import LossyCounter as lc
import testLossyCounter as test

# setup
size = int(1e7)
K = int(1e7)
dataset, count = test.generateTestDataset(size = size, K = K)
# dataset should be list-like, containing items for counting
s = 1e-4

# basic usage example
lossy = lc.LossyCounter(eps = 0.1*s)
lossy.count(dataset)
freqItems = lossy.getFreqItems(threshold = s)
# returns a list of frequent items
```

Here `eps` is the error bound of the relative frequency for the approximate counting. That is, the approximate relative frequency would not differ from the true relative frequency by more than `eps`. Thus `eps` should be smaller than threshold `s`. In this basic example I used `eps = 0.1s`.

### Features

#### Small memory footprint

Lossy counting greatly reduces memory footprint by periodically removing rare items from counting. When used to find frequent 5-grams in 3.6 million Amazon reviews, it **decreases memory footprint from nearly 32GB to 1GB**.

#### Low running time overhead

By introducing a cache, this implementation is about **twice as fast as the basic lossy counting algorithm**. The running time is comparable to that of an exact counter. For finding frequent 5-grams in Amazon reviews, the overhead is about 26s, or 10\% of the exact counter running time.

#### High Accuracy

The frequent items found by lossy counting are almost exact. Again using Amazon review 5-gram example, the algorithm achieves a **F1-score of 0.9989**. 

<!---
The relationship between approximate frequency and true frequency is visualized below. Above the frequency threshold (the dashed line on the right), the approximate frequency is in line with the true value.
-->

### Choice of Parameters

While the performance is overall robust to the choice of parameters, good choice still offers improvement. As a rule of thumb, I recommend setting `eps` to be `s/2` (`s` being frequency threshold) and `flush_limit` to be `math.ceil(5/eps)`. This usually offers nice tradeoff between space and accuracy.

### Reference

[1] Manku, G. S., & Motwani, R. (2002, January). Approximate frequency counts over data streams. In VLDB'02: Proceedings of the 28th International Conference on Very Large Databases (pp. 346-357). Morgan Kaufmann.
