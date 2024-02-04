import LossyCounter as lc
from collections import Counter
import numpy as np
import math

def generateTestDataset(size = int(1e7), K=350000, a=2.7):
    """
    Generate dataset and correct count
    for testing the lossy counter. The
    dataset is generated according to
    Zipf law, with the probability of
    the i-th outcome being observed
    being

    P_{i} \propto \frac{1}{a + i}

    or

    P(i) ~ 1/(a+i)

    with i = 1,2,3,...,K

    Parameters:
    -----------

    size: int, default 1e7
        The total number of observations
        in the generated dataset. That
        would correspond to total word
        count.

    K: int, default 350,000
        Number of unique keys (outcomes)
        For example, there are 350,000
        different words in English [1]

    a: float, default 2.7
        Parameter for the Zipf law, for
        English, it would be about 2.7

    Returns:
    --------

    dataset: list
        A list of keys observed in the
        dataset.

    count: Counter
        The exact count of each key in
        the dataset.

    Reference:
    ----------

    [1] https://en.wikipedia.org/wiki/List_of_dictionaries_by_number_of_words
    """

    #distribution according to Zipf law
    p = 1/(a+np.arange(K))
    p /= p.sum()

    #generate dataset
    dataset = np.random.choice(K,size=size,p=p)

    #exact count
    count = Counter(dataset)

    return dataset, count
    
def testLossyCount(eps = 1e-6,
                  prune_limit = None, 
                  flush_limit = None, 
                  chunk_size = None, 
                  size = int(1e7), 
                  K=350000, 
                  a=2.7):
    
    dataset, count = generateTestDataset(size, K, a)

    #-----------------------
    # test count cache flush
    #-----------------------
    
    lossy = lc.LossyCounter(eps, prune_limit, flush_limit)
    lossy.count(dataset)
    assert lossy.total == size

    keys = [i for i in range(K+1)]
    
    #---------------
    # test getCounts
    #---------------
    
    lower = lossy.getCounts(keys, 'lower')
    wrong_keys = [k for k in range(K+1) if 
                  count[k] < lower[k]]
    assert len(wrong_keys) == 0

    upper = lossy.getCounts(keys, 'upper')
    wrong_keys = [k for k in range(K+1) if 
                  count[k] > upper[k]]
    assert len(wrong_keys) == 0

    median = lossy.getCounts(keys, 'median')
    wrong_keys = [k for k in range(K+1) if 
                  np.abs(2*median[k]-lower[k]-upper[k]) > 1e-8]
    assert len(wrong_keys) == 0

    try:
        not_implemented = lossy.getCounts(keys, 'foobar')
        raise RuntimeError
    except NotImplementedError:
        pass
        
    #------------------
    # test getFreqItems
    #------------------
    
    thresholds = [eps/2, eps, eps*2]
    approxs = ['lower','upper','median']
    for threshold in thresholds:
        for approx in approxs:
            #print(approx)
            freq_items_frac = lossy.getFreqItems(threshold, approx)
            count_threshold = threshold*size
            if count_threshold > 1:
                freq_items_count = lossy.getFreqItems(count_threshold, approx)
                #print(lossy.counter)
                #print(freq_items_frac)
                #print(freq_items_count)
                #print([k for k in freq_items_count if k not in freq_items_frac])
                assert set(freq_items_frac) == set(freq_items_count)
            approx_count = lossy.getCounts([k for k in lossy.counter], approx)
            freq_items_true = [k for k in approx_count
                               if approx_count[k] > threshold * size
                               and count[k] > 0.5 ]
            #print([k for k in freq_items_true if k not in freq_items_frac])
            assert set(freq_items_frac) == set(freq_items_true)

    #------------------------
    # test getCountsAndErrors
    #------------------------
    
    count_and_error = lossy.getCountsAndErrors(keys)
    wrong_keys = [k for k in range(K) if 
                  lower[k] != count_and_error[k]['count'] or 
                  upper[k] - lower[k] != count_and_error[k]['error']]
    assert len(wrong_keys) == 0
    
    #---------------
    # test getBounds
    #---------------
    
    bounds = lossy.getBounds([i for i in range(K+1)])
    wrong_keys = [k for k in range(K) if 
                  count[k] > bounds[k]['upper'] or 
                  count[k] < bounds[k]['lower']]
    assert len(wrong_keys) == 0
    assert bounds[K]['lower'] == 0
    assert bounds[K]['upper'] <= size*eps
    
    
if __name__ == '__main__':
    eps_list = [0.0123, 0.000456]
    factor_list = [1,10]
    size_list = [789, 12345, int(1e5)]
    K_list = [10, 100000]
    for eps in eps_list:
        w = math.ceil(int(1/eps))
        for prune_factor in factor_list:
            prune_limit = w*prune_factor
            for flush_factor in factor_list:
                flush_limit = w*flush_factor
                for chunk_factor in factor_list:
                    chunk_size = w*chunk_factor
                    for size in size_list:
                        for K in K_list:
                            testLossyCount(eps = eps, 
                                           prune_limit = prune_limit, 
                                           flush_limit = flush_limit, 
                                           chunk_size = chunk_size, 
                                           size = size, 
                                           K = K
                                          )
