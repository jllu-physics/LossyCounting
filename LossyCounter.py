from collections import Counter
import math
from itertools import islice
from tqdm import tqdm
import warnings

def batched(iterable, n):
    """
    Batch data into lists of length n. 
    The last batch may be shorter.

    From:
    https://github.com/python/cpython/issues/98363
    """
    if n < 1:
        raise ValueError('n must be >= 1')
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch

class LossyCounter:
    """
    Lossy Counting in the spirit of Section 4.2 in:
    
     Manku, G. S., & Motwani, R. (2002, January). 
     Approximate frequency counts over data streams. 
     In VLDB'02: Proceedings of the 28th International 
     Conference on Very Large Databases (pp. 346-357). 
     Morgan Kaufmann.

    Lossy Counting algorithm can find approximate
    frequency of each item within a specified error
    bound, eps, while greatly reducing the memory
    requirement compared with exact counting. This
    is done by periodically prune infrequent items
    during counting.

    This implementation is optimized for speed and
    precision, when the objective is to find all
    frequent items with frequency above certain
    threshold.

    Parameters:
    -----------

    eps: float, default 1e-6
        RELATIVE error bound for the frequency of
        each item.
        Smaller eps means more accurate results,
        but also requires more space.

    prune_limit: int, default None
        Threshold for the size of the internal
        counter. When its size is larger than
        prune_limit, it would be pruned, removing
        infrequent items.
        If None, prune_limit is set to
            10*CEIL(1/eps)
        Larger prune_limit means faster processing
        and more accurate results, but also takes
        more space.

    flush_limit: int, default None
        Threshold for the total count of items in
        cache. If we go above the threshold,
        we flush the frequent items in cache into
        the internal counter.

    Attributes:
    -----------

    eps: float
        Attribute remembering the eps specified in
        parameters

    w: int
        Equal to 
            CEIL(1/eps)
        Quantity defined in the original paper,
        called bucket width

    total: int
        The total number of items that the counter
        has seen. Note as the counter is lossy (not
        all elements are counted), the total counts
        from the counter is in general unequal to
        the total attribute.

    """

    def __init__(self, 
                 eps = 1e-6, 
                 prune_limit = None, 
                 flush_limit = None
                ):
        
        self.eps = eps
        self.w = math.ceil(1/eps)
        self._cache_counter = Counter()
        self._cache_total = 0
        self.counter = Counter()
        self.error = {}
        self.total = 0
        self._nbucket = 0
        
        if prune_limit is None:
            self.prune_limit = 10*self.w
        else:
            self.prune_limit = prune_limit
        if flush_limit is None:
            self.flush_limit = 10*self.w
        else:
            self.flush_limit = flush_limit

    def count(self, iterable, chunk_size = None):
        """
        Perform lossy counting on the iterable. The
        elements of iterable should be the items to
        be counted.

        Parameters:
        -----------

        iterable: iterable data type
            An iterable of items to be counted. For
            example, when counting words, iterable
            could be ['the', 'big', 'brown', 'fox']

        chunk_size: int, default None
            The maximal number of items in a chunk.
            The iterable is fed to the counter in
            chunks, in case it contains too many
            items and cannot fit in memory at once.

        Returns:
        --------
        Nothing. This method changes the inner state
        of the object.
        """
        
        if chunk_size is None:
            chunk_size = self.flush_limit
        for chunk in tqdm(batched(iterable, chunk_size)):
            self.cache(chunk)
        self.flush()
        if len(self.counter) < self.prune_limit:
            self.prune()

    def cache(self, chunk):
        """
        Count items in chunk and put them in cache.
        When the total count is above flush_limit,
        the counts are flushed into main counter.

        Parameters:
        -----------

        chunk: list-like
            The elements are the items to be counted

        Returns:
        --------
        Nothing. This method changes the inner state
        of the object.
        """
        
        self._cache_counter.update(chunk)
        self._cache_total += len(chunk)
        
        #if len not available:
        #self._cache_total = self._cache_counter.total()
        if self._cache_total >= self.flush_limit:
            self.flush()
    

    def flush(self):
        """
        Flush item counts in cache into internal
        counter. 
        """
        
        # b is the number of buckets we have in the
        # the cache. It is also the threshold in
        # count. Only keys with count more than b
        # need to be added, if they are not seen
        # before (or seen but pruned)
        b = self._cache_total // self.w
        
        for key in self._cache_counter:
        
            # If the key has been seen before
            # accumulate count
            # We don't worry here if they become
            # too rare to be worth tracking.
            # That is the task of pruning part.
            if key in self.counter:
                self.counter[key] += self._cache_counter[key]
                
            # If the key is not currently in counter
            # We add it if it is frequent enough in
            # cache.
            # This could mean new key, but also old
            # key pruned because they were rare.
            elif self._cache_counter[key] > b:
                self.counter[key] = self._cache_counter[key]
                self.error[key] = self._nbucket
                
        self._nbucket += b
        self.total += self._cache_total
        
        # remove everything from cache
        self._cache_counter = Counter()
        self._cache_total = 0
        
        if len(self.counter) >= self.prune_limit:
            self.prune()

    def prune(self):
        """
        Remove infrequent items currently in the
        counter.
        """
        
        # It would be nice if we could do
        #
        # for key in self.counter:
        # 	if self.counter[key] + self.error[key] \
        #        <= self._nbucket:
        #       	del self.counter[key]
        #		del self.error[key]
        #
        # But we cannot, we cannot modify a dictionary
        # when we are iterating through it.
        
        # These are the infrequent keys
        # If removed (i.e. setting count to 0)
        # the error is within eps tolerance
        keys = [key for key in self.counter 
                if self.counter[key] + self.error[key] 
                <= self._nbucket
               ]
        
        for key in keys:
            del self.counter[key]
            del self.error[key]

    def getCounts(self, keys, approx = 'lower'):
        """
        Get the approximate counts of the items
        listed in keys.

        Parameters:
        -----------

        keys: iterable
            contains the keys (items) whose
            counts are requested.

        approx: string, could be
            - 'lower': lower bound
            - 'upper': upper bound
            - 'median': middle point between
                lower bound and upper bound
            By default, approx is 'lower'.

        Returns:
        --------

        result: a dictionary where the keys
            are the items and the values are
            the (approximate) counts
        """
        
        result = {}
        if approx == 'lower':
            for key in keys:
                if key in self.counter:
                    result[key] = self.counter[key]
                else:
                    result[key] = 0
        elif approx == 'upper':
            for key in keys:
                if key in self.counter:
                    result[key] = self.counter[key] + \
                                  self.error[key]
                else:
                    result[key] = self._nbucket
        elif approx == 'median':
            for key in keys:
                if key in self.counter:
                    result[key] = self.counter[key] + \
                                  self.error[key]/2
                else:
                    result[key] = self._nbucket/2
        else:
            raise NotImplementedError
        return result

    def getFreqItems(self, threshold = None, approx = 'lower'):
        """
        Get a list of items whose approximate 
        frequencies are above specified 
        threshold.

        Parameters:
        -----------

        threshold: int or float, default None
            If smaller than 1, threshold is
            interpreted as relative frequency
            Otherwise, it is taken as count. 
            If None, threshold is set to be 
            eps.
            
        approx: string, could be
            - 'lower': lower bound
            - 'upper': upper bound
            - 'median': middle point between
                lower bound and upper bound
            By default, approx is 'lower'.

        Returns:
        --------

        result: a list of items
        """
        
        if threshold is None:
            threshold = self.eps
        if threshold < 1:
            threshold *= self.total
            
        if approx == 'lower':
            #if threshold < self.eps * self.total:
            #    warnings.warn("Threshold lower than error tolerance. The result might be incomplete.")
            result = [key for key in self.counter
                      if self.counter[key] > 
                      threshold]
        elif approx == 'upper':
            result = [key for key in self.counter
                      if self.counter[key] +
                      self.error[key] > threshold]
        elif approx == 'median':
            #if threshold < self.eps * self.total / 2:
            #    warnings.warn("Threshold lower than error tolerance. The result might be incomplete.")
            result = [key for key in self.counter
                      if self.counter[key] +
                      self.error[key]/2  > threshold]
        else:
            raise NotImplementedError
                  
        return result

    def getCountsAndErrors(self, keys):
        """
        Get the counts and errors of 
        counts of the items listed in keys.

        Parameters:
        -----------

        keys: iterable
            contains the keys (items) whose
            counts and errors are requested.

        Returns:
        --------

        result: a dictionary where the keys
            are the items and the values are
            dictionaries such as
            {'count': lower_bound,
            'error': upper_bound-lower_bound}
        """
        
        result = {}
        for key in keys:
            result[key] = {'count':self.counter[key], 
                           'error':self.error.get(key, self._nbucket)}
        return result

    def getBounds(self, keys):
        """
        Get the lower and upper bounds of 
        counts of the items listed in keys.

        Parameters:
        -----------

        keys: iterable
            contains the keys (items) whose
            bounds are requested.

        Returns:
        --------

        result: a dictionary where the keys
            are the items and the values are
            dictionaries such as
            {'lower': lower_bound,
            'upper': upper_bound}
        """
        
        result = {}
        for key in keys:
            result[key] = {'lower':self.counter[key], 
                           'upper':self.counter[key] + 
                           self.error.get(key, self._nbucket)}
        return result
