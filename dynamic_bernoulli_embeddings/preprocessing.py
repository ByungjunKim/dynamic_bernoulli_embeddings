"""Implements batching logic"""
from collections import Counter

import numpy as np
import pandas as pd
import torch


class Data:
    """Class for easy batched iteration over the dataset

    Batches consist of a roughly equal proportion of documents from each time slice. In
    other words, if we want to split the dataset into `m` batches, each batch will have
    1 / `m` proportion of docs from each time slice.

    Attributes
    ----------
    unigram_logits : list
        Unigram distribution to be used for negative sampling.
    T : int
        The number of time slices.
    """

    def __init__(
        self, df, dictionary, time_col="time", bow_col="bow", m=100, cs=5 #6
    ):
        """
        Parameters
        ----------
        df : `pd.DataFrame`
            Pandas dataframe with at least two columns
        dictionary : dict
            Dictionary to use when generating indices from the token bag of words.
        time_col : str
            Name of column corresponding to time buckets. Expected to be integers from
            0 ... T where T is the total number of time buckets.
        bow_col : str
            Name of column corresponding to bag of words, which is just a list of words.
        no_below : int
            Drop any words appearing in fewer than this many rows.
        cs : int
            Context size.
        """
        self.cs = cs
        self.dictionary = dictionary
        self.N = df.shape[0]
        self.ctx = None

        # Generate bow with token indices and remove all unknown words.
        bow_filtered = df[bow_col].apply(
            lambda x: list(
                filter(lambda x: x is not None, [dictionary.get(w, None) for w in x])
            )
        )
        tfs = Counter(word for row in bow_filtered for word in row)

        # Apply a scaling exponent of 3/4 as recommended to generate the unigram
        # distribution for negative sampling.
        scaled_tfs = np.array([cnt for _, cnt in sorted(tfs.items())]) ** 0.75
        total = scaled_tfs.sum()
        self.unigram_logits = torch.tensor(
            [np.log(cnt / (total - cnt)) for cnt in scaled_tfs]
        )

        # Token counts per timestep.
        df_idx = pd.DataFrame({"time": df[time_col], "bow": bow_filtered})
        df_idx = df_idx[bow_filtered.apply(len) > 1]
        m_t = {}
        for t, group in df_idx.groupby(time_col):
            m_t[t] = group[bow_col].apply(len).sum()
        self.m_t = m_t
        self.T = len(m_t)
        self.df_idx = df_idx
        self.m = m

        # EDIT
        self.make_items()

    def _context_mask(self, N):
        """Generates a mask to be used for fetching context vectors

        Parameters
        ----------
        N : int
            The length of the sequence

        Returns
        -------
        ctx : `numpy.ndarray`
            The context mask of shape (N, context_size * 2 + 1). This contains the
            indices of the context for each target. For example, if we have a context
            size of 2, the n-th row in this matrix will be [n-2, n-1, n+1, n+2].
        oob : `numpy.ndarray`
            This is a boolean mask indicating which positions are out of bounds for the
            text. For example, if we have a context size of 2, the 0-th row in this
            matrix will be [True, True, False, False] because the first word in the text
            has no pre-context.
        """
        if self.ctx is None or self.ctx.shape[0] < N:
            self.ctx = (
                np.tile(np.arange(N), (self.cs * 2, 1)).T
                + np.delete(np.arange(2 * self.cs + 1), self.cs)
                - self.cs
            )
        ctx = self.ctx[:N]
        oob = (ctx > N) | (ctx < 0)
        return ctx, oob

    def make_items(self):

        # Build separate generators for each of the time slices.
        token_generators = {}
        for t, group in self.df_idx.groupby("time"):
            token_generators[t] = (
                word
                for _, row in group.sample(group.shape[0]).iterrows()
                for word in row.bow
            )
        self.batch_targets = []
        self.batch_contexts = []
        self.batch_times = []
        for _ in range(self.m):
            for t, token_gen in token_generators.items():
                targets = []
                for word in token_gen:
                    targets.append(word)
                    # +1 to avoid a very small final batch due to rounding
                    if len(targets) == self.m_t[t] // self.m + 1:
                        break
                targets = np.array(targets)
                if len(targets) == 0:
                    self.batch_targets = np.concatenate(self.batch_targets)
                    self.batch_contexts = np.concatenate(self.batch_contexts)
                    self.batch_times = np.concatenate(self.batch_times)
                    return  # We've reached the end.
                mask, oob = self._context_mask(len(targets))
                # Use mode="clip" with np.take so that the oob indices don't cause a
                # failure. Set the oob word indices to -1.
                contexts = np.take(targets, mask, mode="clip")
                contexts[oob] = -1
                self.batch_targets.append(targets)
                self.batch_contexts.append(contexts)
                self.batch_times.append(np.repeat(t, len(targets)))
        self.batch_targets = np.concatenate(self.batch_targets)
        self.batch_contexts = np.concatenate(self.batch_contexts)
        self.batch_times = np.concatenate(self.batch_times)


class DBEDataloader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.N = data.N
        self.batch_targets = torch.from_numpy(data.batch_targets).cuda()
        self.batch_contexts = torch.from_numpy(data.batch_contexts).cuda()
        self.batch_times = torch.from_numpy(data.batch_times).cuda()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (self.batch_targets[idx], self.batch_contexts[idx], self.batch_times[idx])

