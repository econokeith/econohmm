from __future__ import division
import numpy.random as npr
import scipy.stats as sps
import numpy.linalg as la
import numpy as np
import sys, os.path
import pandas as pd
import copy

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cystats
# from clustering import gmm_em


class Markov(object):

    @classmethod
    def from_states(cls, states):
        pass

    def __init__(self, t_mat=None):

        if isinstance(t_mat, int):
            self.t_mat = np.ones((t_mat, t_mat)) / t_mat
        elif isinstance(t_mat, list):
            self.t_mat = np.asarray(t_mat)
        else:
            self.t_mat = t_mat

        self.c_mat = None
        self.s_count = None

        self.K = self.t_mat.shape[0]
        self.node_marginal = self.t_mat.sum(axis=0)
        self.node_marginal /= self.node_marginal.sum()

    def sample_states(self, N=100, state0=None, update=False):
        if state0 is None:
            nm = self.node_marginal[:,np.newaxis]
            s0 = cystats.sample_states(nm)[0]
        else:
            s0 = state0
        new_states = cystats.markov_sample(self.t_mat, s0, N)
        if update is True:
            self.update_from_states(new_states)
        return new_states

    def update_from_states(self, states):

        if self.K is None:
            self.K = np.unique(states).shape[0]

        c_mat = np.zeros((self.K, self.K), dtype=np.int32)

        # todo need to fix this
        cystats.switch_count(states.astype(np.int32), c_mat)

        self.c_mat = c_mat
        self.t_mat = c_mat / c_mat.sum(axis=1)[:, np.newaxis]
        #Todo is this the right place for this?
        self.s_count = self.c_mat.sum(axis=0)
        self.s_count[states[0]] += 1

        self.node_marginal  = self.s_count/ self.s_count.sum()

    def __set_k(self, states):

        if self.K is not None:
            return self.K
        else:
            return np.unique(states).shape[0]