from __future__ import division
import numpy.random as npr
import scipy.stats as sps
import numpy.linalg as la
import numpy as np
import sys, os.path
import pandas as pd
import copy
import numpy

from markov_models import Markov
from emission_models import GaussEmission1d, Gauss1d, NormalEmission1d, NormalInvChi2


from cystats import cystats
# from clustering import gmm_em
from utils import normalize

# Todo: need to cython the forwards / back stuff

class HMM(object):
    """

    """
    def __init__(self,  markov_model, emission_model, data=None, N=100, states_0=None, t_mat= None):

        assert markov_model.K == emission_model.K

        self.K = markov_model.K # number of states
        self.N = N # length of data

        self._data = None # hidden data vector

        self.data = data # the series of data
        self.mm = markov_model # current estimate or sample of transition matrix
        self.em = emission_model # emission model

        self.em0 = copy.deepcopy(emission_model) # copy to compare result of estimation
        self.states_0 = states_0  # true states if known

        self.states = None # state sequence
        self.path = None # sample path of model

        self.evidence = None # local evidence matrix  p(x_t | z_t = j)
        self.alpha = None #  p(z_t = j | x_(1:t)) = 1/Z_t * p(x_t | z_t = j) * p(z_t = j | x_(1:t))
        self.beta = None
        self.gamma = None
        self.zs = None

        self.edge_marginal = None
        self.node_marginal = None

        self._z_viterbi = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, datas):
        self._data = datas
        if datas is not None:
            self.N = datas.shape[0]

    @property
    def t_mat(self):
        return self.mm.t_mat

    @t_mat.setter
    def t_mat(self, mat):
        self.mm.t_mat = mat

    # Todo : Figure out how to make this a general function

    def param_path(self, param):
        fields = self.em.get_field_array(param)

        if self.states is not None:
            return fields[self.states]
        else:
            try:
                self.states = self.g_states
                return fields[self.states]
            except:
                pass

    def param_paths(self, param, which_states='a'):
        fields = self.em.get_field_array(param)
        assert which_states in ['a', 'b', 'g', 'v']
        states = eval('self.{}_states'.format(which_states))
        return fields[states]

    @property
    def a_states(self):
        if self.alpha is not None:
            return self.alpha.argmax(axis=1)

    @property
    def b_states(self):
        if self.beta is not None:
            return self.beta.argmax(axis=1)

    @property
    def g_states(self):
        if self.gamma is not None:
            return self.gamma.argmax(axis=1)

    @property
    def v_states(self):
        if self._z_viterbi is None:
            self.viterbi()
        return self._z_viterbi

    def sample_states(self, N=None, save=True, out=True, update=True):
        NN = self.__choose_N(N)
        states = self.mm.sample_states(NN, update=update)

        if save is True:
            self.states = states

        if out is True:
            return states

    def sample_path(self, N=None, new_s=True, save=False, out=True):
        NN = self.__choose_N(N)

        if new_s is True:
            states = self.mm.sample_states(NN)
        else:
            states = self.states

        path = self.em.sample_path(states, NN)

        if save is True:
            self.path = path
            self.states = states

        if out is True:
            return path

    # Todo : probably going to rename log_prob
    def _update_evidence(self):
        evidence = np.empty((self.K, self.N))
        for i, e in enumerate(self.em):
            evidence[i] = e.evidence(self.data)
        self.evidence = evidence.T

    def fwd_filter(self, pis=None):

        if self.alpha is None:
            alpha = np.empty((self.N, self.K))
            zs = np.empty(self.N)

        else:
            alpha = self.alpha
            zs = self.zs

        self._update_evidence()

        ev = self.evidence
        tmt = self.mm.t_mat.T

        if pis is None:
            pi = np.ones(self.K) / self.K
        else:
            pi = pis

        alpha[0], zs[0] = normalize(pi * ev[0])

        for i in xrange(1, self.N):
            alpha[i], zs[i] = normalize(ev[i] * tmt.dot(alpha[i-1]))

        self.alpha = alpha
        self.zs = zs

    def back_pass(self):
        if self.beta is None:
            self.beta = np.ones((self.N, self.K))

        beta = self.beta
        ev = self.evidence[::-1]
        tm = self.t_mat

        for i in xrange(1, self.N):
            beta[i], _ = normalize(tm.dot(ev[i] * beta[i-1]))

        self.beta = beta[::-1]

    def fwd_back(self, pis=None):
        self.fwd_filter(pis=pis)
        self.back_pass()
        self.gamma = self.alpha * self.beta
        self.gamma /= self.gamma.sum(axis=1)[:, np.newaxis]

    def bm_fit(self, iter_num=100, hist=False, pis=None):

        xi = np.empty((self.N-1, self.K, self.K))

        for i in xrange(iter_num):
            #run forward / backwards algo
            self.fwd_back(pis = pis)

            t_mat = self.t_mat
            alpha = self.alpha
            beta = self.beta
            ev = self.evidence
            #calculuate smoothed edge marginals for each state
            for jj in xrange(self.N-1):
                xi_t = t_mat * np.outer(alpha[jj] , ev[jj+1] * beta[jj+1])
                xi[jj] = xi_t / xi_t.sum()
            #find expected number of switches into state
            E_jk = xi.sum(axis=0)
            self.E_jk = E_jk
            #normalize rows to make new t_mat
            #Todo: double check this calculation
            self.t_mat = E_jk / E_jk.sum(axis=1)[:,np.newaxis]
            #update parameter estimates
            self.em.estimate(self.data, self.gamma.T, i)

            #update pis
            pis = self.gamma[0]

    def back_sample(self):
        """
        p(z_t=i | z_t+1, x_t+1) = phi_t+1( j) x trans(i, j) x alpha_t(i) / alpha_t+1(j)
        :return:
        """
        assert isinstance(self.alpha, numpy.ndarray)

        alpha = self.alpha
        phi = self.evidence
        t_mat = self.t_mat

        dist_T = alpha[-1][np.newaxis,:]
        jT = cystats.sample_states(dist_T)[0]

        ll = alpha.shape[0] - 1
        j = jT

        sampled_states = np.empty(ll+1)
        sampled_states[-1] = j

        for t in xrange(1, ll+1):
            tt = ll - t
            s_t_dist = phi[tt+1,j] * t_mat[:,j] * alpha[tt] / alpha[tt+1, j]
            s_t_dist /= s_t_dist.sum()
            j = cystats.sample_states(s_t_dist[np.newaxis,:])[0]
            sampled_states[tt] = j

        self.states = sampled_states

    # todo need to make it sample from posteriors here
    def fit_from_states(self, hist=None, sample=False):
        assert isinstance(self.states, numpy.ndarray)

        for i, cc in enumerate(self.em):
            ss = np.where(self.states==i)[0]
            cc.estimate(self.data[ss], i=hist, sample=sample)

        self.mm.update_from_states(self.states)

    ## Todo: fit from states needs to change. need to make it a sample thing
    def gibbs(self, iter_num=100, extend=False, fields=None, sample=False):

        self.em.comp_hist_init(N=iter_num, extend=extend, fields=fields)
        for i in xrange(1, iter_num):
            self.fwd_filter()
            self.back_sample()
            self.fit_from_states(hist=i, sample=sample)

    def viterbi(self):

        if self.evidence is None:
            return

        N = self.N
        K = self.K

        log_t_mat = np.log(self.t_mat)
        log_evidence = np.log(self.evidence)

        # d_list is the delta from the viterbi algo.
        # a_list is most likely previous state for each current state

        d_list = np.empty((N,K))
        a_list = np.empty((N,K)).astype(int)
        d_list[0] = log_evidence[0] + np.log(np.ones(K) / K)
        z_viterbi = np.empty(N).astype(int)

        for ii in xrange(1, N):
            viterbi_mat = d_list[ii-1][:,np.newaxis] + log_t_mat + log_evidence[ii]
            # find most likely previous state given current state
            d_list[ii] = viterbi_mat.max(axis=0)
            a_list[ii] = viterbi_mat.argmax(axis=0)

        z_viterbi[-1] =  d_list[-1].argmax()

        # z_t =  a_t+1(z_t+1)
        for ii in xrange(N-1):
            z_viterbi[N-2-ii] = a_list[N-1-ii][z_viterbi[N-1-ii]]

        self._z_viterbi = z_viterbi


    def __choose_N(self, N):
        if self.N is None and N is None:
            NN = 100
        elif N is None:
            NN = self.N
        else:
            NN = N
        return NN


class IHMM(object):

    """

    """
    def __init__(self,  markov_model, emission_model, data=None, N=100, states_0=None, t_mat= None):

        #assert markov_model.K == emission_model.K

        self.K = markov_model.K # number of states
        self.N = N # length of data

        self._data = None # hidden data vector

        self.data = data # the series of data
        self.mm = markov_model # current estimate or sample of transition matrix
        self.em = emission_model # emission model

        self.em0 = copy.deepcopy(emission_model) # copy to compare result of estimation
        self.states_0 = states_0  # true states if known

        self.states = None # state sequence
        self.path = None # sample path of model

        self.evidence = None # local evidence matrix  p(x_t | z_t = j)
        self.alpha = None #  p(z_t = j | x_(1:t)) = 1/Z_t * p(x_t | z_t = j) * p(z_t = j | x_(1:t))
        self.beta = None
        self.gamma = None
        self.zs = None

        self.edge_marginal = None
        self.node_marginal = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, datas):
        self._data = datas
        if datas is not None:
            self.N = datas.shape[0]

    @property
    def t_mat(self):
        return self.mm.t_mat

    @t_mat.setter
    def t_mat(self, mat):
        self.mm.t_mat = mat

    # Todo : Figure out how to make this a general function
    @property
    def param_path(self, param):
        fields = self.em.get_field_array(param)

        if self.states is not None:
            return fields[self.states]
        else:
            try:
                self.states = self.g_states
                return fields[self.states]
            except:
                pass

    @property
    def a_states(self):
        if self.alpha is not None:
            return self.alpha.argmax(axis=1)

    @property
    def b_states(self):
        if self.beta is not None:
            return self.beta.argmax(axis=1)

    @property
    def g_states(self):
        if self.gamma is not None:
            return self.gamma.argmax(axis=1)
    ## dif
    def sample_states(self, N=None, save=True, out=True, update=True):
        NN = self.__choose_N(N)
        states = self.mm.sample_states(NN, update=update)

        if save is True:
            self.states = states

        if out is True:
            return states

    ## need to chnage
    def sample_path(self, N=None, new_s=True, save=False, out=True):
        NN = self.__choose_N(N)

        if new_s is True:
            states = self.mm.sample_states(NN)
        else:
            states = self.states

        path = self.em.sample_path(states, NN)

        if save is True:
            self.path = path
            self.states = states

        if out is True:
            return path

    # Todo : probably going to rename log_prob
    def _update_evidence(self):
        evidence = np.empty((self.K, self.N))
        for i in xrange(self.K):
            evidence[i] = self.em[i].evidence(self.data)
        self.evidence = evidence.T

    def fwd_filter(self, pis=None):

        if self.alpha is None:
            alpha = np.empty((self.N, self.K))
            zs = np.empty(self.N)

        self._update_evidence()

        ev = self.evidence
        tmt = self.mm.t_mat.T

        if pis is None:
            pi = np.ones(self.K) / self.K
        else:
            pi = pis

        alpha[0], zs[0] = normalize(pi * ev[0])

        for i in xrange(1, self.N):
            alpha[i], zs[i] = normalize(ev[i] * tmt.dot(alpha[i-1]))

        self.alpha = alpha
        self.zs = zs

    def back_pass(self):
        if self.beta is None:
            self.beta = np.ones((self.N, self.K))

        beta = self.beta
        ev = self.evidence[::-1]
        tm = self.t_mat

        for i in xrange(1, self.N):
            beta[i], _ = normalize(tm.dot(ev[i] * beta[i-1]))

        self.beta = beta[::-1]

    def fwd_back(self, pis=None):
        self.fwd_filter(pis=pis)
        self.back_pass()
        self.gamma = self.alpha * self.beta
        self.gamma /= self.gamma.sum(axis=1)[:, np.newaxis]

    def bm_fit(self, iter_num=100, hist=False, pis=None):

        xi = np.empty((self.N-1, self.K, self.K))

        for i in xrange(iter_num):
            #run forward / backwards algo
            self.fwd_back(pis = pis)

            t_mat = self.t_mat
            alpha = self.alpha
            beta = self.beta
            ev = self.evidence
            #calculuate smoothed edge marginals for each state
            for jj in xrange(self.N-1):
                xi_t = t_mat * np.outer(alpha[jj] , ev[jj+1] * beta[jj+1])
                xi[jj] = xi_t / xi_t.sum()
            #find expected number of switches into state
            E_jk = xi.sum(axis=0)
            self.E_jk = E_jk
            #normalize rows to make new t_mat
            #Todo: double check this calculation
            self.t_mat = E_jk / E_jk.sum(axis=1)[:,np.newaxis]
            #update parameter estimates
            self.em.estimate(self.data, self.gamma.T, i)

            #update pis
            pis = self.gamma[0]

    def back_sample(self, us):
        """
        p(z_t=i | z_t+1, x_t+1) = phi_t+1( j) x trans(i, j) x alpha_t(i) / alpha_t+1(j)
        :return:
        """
        assert isinstance(self.alpha, numpy.ndarray)

        alpha = self.alpha
        phi = self.evidence
        t_mat = self.t_mat

        dist_T = alpha[-1][np.newaxis,:]
        jT = cystats.sample_states(dist_T)[0]

        ll = alpha.shape[0] - 1
        j = jT

        sampled_states = np.empty(ll+1)
        sampled_states[-1] = j

        for t in xrange(1, ll+1):
            tt = ll - t
            s_t_dist = phi[tt+1,j] * t_mat[:,j] * alpha[tt] / alpha[tt+1, j]

            # do the u trick
            t_col = copy.copy(self.t_mat[:,j])
            t_col[t_col < us[t]] = 0
            s_t_dist *= t_col
            s_t_dist /= s_t_dist.sum()

            j = cystats.sample_states(s_t_dist[np.newaxis,:])[0]
            sampled_states[tt] = j

        self.states = sampled_states

    # todo need to make it sample from posteriors here
    def fit_from_states(self, hist=None, sample=False):
        assert isinstance(self.states, numpy.ndarray)

        for i, cc in enumerate(self.em):
            ss = np.where(self.states==i)[0]
            cc.estimate(self.data[ss], i=hist, sample=sample)

        self.mm.update_from_states(self.states)


    def gibbs(self, iter_num=100, extend=False, fields=None, sample=False):

        self.em.comp_hist_init(N=iter_num, extend=extend, fields=fields)
        for i in xrange(1, iter_num):
            self.fwd_filter()
            self.back_sample()
            self.fit_from_states(hist=i, sample=sample)



    def __choose_N(self, N):
        if self.N is None and N is None:
            NN = 100
        elif N is None:
            NN = self.N
        else:
            NN = N
        return NN


