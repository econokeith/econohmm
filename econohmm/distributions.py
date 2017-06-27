from __future__ import division
import numpy.random as npr
import scipy.stats as sps
import numpy.linalg as la
import numpy as np
import sys, os.path
import pandas as pd
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cystats
from mixins import HistorizeMixin

from numba import jit, vectorize, float64, guvectorize
import math

# Todo: this needs to be moved


#todo make abc. move methods
#todo distribution needs a method for plotting
class Distribution(HistorizeMixin):

    def __mul__(self, other):
        new_norm = copy.deepcopy(self)
        new_norm.add_dist(other)
        return new_norm

    def __imul__(self, other):
        return self * other

#Todo: work through reference prior and what happens when there_col's no data
class NormalInvChi2(Distribution):


    D = 1
    _hist_fields = ['mu', 'sigma']
    """
    Normal Inverse Chi-Square Fully Conjugate Prior

    4 tiers of parameters for use in different cases

    1. parameters. these can be sampled or set from posterior hyperparams
    2. prior hyper params
    3. posterior hyper params (also referred to as sufficient statistics.
    4. hyper prior (hyper-hyperparameters) used mainly for resampling/recycling in hierachical models

    def __init__(self, mu=None, sigma=None, m0=0, s0=1, k0=1, nu0=1, V0=None, hypers=None, hyperprior=None):

    """

    def __init__(self, mu=None, sigma=None, m0=0, s0=1, k0=1, nu0=1, V0=None, prior=None, hyperprior=None):

        ## PARAMETERS
        # mean of distribution
        self.mu = mu if mu is not None else m0
        # variance of distribution
        self.sigma = sigma if sigma is not None else s0
        # precision
        # self.precision = 1/sigma

        ## PRIOR HYPERPARAMETERS
        ## if hypers is str, then uses reference hypers params

        if isinstance(prior, str):
            self.m0 = 0  # mean of mu
            self.k0 = 0  # implied number of obs in hypers mean
            self.s0 = 0 # s^2
            self.nu0 = -1 #

        elif isinstance(prior, self.__class__):

            self.m0 = prior.m0
            self.s0 = prior.s0
            self.k0 = prior.k0
            self.nu0 = prior.nu0

        else:
            self.m0 = m0  # mean of mu
            self.k0 = k0  # implied number of obs in hypers mean
            self.s0 = s0 # s^2
            self.nu0 = nu0 # degrees of freedom


        # this just makes it easier to control mean variance of mean and data separately
        if V0 is not None:
            self.k0 = self.s0 / V0

        ## POSTERIOR HYPERPARAMETERS
        self.mN = m0  #
        self.kN = k0  # updated hyper
        self.sN = s0
        self.nuN = nu0  # updated degrees of freedom

        ## CACHED SUFFICIENT STATISTICS
        self.x_bar = self.mu  # sample mean
        self.Nk = 0  # number of observations in data
        self.sum_x2 = None # data sum of squared
        self.x_scat = s0 * nu0  # sum(x^2 - N x_bar^2) = sum((x-x_bar)^2)

        ## HYPER-HYPER-PARAMETERS

        if isinstance(hyperprior, self.__class__):
            self.m00 = hyperprior.m0
            self.s00 = hyperprior.s0
            self.k00 = hyperprior.k0
            self.nu00 = hyperprior.nu0

        ## SAMPLING BOUNDS
        self.mu_bounds = (-np.inf, np.inf)
        self.sigma_bounds = (0, np.inf)

    def __repr__(self):
            return self.__class__.__name__ + '(mu=%f,sigma=%f)' % (self.mu,self.sigma)

    @property
    def hyper_param(self):
        out_dict = {}
        out_dict['m0'] = self.m0
        out_dict['s0'] = self.s0
        out_dict['k0'] = self.k0
        out_dict['nu0'] = self.nu0
        return out_dict

    def copy(self):
        return copy.deepcopy(self)

    def set_hyper(self, mu0=None, s0=None, k0=None, nu0=None):
        if mu0 is not None:
            self.mu0 = mu0
        if k0 is not None:
            self.k0 = k0
        if s0 is not None:
            self.s0 = s0
        if nu0 is not None:
            self.nu0 = nu0


    def set_params(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def reset(self, resample=False):

            if hasattr(self, 'm00') and resample:

                self.mN = self.m00
                self.sN = self.s00
                self.kN = self.k00
                self.nuN = self.nu00
                self.sample_posterior()

                self.s0 = self.sigma
                self.m0 = self.mu

            self.__init__(m0=self.m0, s0=self.s0, k0=self.k0, nu0=self.nu0)

    def set_new(self, m0=None, s0=None, k0=None, nu0=None):
            if m0 is None:
                m0 = self.m0
            if s0 is None:
                s0 = self.s0
            if k0 is None:
                k0 = self.k0
            if nu0 is None:
                nu0 = self.n0

            self.__init__(m0=m0, s0=s0, k0=k0, nu0=nu0)

    def log_partition(self):
        pass

    def rvs(self, size=None):
        if size is None:
            norm = npr.randn()
        else:
            norm = npr.randn(size)
        return norm * np.sqrt(self.sigma) + self.mu

    def add_dist(self, other):
        # not sure if this should be multiply.
        """
        :param other: Must be another NormalInvChi2 distribution
        :return:
        """
        assert isinstance(other, NormalInvChi2)
        m0, m1 = self.mu, other.mu
        s0, s1 = self.sigma, other.sigma
        k0, k1 = self.kN, other.kN
        nu0, nu1 = self.nuN, other.nuN

        self.kN = kN = k0 + k1
        self.nuN = nuN = nu0 + nu1
        self.mN = self.mu = mN = (k0 * m0 + k1 * m1)/ kN
        ## I still feel like this should have a 6 in it somewhere
        self.sN = self.sigma = (nu0 * s0 + nu1 * s1 + k0*(m0-mN)**2 + k1*(m1-mN)**2)/nuN

    def evidence(self, data):
        return norm_pdf_vec(data, self.mu, self.sigma)

    def update(self, data, weights=None, setparam=True, i=None, sample=False):
        pass

    def downdate(self, data, weights=None, setparam=True, i=None, sample=False):
        pass

    def estimate(self, data, weights=None, setparam=True, i=None, sample=False):
        """
        MAP estimate of parameters jointly
        :param data:
        :param weights: if weights is None, it's assumed to be an array of 1s
        :return:
        """
        if data.shape[0] == 0:
            self.reset()
            return

        m0 = self.m0
        k0 = self.k0
        nu0 = self.nu0
        s0 = self.s0

        if data.shape[0]!=0:
            weight, Nk = self.__weightNk(data, weights)

            self.kN = kN = k0 + Nk  # updated kappa
            self.nuN = nuN = nu0 + Nk  # updated nu
            self.x_bar = x_bar = weight.dot(data) / Nk  # weighted mean
            self.sum_x2 = sum_x2 = weight.dot(data**2)  # weighted sum of square

            self.mN = (k0 * m0 + Nk * x_bar) / kN  # posterior mean
            # mean_adj adjusts difference between sample mean and prior mean
            mean_adj = (k0 * Nk / kN ) * (x_bar - m0)**2
            x_scat = sum_x2 - Nk * x_bar**2

            self.sN = ( s0 * nu0 + x_scat + mean_adj) / nuN  # posterior variance
            self.mean_adj = mean_adj
            self.x_scat = x_scat

        else:
            self.reset()

        if setparam is True:
            if sample is False:
                self.sigma = self.sN
                self.mu = self.mN
            elif sample == 'joint':
                self.sample_posterior()
            else:
                self.sample_post_cond_mean()
                self.sample_post_cond_var()

            if isinstance(i, int):
                self.hist['mu'][i] = self.mu
                self.hist['sigma'][i] = self.sigma

    #todo : why does this give weird output when gibbsing
    def sample_posterior(self, data=None, weights=None, size=1, new=False):

        if data is not None:
            self.estimate(data, weights)

        sigma = 1./npr.gamma(self.nuN/2, scale=2/(self.nuN * self.sN))
        mu = npr.randn() * np.sqrt(sigma / self.kN) + self.mN

        if new is True:
            return NormalInvChi2(m0=mu, s0=sigma, k0=self.kN, nu0=self.nuN)
        else:
            self.sigma = sigma
            self.mu = mu


    def sample_post_cond_mean(self, data=None, weights=None):

        m0 = self.m0
        try:
            V0 = self.s0 / self.k0
        except:
            V0 = np.inf

        sig = self.sigma

        weight, Nk = self.__weightNk(data, weights)

        try:
            VN = 1 / (1 / V0 + Nk / sig)
        except:
            return V0

        if data is not None:
            x_bar = weight.dot(data) / Nk
        else:
            x_bar = self.x_bar

        # calculate condition mean
        mk = VN * (Nk * x_bar / sig + m0 / V0)
        # sample mean
        self.mu = npr.randn() * np.sqrt(VN) + mk
        self.mk = mk

    def sample_post_cond_var(self, data=None, weights=None):
        weight, Nk = self.__weightNk(data, weights)
        mu = self.mu
        nu0 = self.nu0
        s0 = self.s0
        # if new data (i.e for gibbs, then will update sufficient stats
        # otherwise will use existing ones.
        if data is None:
            x_bar = self.x_bar
            sum_sq = self.x_scat + Nk*(x_bar - mu)**2
        else:
            sum_sq = weight.dot((data-mu)**2)

        sN = s0 * nu0 + sum_sq
        nuN = nu0 + Nk

        self.sK = sN
        self.sigma = 1./npr.gamma(nuN/2, scale=2/sN)

    #todo this is needs to be part of a parents class
    def __weightNk(self, data, weights):
        if data is None:
            Nk = self.Nk
            weight = None
        elif weights is None:
            self.Nk = Nk = data.shape[0]
            weight = np.ones(Nk)
        else:
            self.Nk = Nk = weights.sum()
            weight = weights
        return weight, Nk

class NormalInvWish(Distribution):

    _hist_fields = ['mu', 'sigma']
    """
    Normal Inverse Chi-Square Fully Conjugate Prior
    """

    def __init__(self, mu=None, sigma=None, m0=0, s0=1, k0=1, nu0=1, prior=None):

        ## PARAMETERS
        # mean of distribution
        self.mu = mu if mu is not None else m0
        # variance of distribution
        self.sigma = sigma if sigma is not None else s0
        # precision
        #self.precision = 1/sigma

        ## PRIOR HYPERPARAMETERS
        ## if prior is 0, then uses reference prior params
        if isinstance(prior, str):
            self.m0 = 0  # mean of mu
            self.k0 = 0  # implied number of obs in prior mean
            self.s0 = 0 # s^2
            self.nu0 = 0 #
        else:
            self.m0 = m0  # mean of mu
            self.k0 = k0  # implied number of obs in prior mean
            self.s0 = s0 # s^2
            self.nu0 = nu0 # degrees of freedom

        ## POSTERIOR HYPERPARAMETERS
        self.mN = m0  #
        self.kN = k0  # updated hyper
        self.sN = s0
        self.nuN = nu0  # updated degrees of freedom

        ## CACHED SUFFICIENT STATISTICS
        self.x_bar = self.mu  # sample mean
        self.Nk = nu0  # number of observations in data
        self.sum_x2 = None # data sum of squared
        self.x_scat = s0 * nu0  # sum(x^2 - N x_bar^2) = sum((x-x_bar)^2)

        ## CACHED OTHER STATS

    def __repr__(self):
            return self.__class__.__name__ + '(mu=%f,sigma=%f)' % (self.mu,self.sigma)

    @property
    def hyper_param(self):
        out_dict = {}
        out_dict['m0'] = self.m0
        out_dict['s0'] = self.s0
        out_dict['k0'] = self.k0
        out_dict['nu0'] = self.nu0
        return out_dict

    def copy(self):
        return copy.deepcopy(self)

    def set_hyper(self, mu0=None, s0=None, k0=None, nu0=None):
        if mu0 is not None:
            self.mu0 = mu0
        if k0 is not None:
            self.k0 = k0
        if s0 is not None:
            self.s0 = s0
        if nu0 is not None:
            self.nu0 = nu0

    def log_partition(self):
        pass

    def reset(self):
            self.__init__(mu0=self.mu0, s0=self.s0, k0=self.k0, nu0=self.nu0)

    def add_dist(self, other):
        # not sure if this should be multiply.
        """
        :param other: Must be another NormalInvChi2 distribution
        :return:
        """
        assert isinstance(other, NormalInvChi2)
        m0, m1 = self.mu, other.mu
        s0, s1 = self.sigma, other.sigma
        k0, k1 = self.kN, other.kN
        nu0, nu1 = self.nuN, other.nuN

        self.kN = kN = k0 + k1
        self.nuN = nuN = nu0 + nu1
        self.mN = self.mu = mN = (k0 * m0 + k1 * m1)/ kN
        ## I still feel like this should have a 6 in it somewhere
        self.sN = self.sigma = (nu0 * s0 + nu1 * s1 + k0*(m0-mN)**2 + k1*(m1-mN)**2)/nuN

    def evidence(self, data):
        return norm_pdf_vec(data, self.mu, self.sigma)

    def estimate(self, data, weights=None, setparam=True, i=None, sample=False):
        """
        MAP estimate of parameters jointly
        :param data:
        :param weights: if weights is None, it's assumed to be an array of 1s
        :return:
        """
        m0 = self.m0
        k0 = self.k0
        nu0 = self.nu0
        s0 = self.s0

        weight, Nk = self.__weightNk(data, weights)

        self.kN = kN = k0 + Nk  # updated kappa
        self.nuN = nuN = nu0 + Nk  # updated nu
        self.x_bar = x_bar = weight.dot(data) / Nk  # weighted mean
        self.sum_x2 = sum_x2 = weight.dot(data**2)  # weighted sum of square

        self.mN = (k0 * m0 + Nk * x_bar) / kN  # posterior mean
        # mean_adj adjusts difference between sample mean and prior mean
        mean_adj = (k0 * Nk / kN ) * (x_bar - m0)**2
        x_scat = sum_x2 - Nk * x_bar**2

        self.sN = ( s0 * nu0 + x_scat + mean_adj) / nuN  # posterior variance
        self.mean_adj = mean_adj
        self.x_scat = x_scat

        if setparam is True:
            if sample is False:
                self.sigma = self.sN
                self.mu = self.mN
            else:
                self.sample_posterior()

            if isinstance(i, int):
                self.hist['mu'][i] = self.mu
                self.hist['sigma'][i] = self.sigma

    def sample_posterior(self, data=None, weights=None, size=1):

        if data is not None:
            self.estimate(data, weights)

        sigma = 1./npr.gamma(self.nuN/2, scale=2/(self.nuN * self.sN))
        mu = npr.randn() * np.sqrt(sigma / self.kN) + self.mN

        self.sigma = sigma
        self.mu = mu

    def sample_post_cond_mean(self, data=None, weights=None):

        m0 = self.m0
        V0 = self.s0 / self.k0
        sig = self.sigma

        weight, Nk = self.__weightNk(data, weights)
        VN = 1 / (1 / V0 + Nk / sig)

        if data is not None:
            x_bar = weight.dot(data) / Nk
        else:
            x_bar = self.x_bar

        # calculate condition mean
        mk = VN * (Nk * x_bar / sig + m0 / V0)
        # sample mean
        self.mu = npr.randn() * np.sqrt(VN) + mk
        self.mk = mk

    def sample_post_cond_var(self, data=None, weights=None):
        weight, Nk = self.__weightNk(data, weights)
        mu = self.mu
        nu0 = self.nu0
        s0 = self.s0
        # if new data (i.e for gibbs, then will update sufficient stats
        # otherwise will use existing ones.
        if data is None:
            x_bar = self.x_bar
            sum_sq = self.x_scat + Nk*(x_bar - mu)**2
        else:
            sum_sq = weight.dot((data-mu)**2)

        sN = s0 * nu0 + sum_sq
        nuN = nu0 + Nk

        self.sK = sN
        self.sigma = 1./npr.gamma(nuN/2, scale=2/sN)

    def __weightNk(self, data, weights):
        if data is None:
            Nk = self.Nk
            weight = None
        elif weights is None:
            self.Nk = Nk = data.shape[0]
            weight = np.ones(Nk)
        else:
            self.Nk = Nk = weights.sum()
            weight = weights
        return weight, Nk


class Gauss1d(object):

    def __init__(self, mu=None, sigma=None):

        self.mu = mu if mu is not None else 0
        self.sigma = sigma if sigma is not None else 1

    def evidence(self, data):
        return norm_pdf_vec(data, self.mu, self.sigma)

    def estimate(self, data, weights, save=True, out=False):

        Nk = weights.sum()
        E_x = weights.dot(data)
        E_xx = weights.dot(data**2)
        mu = E_x / Nk
        sigma = (E_xx - Nk * mu ** 2) / Nk

        if save is True:
            self.mu, self.sigma = mu, sigma
        if out is True:
            return mu, sigma


@vectorize([float64(float64, float64, float64)])
def norm_pdf_vec(x, m, s):
    return 1 / (math.sqrt(s * 2 * math.pi)) * math.exp(-(x - m) ** 2 / (2 * s))