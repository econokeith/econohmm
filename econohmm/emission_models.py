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
# Todo: this needs to be moved

from mixins import ContainerMixin
from distributions import NormalInvChi2, Gauss1d





class EmissionModel(ContainerMixin):


    _hist_fields = ['K']

    """
    def __init__(self, comp, params=None, prior=None, rand=False, hyperprior=None):

    params are the model parameters to be set if they are different from the prior
    prior is an example of the prior distribution.

    individual components can be instantiated in 4 ways:

    1. all identical
    2. identical prior but different parameters
    3. in hierarchical model where they are sampled from a hyper_prior
    4. in hierarchical model where the hyper_params are set instead of the params via the param input

    """
    #todo: set K as a property
    def __init__(self, comp, params=None, prior=None, rand=False, hyperprior=None, K=5, paramstohyper=False):

        # allow for prior of several forms
        if prior is None:
            prior = comp()

        elif isinstance(prior, str):
            prior = comp(prior='ref')

        elif isinstance(prior, comp):
            prior = prior

        elif isinstance(prior, dict):
            prior_dict = prior
            prior = comp(**prior_dict)

        else:
            pass

        self.prior = prior
        self.prior_dict = prior_dict = prior.hyper_param
        self.hyperprior = hyperprior

        # initial length can either be set by length params or K
        if params is not None:
            K = params.shape[0]



        # make component list
        self.components = [comp(hyperprior=hyperprior, **prior_dict) for _ in xrange(K)]

        # update individual components

        # sets as random hierarchical components
        if hyperprior is not None and rand is True:
            for c in self.components:
                c.reset(resample=True)

        #will have to check this later for new stuff.
        elif params is not None and paramstohyper is True:
            for i, c in enumerate(self.components):
                c.set_new(*list(params[i]))


        elif params is not None:
            for i, c in enumerate(self.components):
                c.set_params(*list(params[i]))

        else:
            pass


        #
        #     self.components = [comp(**prior_dict) for _ in xrange(self.K)]
        #
        # else:
        #     self.components = []
        #     for param in params:
        #         self.components.append(comp(*param, **prior_dict))
        #
        # if isinstance(hyperprior, comp):
        #
        #
        # elif rand is True:
        #     self.components = []
        #     for _ in xrange(self.K):
        #         self.prior.sample_posterior()
        #         self.components.append(comp())

    @property
    def K(self):
        return len(self.components)

    def sample_path(self, states, N):
        pass

    def estimate(self, data, weights, i=None, sample=False):
        for j, c in enumerate(self.components):
            c.estimate(data, weights[j], i=i, sample=sample)

    def add_dist(self, dist):
        assert isinstance(dist, self._comp_type)
        self.components.append(dist)
        # self.K +=1


#Todo give this a reset.
class NormalEmission(EmissionModel):

    _comp_type = NormalInvChi2
    _hist_fields = ['K']

    def __init__(self, params=None, prior=None, rand=False, hyperprior=None, K=5, paramstohyper=False):
        super(NormalEmission, self).__init__(self._comp_type, params=params, K=K, prior=prior, rand=rand,
                                             hyperprior=hyperprior, paramstohyper=paramstohyper)

    #todo do something with this N
    def sample_path(self, states, N):
        out = npr.randn(N)
        sig = self.get_field_array('sigma') ** .5
        mu = self.get_field_array('mu')
        return sig[states] * out + mu[states]




class GaussEmission1d(object):

    comp_type = Gauss1d
    """
    e_params will build the 1d Gaussians, but must be in the form of [[mu1, sig1],
    [mu2, sig2],...,[]]
    """


    def __init__(self, e_params=None, K=None, e_prior=None):
        if e_params is not None and K is not None:
            assert K == len(e_params)

        self.components = []
        self.mu = []
        self.sigma = []

        if e_params is None:
            self.K = K
            if K is not None:
                self.components =[Gauss1d() for _ in xrange(K)]
                self.mu = [0] * K
                self.sigma = [1] * K

        else:
            self.K = len(e_params)
            for mu, sig in e_params:
                self.components.append(Gauss1d(mu, sig))
                self.mu.append(mu)
                self.sigma.append(sig)

        self.components = self.components
        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def __getitem__(self, key):
        return self.components[key]

    def sample_path(self, states, N):
        out = npr.randn(N)
        sig = self.sigma ** .5
        return sig[states] * out + self.mu[states]

    def estimate(self, data, weights,i=None):
        for j, c in enumerate(self.components):
            c.estimate(data, weights[j])
            self.mu[j] = c.mu
            self.sigma[j] = c.sigma




# Todo need to clean up this one
class NormalEmission1d(object):


    """
    e_params will build the 1d Gaussians, but must be in the form of [[mu1, sig1],
    [mu2, sig2],...,[]]
    """
    comp_type = NormalInvChi2

    def __init__(self, e_params=None, K=None, e_prior=None):
        if e_params is not None and K is not None:
            assert K == len(e_params)

        self.components = []
        self.mu = []
        self.sigma = []

        if e_params is None:
            self.K = K
            if K is not None:
                self.components =[self.comp_type() for _ in xrange(K)]
                self.mu = [0] * K
                self.sigma = [1] * K

        else:
            self.K = len(e_params)
            for mu, sig in e_params:
                self.components.append(self.comp_type(mu, sig))
                self.mu.append(mu)
                self.sigma.append(sig)

        self.components = self.components
        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def __getitem__(self, key):
        return self.components[key]

    def __getattr__(self, item):
        for c in self.components:
            c.item

    def sample_path(self, states, N):
        out = npr.randn(N)
        sig = self.sigma ** .5
        return sig[states] * out + self.mu[states]

    def estimate(self, data, weights, i=None):
        for j, c in enumerate(self.components):
            c.estimate(data, weights[j])
            self.mu[j] = c.mu
            self.sigma[j] = c.sigma


# class EmissionModel(ContainerMixin):
#
#     _hist_fields = ['K']
#     #todo: set K as a property
#     def __init__(self, comp, params=None, K=5, prior=None, rand=False, hyperprior=None):
#
#         self.components=[]
#         self.hyperprior = hyperprior
#
#         if params is not None and K is not None:
#             assert K == len(params)
#             self.K = K
#         elif K is None:
#             self.K = len(params)
#         else:
#             self.K = K
#
#         if prior is None:
#             prior = comp()
#
#         elif isinstance(prior, str):
#             prior = comp(prior='ref')
#
#         elif isinstance(prior, comp):
#             prior = prior
#
#         else:
#             prior_dict = prior
#             prior = comp(**prior_dict)
#
#         self.prior = prior
#         self.prior_dict = prior_dict = prior.hyper_param
#
#         if params is None:
#             self.components = [comp(**prior_dict) for _ in xrange(self.K)]
#
#         else:
#             self.components = []
#             for param in params:
#                 self.components.append(comp(*param, **prior_dict))
#
#         if isinstance(hyperprior, comp):
#
#
#
#         elif rand is True:
#             self.components = []
#             for _ in xrange(self.K):
#                 self.prior.sample_posterior()
#                 self.components.append(comp())


