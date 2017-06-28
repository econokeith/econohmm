# econohmm
#### Framework for Bayesian Estimation of HMM's and Mixtures Models

econohmm is the foundation of a generalized package for estimating various types of
Hidden Markov Models (HMMs). HMMs can be thought of as having two parts:

1. **Switching Model**: Most often a Markov Process that describes the transitions
between states. In each state, the data generating process is different.

2. **Emissions Model**: The generalized form of the data generating process such as
Gaussian, Laplace, AR(1), linear regression, etc. In each state, the parameter values
of the DGP are different.

The purpose of econohmm is provide a set of tools to work with different emission
 and switching models interchangably. econohmm has basic mixins and objects for emission models and switching models that are combined
   and used in with standard HMM algorithmic framework.

#### Example Notebooks

econohmm is still in the early stages of development. However, I've included two iPython
Notebooks demonstrating some of its potential uses.

##### Estimate Mixture Models and Dirichlet Process Mixtures Models with econohmm.ipynb


Shows how to use the container and emission classes of econohmm to estimate a 1D Gaussian mixture model. I use three methodologies:
1. Gibbs Sampling with specified number of clusters
2. Expectations Maximization with specified number of clusters
3. Dirichlet Process mixture model that determines the number of clusters given the data.


##### HMM Presentation.ipynb

Shows how to set up the basic HMM, switching, and emission objects in order to estimate the states and
parameters for a given time-series. Demonstrates how to use either the Baum-Welch or Forwards Filter
Backwards Sample algorithm for estimation. Applies the model to artificial data, to 10 Year US Treasury
bond yields, and to annual returns of US public pension funds since World War II.

#### The Future

I'm currently in the process of finalizing more complicated emissions distributions such as Linear Regression and
AR(p) models, and N-dimensional Gaussians as well as expanding the switching models to include time-varying, hierarchical, and stick-breaking type transition matrices.

