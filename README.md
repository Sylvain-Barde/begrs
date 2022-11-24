# begrs

This toolbox implements the Bayesian Estimation with Gaussian process Regression Surrogate (BEGRS) methodology outlined in ''Bayesian Estimation of Large-Scale Simulation Models using a Gaussian Process Regression Surrogate''

## Requirements and Installation

The toolbox uses [GPytorch](https://gpytorch.ai/), a Gaussian Process toolbox, which itself is uses [PyTorch](https://pytorch.org/). Both packages need to be installed for begrs to work. Additional (standard) packages required are: `os, sys, numpy, tqdm, warnings`.

At the moment the begrs toolbox is still in development, and does not yet have a distributable package (this is on the to-do list!). The functionality is all contained in the `begrs.py` module, and can be obtained simply by placing a copy of the file in the relevant directory.

## Usage

The `begrs` class contains all the functionality required, and is imported as follows.

```python
from begrs import begrs

# An empty estimation task is setup as follows:
begrsEst = begrs()
```

### Training a GP surrogate

Let there be $M$ empirical variables, $T$ observations, and $N$ samples from the $D$-dimensional parameter space. Training the begrs surrogate GP will require the following variables:
- `simData`, a $T \times M \times N$ numpy array of simulated data
- `samples`, a $N \times D$ numpy array of parameter samples
- `parameter_range`, a $D \times 2$ set of bounds for the parameter samples.

In addition to this, when loading the training data, one has the option to winsorize the data to remove outliers, as well as normalise each variable so it has mean 0 and standard deviation 1, in line with many machine-learning methods
- `wins=0.05`, a float giving the tail mass to be winsorized. Here, the top/bottom 2.5% would be winsorized. This is set to `None` by default, and no winsorization is carried out in this case.
- `normalise=True`, a Boolean flag indicating that the traning data should be normalised. This is true by default, but can be turned off by setting to `False`.

Several hyper-parameters need to be set for the training itself:
- `num_latents`, the number of latents used to model the $M$ empirical variables
- `num_inducing_pts`, the number of inducing points used in the variational approximation
- `batch_size`, the size of the minibatches used in training the GP via stochastic optimisation
- `numiter`, the number of epochs used by the GP optimiser
- `learning_rate`, the learning rate of the Adam optimiser

Given this, training the GP surrogate simply involves providing the training data and launching the training.

```python
# Loading the training data into the estimation object
begrsEst.setTrainingData(simData, samples, parameter_range)

# Training the estimation
begrsEst.train(num_latents, num_inducing_pts, batchSize, numiter, learning_rate)
```



### Saving and loading surrogate models

Once the training is complete, the state of the BEGRS surrogate can be save for later retrieval.

```python
# Saving to 'savePath/saveDir'
begrsEst.save(savePath + '/saveDir')
```

**Notes:**
- The saving directory is created by `begrs`, and should not already exist. This is to explicitly avoid overwriting existing saved states.
- The save function of `begrsEst` will only save attributes related to training data. See below for more details.

Given a directory containing a saved state, `loadPath`, loading a previously saved trained surrogate is simply a matter of creating an empty estimation object and loading a state into it.

```python
# Loading a saved state from 'loadPath'
begrsEst = begrs()
begrsEst.load(loadPath)
```

### Estimation

Because BEGRS only provides a surrogate likelihood, third part methods are used to actually carry out the estimation. All that `begrs` provides is the (surrogate) log-likelihood function and a basic minimal prior that can be passed to a Bayesian sampling method of choice. Both the likelihood and prior are provided with their gradients, allowing the use of Hamiltonian Monte-Carlo (HMC).

#### Loading the empirical data

Before running a BEGRS estimation, one needs to first set the empirical data. Letting `empData` be a $T \times M$ array of empirical data, this is done as follows:

```python
# Loading the empirical data into the estimation object
begrsEst.setTestingData(empData)
```

**Notes**
- If the normalisation option was picked when the training data was set, the empirical data will also be normalised in the same way when it is set.
- Always set the empirical data **after** completing the training or after loading a previosuly trained surrogate model. This is because setting the empirical data switches the surrogate model from training mode into evaluation mode.
- If a surrogate model is saved after running `setTestingData`, the attributes relating to the empirical data will **not** be saved. This is done because `setTestingData` not only loads empirical data and switches to evaluation model, but also pre-computes fixed likelihood attributes to gain time. Saving only the training attributes ensure that saved surrogate models are always loaded in a 'clean' state and can be re-used on different empirical datasets.

#### Specifying the log posterior.

Before being able to run a Bayesian estimation, the posterior needs to be defined. Furthermore, as explained in the paper, BEGRS necessitates a minimal prior that restricts the posterior to the parameter bounds. It is left up to the user to write a function for the posterior, so that they can integrate a more informative prior if needed.

The most basic case is one where the only the minimal soft prior is required. Given an arbitrary paramete vector `sample`, the log posterior is given by:

```python
def logPosterior(sample):

    prior = begrsEst.softLogPrior(sample)
    logP = begrsEst.logP(sample)

    return (prior[0] + logP[0], prior[1] + logP[1])
```

**Note:** The structure of `prior` and `logP` is a tuple, with the value in the 0 location and the gradient in the 1 entry. Any user-provided prior and posterior function should follow this structure in order to be able to use gradient-based methods such as HMC.

#### Posterior Mode

The example below shows how to use the BFGS algorithm (from `scipy.optimize.minimize`) to locate the mode of the posterior.

```python
from scipy.optimize import minimize

# Flip the signs on the posterior
negLogPosterior = lambda *args: tuple( -i for i in logPosterior(*args))

# Specify bounds and a starting point (Note the paramater vectors are centered)
numParams = begrsEst.parameter_range.shape[0]
bounds = numParams*[(-3**0.5,3**0.5)]
init = np.zeros(numParams)

# Find the mode with bounded BFGS
sampleMAP = minimize(negLogPosterior, init, method='L-BFGS-B',
              bounds = bounds, jac = True)

# Uncenter the solution to get the mode of the posterior
sampleMode = begrsEst.uncenter(sampleMAP.x)
```

#### HMC Posterior density and posterior mean

The example below shows how to use Hamiltonian Monte Carlo, specifically the No U-Turn Sampling (NUTS) algorithm provided by the `sampyl` package to take 10,000 samples from the posterior and estimate the density. Note that the NUTS algorithm is initialised at the posterior mode.

```python
import sampyl as smp
import numpy as np

# Set starting state and starting point
start = smp.state.State.fromfunc(logPosterior)
start.update({'sample': sampleMode})

# Specify a cutoff (saves time)
E_cutoff = -2*logP(vec)[0]

# Run NUTS
nuts = smp.NUTS(logPosterior, start,
                grad_logp = True,
                step_size = 0.01,
                Emax = E_cutoff)
chain = nuts.sample(10100, burn=100)

# Uncenter the samples, get the mean
posteriorSamples = np.zeros_like(chain.sample)
for i, sampleCntrd in enumerate(chain.sample):
    posteriorSamples[i,:] = begrsEst.uncenter(sampleCntrd)

sampleMean = np.mean(posteriorSamples, axis = 0)

```
