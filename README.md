# begrs

This toolbox implements the Bayesian Estimation with Gaussian process Regression Surrogate (BEGRS) methodology outlined in ''Bayesian Estimation of Large-Scale Simulation Models using a Gaussian Process Regression Surrogate''

## Requirements and Installation

A `requirements.txt` file is provided. The toolbox is built using [GPytorch](https://gpytorch.ai/), a Gaussian Process toolbox, which itself uses [PyTorch](https://pytorch.org/). Both packages need to be installed for the surrogate model in `begrs` to work. Note that the specific version requirement (1.2.0), as later versions are not (yet) compatible, this is on the to-do list. The toolbox relies on the `minimize` function of `scipy.optimize`, `sampyl-mcmc` and the Effective sample size (ESS) estimator provided by `arviz` for Bayesian estimation using the surrogate. Additional (standard) packages required are: `os, sys, numpy, time, tqdm, warnings`.

At the moment the `begrs` toolbox is still in development, and does not yet have a distributable package (this is on the to-do list!). The functionality is all contained in the `begrs.py` module, and can be obtained simply by installing the packages required and placing a copy of the file in the relevant directory.

## First stage: Gaussian process surrogate

The `begrs` class contains all the functionality required for generating a GP surrogate, and is imported as follows.

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
- `num_latents`, the number of latent GP variables used to model the $M$ empirical variables
- `num_inducing_pts`, the number of inducing points used in the variational approximation
- `batch_size`, the size of the minibatches used in training the GP via stochastic optimisation
- `numiter`, the number of epochs used by the GP optimiser
- `learning_rate`, the learning rate of the Adam optimiser

Given this, training the GP surrogate simply involves providing the training data and launching the training.

```python
# Loading the training data into the estimation object
# Outliers are winsorized and variables are normalised
begrsEst.setTrainingData(simData, samples, parameter_range, wins=0.05, normalise=True)

# Training the surrogate model on the simulated data
begrsEst.train(num_latents, num_inducing_pts, batchSize, numiter, learning_rate)
```

### Saving and loading surrogate models

Once the training is complete, the state of the BEGRS surrogate can be save for later retrieval.

```python
# Saving to 'savePath/saveDir'
begrsEst.save('savePath/saveDir')
```

**Notes:**
- The saving directory `savePath/saveDir` is created by `begrs`, and should not already exist. Attempting to write to an existing directory will raise an error. This is to explicitly avoid overwriting existing saved states.
- The save function of `begrsEst` will only save attributes related to training data. See below for more details.

Given a directory containing a saved state, `loadPath`, loading a previously saved trained surrogate is simply a matter of creating an empty estimation object and loading a state into it.

```python
# Loading a saved state from 'loadPath'
begrsEst = begrs()
begrsEst.load('loadPath')
```

## Second stage: Estimation

Because BEGRS only provides a surrogate likelihood, third party methods are used to actually carry out the estimation. All that `begrs` provides is the (surrogate) log-likelihood function and a basic minimal prior that can be passed to a Bayesian sampling method of choice. Both the likelihood and prior are provided with their gradients, allowing the use of Hamiltonian Monte-Carlo (HMC). In principle researchers can use whatever posterior sampling method they wish, however the toolbox provides a default sampler using gradient-based methods. In the following we assume that `empData` is a $T \times M$ `Numpy` array of empirical data.

### Specifying the log posterior.

Before being able to run a Bayesian estimation, the posterior needs to be defined. Furthermore, as explained in the paper, BEGRS necessitates a minimal prior that restricts the posterior to the parameter bounds. It is left up to the user to write a function for the posterior, so that they can integrate a more informative prior if needed.

The most basic case is one where the only the minimal soft prior is required. Given an arbitrary parameter vector `sample`, the log posterior is given by:

```python
def logPosterior(sample):

    prior = begrsEst.softLogPrior(sample)
    logP = begrsEst.logP(sample)

    return (prior[0] + logP[0], prior[1] + logP[1])
```

**Note:** The structure of `prior` and `logP` is a tuple, with the value in the 0 location and the gradient in the 1 entry. Any user-provided prior and posterior function should follow this structure in order to be able to use gradient-based methods such as HMC.

### Estimation using the built-in Sampler

The No U-Turn Sampling (NUTS) sampler class for the `begrs` module is imported and initialized as follows:

```python
from begrs import begrsNutsSampler
import numpy as np

# To create a sampler object, pass:
# - the BEGRS surrogate object
# - the log posterior function:
posteriorSampler = begrsNutsSampler(begrsEst, logPosterior)
```

Note that the log posterior must explicitly be provided by the researcher, even if it is constructed as above, using the built-in soft prior and surrogate likelihood. This is to allow researchers to specify a prior of their choice. Once this is done, the sampler is setup by passing the empirical data and a starting point in the **centred** parameter space for the mode-finding algorithm. Given the centring of the parameter space around 0, a vector of zeros is typically a good bet in the absence of any other information.

```python
init = np.zeros(begrsEst.num_param)
posteriorSampler.setup(empData, init)
mode = begrsEst.uncenter(posteriorSampler.mode)
```

At this point the `posteriorSampler` object will:
- Pass the empirical data to the `begrsEst` object, so that the surrogate likelihood can be calculated when `logPosterior` is called.
- Use the `minimize` function of `scipy.optimize` to find the mode of the `logPosterior` via BFGS, taking advantage of the gradient information. Note, the `mode` attribute of the `posteriorSampler` is still centred, so the `uncenter` method of the `begrsEst` object needs to be used to convert it back to the correct values.

Once the `posteriorSampler` is setup and the mode of the posterior have been found, samples can be drawn from the posterior using NUTS to take advantage of the gradient information. The example below shows how to draw 10,000 samples from the posterior and estimate the density. Note that the NUTS algorithm is initialised at the posterior mode, which is determined during the setup phase.

```python
N = 10000
burn = 100
posteriorSamples = posteriorSampler.run(N, burn)
sampleESS = posteriorSampler.minESS(posteriorSamples)
```

### Manually sampling from the posterior

As stated above, it is possible to use the surrogate likelihood provided by the `begrsEst` object in other samplers. The instructions below lay out how to manually obtain the posterior mode and sample from the posterior. The code below essentially replicates what the `begrsNutsSampler` class automatically does, but highlights the features that a researcher must be aware of when using an alternative sampler.

### Loading the empirical data

Before running a BEGRS estimation, one needs to first set the empirical data. This is done as follows:

```python
# Loading the empirical data into the estimation object
begrsEst.setTestingData(empData)
```

**Notes**
- If the normalisation option was picked when the training data was set, the empirical data will also be normalised in the same way when it is set.
- Always set the empirical data **after** completing the training or after loading a previously trained surrogate model. This is because setting the empirical data switches the surrogate model from training mode into evaluation mode.
- If a surrogate model is saved after running `setTestingData`, the attributes relating to the empirical data will **not** be saved. This is done because `setTestingData` not only loads empirical data and switches to evaluation model, but also pre-computes fixed likelihood attributes to gain time. Saving only the training attributes ensure that saved surrogate models are always loaded in a 'clean' state and can be re-used on different empirical datasets.

#### Posterior Mode

The example below shows how to use the BFGS algorithm (from `scipy.optimize.minimize`) to locate the mode of the posterior.

```python
from scipy.optimize import minimize

# Flip the signs on the posterior
negLogPosterior = lambda *args: tuple( -i for i in logPosterior(*args))

# Specify bounds and a starting point (Note the parameter vectors are centred using the standard deviation of a uniform distribution)
bounds = begrsEst.num_param*[(-3**0.5,3**0.5)]
init = np.zeros(begrsEst.num_param)

# Find the mode with bounded BFGS
sampleMAP = minimize(negLogPosterior, init, method='L-BFGS-B',
              bounds = bounds, jac = True)

# Uncenter the solution to get the mode of the posterior
sampleMode = begrsEst.uncenter(sampleMAP.x)
```

#### HMC Posterior density and posterior mean

The example below shows how to use the NUTS algorithm provided by the `sampyl-mcmc` package to take 10,000 samples from the posterior and estimate the density. Note again that the NUTS algorithm is initialised at the posterior mode.

```python
import sampyl as smp
import numpy as np

# Set starting state and starting point
start = smp.state.State.fromfunc(logPosterior)
start.update({'sample': sampleMode})

# Specify a cutoff as a multiple of the mode likelihood (saves time)
E_cutoff = -2*logP(sampleMAP.x)[0]

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

## Simulated Bayesian Computing diagnostics

The toolbox contains a Simulated Bayesian Computing (SBC) class that can perform posterior coverage diagnostics following the methodology of *Talts, Sean, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman (2018) “Validating Bayesian inference algorithms with simulation-based calibration,” arXiv preprint arXiv:1804.06788*. Please refer to that article for details of the SBC approach itself.

The SBC class for the `begrs` module is imported and initialized as follows:

```python
from begrs import begrsSbc

# Create an empty SBC object as follows:
SBC = begrsSbc()
```

The SBC analysis requires several inputs
- An instance of the `begrsNutsSampler` class to draw posterior samples from the `begrsEst` surrogate model.
- Testing samples `testSamples`, drawn from the simulation model's prior.
- Simulated data `testData` generated by the simulation model for each of these testing samples.

The latter two sets of data are formatted in the same way as the simulation data used to train the surrogate, i.e.  $T \times M \times N$ for `testData` and $N \times D$ for `testSamples`. If computationally convenient, the simulations for the training and testing datasets can be run at the same time. The `SBC` object is configured as follows:

```python
# Configure SBC object.
SBC.setTestData(testSamples,testData)
SBC.setPosteriorSampler(begrsNutsSampler(begrsEst, logP))
```

Once `SBC` object has the testing data and sampler, the analysis itself is configured by picking the number of draws `N` and the burn in period `burn`. This directly determines the number of bins in the rank histogram generated by the SBC analysis, which will `N - burn + 1`. In the example below, for example, this will result in 50 histogram bins. **Important Note:** The SBC analysis **much more** time-consuming than a straight estimation, as it requires drawing a reasonable posterior sample for each of the testing parameterisations, and a large number of testing parameterizations are needed. If required, the `testSamples` and `testData` inputs can be broken down into batches and either analysed in parallel (if multiple GPUs are available) or separately, with the resulting histogram counts added at the end.

```python
# Chose the length of the sample and pick starting point for sampler
N = 149             # Number of draws
burn = 100          # Burn-in period to discard
init = np.zeros(begrsEst.num_param) # Again, a vector of 0s is good for centred samples.

# Run SBC with automatic thinning and an ESS cut-off of 10
SBC.run(N, burn, init, autoThin = True, essCutoff = 10)   

# Save results once run is complete
SBC.saveData(savePath/sbcFile)
```

Because MCMC methods can generate auto-correlated posterior samples, the `SBC` object uses thinning by default. This option can be switched off by setting `autoThin = False` in the `run` call, but it is recommended to rely on thinning. When thinning, the `SBC` object will start by drawing `N` samples, discarding `burn` of them and calculating the ESS. If the ESS is falls below 95% of the required  `N - burn` samples, the object will multiply the sample size by `(N - burn)/ESS - 1` and draw the missing samples. At that point the ESS is evaluated again and the methodology continues iteratively until the condition is met. As a protection against the number of draws blowing up for very small values of the ESS (for instance if the burn-in period is set too low and the early-chain samples are highly autocorrelated), a manual cut-off can be set using `essCutoff`. The increment becomes `(N - burn)/max(ESS, essCutoff) - 1` and in cases of low ESS values, the methodology might have to iterate several times, but the overall number of draws required will be smaller.

Finally, the SBC histogram can be accessed either directly from the `SBC` object or from the saved pickle file:

```python
# Direct access from object
hist = SBC.hist

# Loading from saved file
import pickle
import zlib

with  open(savePath/sbcFile, rb') as f:
  data = zlib.decompress(f.read(),0)

sbcData = pickle.loads(data,encoding="bytes")
hist = sbcData['hist']   

```
