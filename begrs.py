# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 08:48:12 2021

@author: Sylvain Barde, University of Kent

Implements Bayesian Estimation with a Gaussian Process Regression Surrogate.

Requires the following packages:

    torch
    gpytorch
    scipy
    sampyl-mcmc
    numpy
    arviz

Classes:

    begrsGPModel
    begrs
    begrsNutsSampler
    begrsSbc

Utilities:

    cholesky_factor

"""

import os
import sys
import pickle
import zlib
import time
import torch
import gpytorch
import numpy as np
import sampyl as smp

from tqdm import tqdm
from warnings import filterwarnings
from arviz.stats import ess
from scipy.optimize import minimize

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.lazy import (delazify, TriangularLazyTensor,
                           ConstantDiagLazyTensor, SumLazyTensor,
                           MatmulLazyTensor, DiagLazyTensor,
                           BlockDiagLazyTensor, KroneckerProductLazyTensor)

#------------------------------------------------------------------------------
# Cholesky factorisation utility
def cholesky_factor(induc_induc_covar):
    """
    Performs the Cholesky factorisation of the Kernel matrix of inducing points
    required for the whitening transformation used by GPytorch.

    Note: this utility is used by the toolbox and is not meant to be called
    directly by the user.

        Arguments:
            induc_induc_covar (AddedDiagLazyTensor):
                The kernel matrix for the inducing points

        Returns:
            (TriangularLazyTensor) :
                The corresponding Cholesky factorisation

    """

    L = psd_safe_cholesky(delazify(induc_induc_covar).double())

    return TriangularLazyTensor(L)

# Filter user warnings: catch CUDA initialisation warning if no GPU is present
filterwarnings("ignore", category = UserWarning)
#------------------------------------------------------------------------------
# Main classes
class begrsGPModel(gpytorch.models.ApproximateGP):
    """
    Underlying LMC model used byt the 'begrs' class.
    Extension of the 'gpytorch.models.ApproximateGP' class

        Attributes:
            variational_strategy (gpytorch.variational.LMCVariationalStrategy):
                The variational approximation for the GP
            mean_module (gpytorch.means.ConstantMean):
                The prediction mean function
            covar_module (gpytorch.kernels.ScaleKernel):
                The prediction covariance function

        Methods:
            __init__ :
                Initialises the class and associated variational strategy
            forward(x):
                Returns the GP prediction at x
    """

    # Multitask Gaussian process, with the batch dimension serving to store
    # the multiple tasks.

    def __init__(self, num_vars, num_param, num_latents, num_inducing_pts):
        """
        Initialises the class and associated variational strategy

            Arguments:
                num_vars (int):
                    Number of observable variables in the data
                num_param (int):
                    Number of models parameters to estimate
                num_latents (int):
                    Number of latent variables for LMC
                num_inducing_pts (int):
                    Number of inducing points for the variational strategy
        """
        # Separate inducing points for each latent function
        inducing_points = torch.rand(num_latents,
                                     num_inducing_pts,
                                     num_vars + num_param)

        # Mean field variational strategy (batched) for computational speedup
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            inducing_points.size(-2),
            batch_shape = torch.Size([num_latents])
        )

        # VariationalStrategy is wrapped a LMCVariationalStrategy to combine
        # the task-level distributions into a single multivariate normal
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations = True
            ),
            num_tasks = num_vars,
            num_latents = num_latents,
            latent_dim = -1
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules
        # Also batched, providing one module per latent variable
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        """
        Returns the GP prediction at x

        Arguments:
            x (Tensor):
                An input observation

        Returns:
            (gpytorch.distributions.MultivariateNormal) :
                The LMC prediction at x
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class begrs:
    """
    Main module class for the package. See function-level help for more details.

    Attributes:
        model (begrsGPModel):
            Instance of the begrsGPModel class
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood):
            Likelihood for the LMC model
        parameter_range (ndarray):
            Bounds for the parameter samples
        trainX (Tensor):
            Training (simulated) inputs
        trainY (Tensor):
            Training (simulated) outputs
        testX (Tensor):
            Evaluation (empirical) inputs
        testY (Tensor):
            Evaluation (empirical) outputs
        trainMean (ndarray):
            Mean of training data, used for normalisation
        trainStd (ndarray):
            Std.dev. of training data, used for normalisation
        losses (list):
            list of ELBO values for each training epoch
        KLdiv ():
            list of variational KL distance values for each training epoch
        N (int):
            Number of emprical observations available
        num_vars (int):
            Number of observable variables in the data
        num_param (int):
            Number of models parameters to estimate
        num_latents (int):
            Number of latent variables for the LMC
        num_inducing_pts (int):
            Number of inducing points for the variational strategy
        useGPU (bool):
            Flags use of GPU acceleration (if available)
        haveGPU (bool):
            Flags availability of GPU acceleration

    Methods:
        __init__ :
            Initialises an empty instance of the class
        center:
            Centers a raw parameter vector
        uncenter:
            Uncenters a centered parameter vector
        save:
            Saves the state of the underlying LMC surrogate
        load:
            Load a saved LMC surrogate
        setTrainingData:
            Build a dataset for LMC training
        setTestingData:
            Set an empirical dataset for estimation (LMC evaluation)
        train:
            Train the LMC surrogate on a pre-specified training dataset
        logP:
            Evaluate the log-likelihood of the LMC surrogate on the empirical
            data
        softLogPrior:
            Evaluate the logarithm of the soft minimal prior
        logPSingle (utility):
            Calculates the log-likelihood based on single transitions (diagonal
            covariance matrix)
        logPBatched (utility):
            Calculates the log-likelihood based on batches of transitions
        dlogP (utility):
            Calculates the gradient of the batched likelihood using the
            analytical derivation
    """

    def __init__(self, useGPU = True):
        """
        Initialises an empty instance of the begrs class

        Arguments:
            useGPU (bool):
                Flags use of GPU acceleration. True by default, only actually
                used if GPU acceleration is detected on initialisation.
        """
        # Initialise empty fields
        self.model = None
        self.likelihood = None
        self.parameter_range = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.trainMean = None
        self.trainStd = None
        self.losses = None
        self.KLdiv = None
        self.N = 0
        self.num_vars = 0
        self.num_param = 0
        self.num_latents = 0
        self.num_inducing_pts = 0
        self.useGPU = useGPU

        if torch.cuda.is_available():
            self.haveGPU = True
        else:
            self.haveGPU = False


    def center(self, sample):
        """
        Centers a raw parameter vector, based on the 'parameter_range' attribute
        of the begrs class

        Arguments:
            sample (ndarray):
                A raw sample

        Returns:
            sampleCntrd (ndarray):
                A centred sample
        """
        mean = (self.parameter_range[:,0] + self.parameter_range[:,1])/2
        stdDev = (self.parameter_range[:,1] - self.parameter_range[:,0])/np.sqrt(12)

        sampleCntrd = (sample - mean)/stdDev

        return sampleCntrd


    def uncenter(self, sampleCntrd):
        """
        Uncenters a centred parameter vector, based on the 'parameter_range'
        attribute of the begrs class

        Arguments:
            sampleCntrd (ndarray):
                A centred sample

        Returns:
            sample (ndarray):
                A raw sample
        """
        mean = (self.parameter_range[:,0] + self.parameter_range[:,1])/2
        stdDev = (self.parameter_range[:,1] - self.parameter_range[:,0])/np.sqrt(12)

        sample = mean + stdDev*sampleCntrd

        return sample


    def save(self, path):
        """
        Saves the state of the underlying LMC surrogate

        This is designed to save the state of the LMC surrogate after training,
        attributes relating to the testing/evaluation methods are NOT saved.
        This requires setting the testing data explicitly every time an
        estimation needs to be run, but ensures that saved LMC surrogates can be
        used on multiple emprirical datasets. See the 'setTestingData' help for
        more details on this aspect.

        The specified saving folder is created by the method, and the method
        will fail if the path points to a pre-existing folder. This is to avoid
        overwriting pre-existing saved states.

        Arguments:
            path (str):
                A path to a folder - the folder cannot already exist

        Returns:
            None
        """
        print(u'\u2500' * 75)
        print(' Saving model to: {:s} '.format(path), end="", flush=True)

        # Check saving directory exists
        if not os.path.exists(path):

            if self.model is not None:
                os.makedirs(path,mode=0o777)

                saveDict = {'parameter_range':self.parameter_range,
                            'trainX':self.trainX,
                            'trainY':self.trainY,
                            'trainMean':self.trainMean,
                            'trainStd':self.trainStd,
                            'losses':self.losses,
                            'KLdiv':self.KLdiv,
                            'num_vars':self.num_vars,
                            'num_param':self.num_param,
                            'num_latents':self.num_latents,
                            'num_inducing_pts':self.num_inducing_pts}

                torch.save(saveDict,
                           path + '/data_parameters.pt')

                torch.save(self.model.state_dict(),
                            path + '/model_state.pt')
                torch.save(self.likelihood.state_dict(),
                            path + '/likelihood_state.pt')

                print(' - Done')

            else:

                print('\n Cannot write to {:s}, empty model'.format(path))

        else:

            print('\n Cannot write to {:s}, folder already exists'.format(path))


    def load(self, path):
        """
        Load a saved LMC surrogate

        Arguments:
            path (str):
                A path to a folder containing a saved state

        Returns:
            None
        """
        print(u'\u2500' * 75)
        print(' Loading model from: {:s} '.format(path), end="", flush=True)

        if os.path.exists(path):

            parameter_path = path + '/data_parameters.pt'
            likelihood_path = path + '/likelihood_state.pt'
            model_path = path + '/model_state.pt'

            if os.path.exists(parameter_path):
                begrs_dict = torch.load(parameter_path)
                self.parameter_range = begrs_dict['parameter_range']
                self.trainX = begrs_dict['trainX']
                self.trainY = begrs_dict['trainY']
                self.trainMean = begrs_dict['trainMean']
                self.trainStd = begrs_dict['trainStd']
                self.losses = begrs_dict['losses']
                self.KLdiv = begrs_dict['KLdiv']
                self.num_vars = begrs_dict['num_vars']
                self.num_param = begrs_dict['num_param']
                self.num_latents = begrs_dict['num_latents']
                self.num_inducing_pts = begrs_dict['num_inducing_pts']

            else:

                print("\n Missing file 'data_parameters.pt' in: {:s}".format(path))

            if os.path.exists(likelihood_path):
                lik_state_dict = torch.load(likelihood_path)
                self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self.num_vars)
                self.likelihood.load_state_dict(lik_state_dict, strict = False)

                if self.haveGPU is True and self.useGPU is True:
                    self.likelihood = self.likelihood.cuda()

            else:

                print("\n Missing file 'likelihood_state.pt' in: {:s}".format(path))

            if os.path.exists(model_path):

                mod_state_dict = torch.load(model_path)
                self.model = begrsGPModel(self.num_vars,
                                      self.num_param,
                                      self.num_latents,
                                      self.num_inducing_pts)
                self.model.load_state_dict(mod_state_dict, strict = False)

                if self.haveGPU is True and self.useGPU is True:
                    self.model = self.model.cuda()

            else:

                print("\n Missing file 'model_state.pt' in: {:s}".format(path))

            print(' - Done')

        else:

            print('\n Cannot load from {:s}, folder does not exist'.format(path))


    def setTrainingData(self, trainingData, trainingSamples, parameter_range,
                        wins = None, normalise = True):
        """
        Build a dataset for LMC training

        The method builds and saves the training data structure based on the
        simulated data and the simulation samples. The input structure in
        particular will contain lags of the observable variables and the
        parameter setttings. NaN values are dropped from the dataset.

        Note the specific structure that the first 3 arguments must take.

        Arguments:
            trainingData (ndarray)
                3D numpy array containing the simulated data. Structure is:

                    T x num_vars x numSamples

            trainingSamples (ndarray)
                2D numpy array containing the parameter vectors corresponding
                to each simulation run. Structure is:

                    numSamples x num_param

            parameter_range (ndarray)
                2D numpy array containing the parameter bounds for the samples.
                Structure is:

                    num_param x 2

                Lower bound in the first column, upper bound in the second.

            wins (None or float)
                float in [0,1] interval, sets the tail quantiles to remove by
                winsorization. 'wins = 0.05' will winsorize the top/bottom
                2.5% of the training data for each variable. Set to 'None' by
                default, omit this parameter if winsorization is not required.

            normalise (bool)
                Boolean flag for normalising each variable in the training
                data to zero mean and unit standard deviation. Set to 'True'
                by default.

        Returns:
            None
        """
        print(u'\u2500' * 75)
        print(' Setting training data set', end="", flush = True)

        # Check if dimensions of inuts match
        numSamples = trainingSamples.shape[0]

        if (numSamples == trainingData.shape[2] and
            trainingSamples.shape[1] == parameter_range.shape[0]):

            # Allocate class variables from datasets
            self.parameter_range = parameter_range
            self.num_param = trainingSamples.shape[1]
            self.num_vars = trainingData.shape[1]
            numObs = trainingData.shape[0]


            # Winsorise data if required (NaN - robust)
            outStr = ''
            if wins is not None:
                outStr += ' - Winsorised top/bottom {:.2f}%\n'.format(100*wins)

                LB = np.nanquantile(np.vstack(np.moveaxis(trainingData,-1,-2)),
                                   wins/2,axis = 0)[None,:,None]
                UB = np.nanquantile(np.vstack(np.moveaxis(trainingData,-1,-2)),
                                   1-wins/2,axis = 0)[None,:,None]

                LBCheck = np.less(trainingData, LB,
                                  where =~ np.isnan(trainingData))
                UBCheck = np.greater(trainingData, UB,
                                  where =~ np.isnan(trainingData))

                trainingData[LBCheck] = np.tile(LB,
                                                [numObs,1,numSamples])[LBCheck]
                trainingData[UBCheck] = np.tile(UB,
                                                [numObs,1,numSamples])[UBCheck]

            # Normalise data if required  (NaN - robust)
            if normalise:
                outStr += ' - Normalised variables (0 mean, 1 std. dev.)\n'

                self.trainMean = np.nanmean(trainingData,
                                         axis = (0,2))[None,:,None]
                self.trainStd = np.nanstd(trainingData,
                                       axis = (0,2))[None,:,None]

                trainingData -= self.trainMean
                trainingData /= self.trainStd

            # Repackage training data and samples
            paramInds = self.num_vars + self.num_param
            train_x_array = np.zeros([numSamples*(numObs-1),paramInds])
            train_y_array = np.zeros([numSamples*(numObs-1),self.num_vars])

            samples = self.center(trainingSamples)
            for i in range(numSamples):

                y = trainingData[1:,:,i]
                x = trainingData[0:-1,:,i]
                sample = samples[i,:]

                train_y_array[i*(numObs-1):(i+1)*(numObs-1),:] = y
                train_x_array[i*(numObs-1):(i+1)*(numObs-1),
                              0:self.num_vars] = x
                train_x_array[i*(numObs-1):(i+1)*(numObs-1),
                              self.num_vars:paramInds] = np.tile(sample,
                                                                  (numObs-1,1))

            # Remove rows that contain NaN values
            yNaNs = np.any(np.isnan(train_y_array), axis=1)
            xNaNs = np.any(np.isnan(train_x_array), axis=1)
            dropNans = np.where(np.logical_or(xNaNs,yNaNs))[0]
            if not dropNans.size == 0:
                outStr += ' - Dropping {:d} NaN observations\n'.format(len(dropNans))

            train_x_array = np.delete(train_x_array, dropNans, axis=0)
            train_y_array = np.delete(train_y_array, dropNans, axis=0)

            # Convert to tensors and store
            self.trainX = torch.from_numpy(train_x_array).float()
            self.trainY = torch.from_numpy(train_y_array).float()

            print(' - Done', flush = True)
            print('{:s}\n'.format(outStr))
            print(' N' + u'\u00B0' + ' of parameters: {:>5}'.format(self.num_param))
            print(' N' + u'\u00B0' + ' of variables: {:>5}'.format(self.num_vars))
            print(' N' + u'\u00B0' + ' of parameter samples: {:>5}'.format(numSamples))

        else:

            print(' Error, inconsistent sample dimensions')
            print(' N' + u'\u00B0' +' of parameters: ' + ' '*5 +'N' + u'\u00B0' + \
                  ' of parameter samples: ')
            print(' - in range matrix: {:>5}'.format(parameter_range.shape[0]),
                  end="", flush=True)
            print(' - in DOE matrix: {:>5}'.format(trainingSamples.shape[1]))
            print(' - in training data: {:>5}'.format(trainingData.shape[2]),
                  end="", flush=True)
            print(' - in DOE matrix: {:>5}'.format(trainingSamples.shape[0]),
                  end="", flush=True)


    def setTestingData(self, testingData):
        """
        Set an empirical dataset for estimation (LMC evaluation)

        The method builds the evaluation structure from the empirical dataset,
        precomputes some fixed likelihood components (to save time) and sets
        the model to evaluation mode.

        Notes:
        - If the 'normalise' option was used for the training data (default),
          the testing data is automatically normalised using the same mean and
          standard deviation.
        - NaN values are dropped from the dataset.
        - The testing data will need to be explicitly set every time the user
          wants to load a given begrs surrogate to run an empirical estimation.
          This is to avoid using the wrong empirical data, and also to ensure
          that the LMC model is set evaluation mode prior to calculating
          surrogate likelihoods.

        This means that the following attributes of the begrs class are set by
        this method but are not saved by the 'save' method:
            N
            testX
            testY
            L
            lmcCoeff
            middle_diag
            noise

        Arguments:
            testingData (ndarray)
                2D numpy array containing the emprical data. Structure is:

                    T x num_vars

        Returns:
            None
        """
        print(u'\u2500' * 75)
        print(' Setting testing data set', end="", flush=True)

        # Check consistency with number of variables
        if testingData.shape[1] == self.num_vars:
            outStr = ''

            # Normalise the testing data if needed.
            if self.trainMean is not None and self.trainStd is not None:
                testingData -= self.trainMean.squeeze(-1)
                testingData /= self.trainStd.squeeze(-1)
                outStr += ' - Normalised variables (0 mean, 1 std. dev.)\n'
                
            # Remove rows that contain NaN values
            testNaNs = np.any(np.isnan(testingData), axis=1)
            dropNans = np.where(testNaNs)[0]
            testingData = np.delete(testingData, dropNans, axis=0)

            if not dropNans.size == 0:
                outStr += ' - Dropping {:d} NaN observations\n'.format(
                            len(dropNans))

            # Save testing data to class attributes
            self.N = testingData.shape[0] - 1
            self.testY = torch.from_numpy(testingData[1:,:]).float()

            test_x_array = np.zeros([self.N, self.num_vars+self.num_param])
            test_x_array[:,0:self.num_vars] = testingData[0:-1]
            self.testX = torch.from_numpy(test_x_array).float()

            # CUDA check
            if self.haveGPU is True and self.useGPU is True:
                self.testY = self.testY.cuda()
                self.testX = self.testX.cuda()

            self.model.eval()
            self.likelihood.eval()

            print(' - Done', flush = True)
            print('{:s}'.format(outStr))

            print(' Precomputing Likelihood components', end="", flush=True)

            # Whitening (Cholesky) on the inducing points kernel covariance
            indpts = self.model.variational_strategy.base_variational_strategy.inducing_points
            induc_induc_covar = self.model.covar_module(indpts,indpts).add_jitter()
            self.L = cholesky_factor(induc_induc_covar)

            # LMC coefficients
            lmc_coefficients = self.model.variational_strategy.lmc_coefficients.expand(*torch.Size([self.num_latents]),
                                                                                self.model.variational_strategy.lmc_coefficients.size(-1))
            lmc_factor = MatmulLazyTensor(lmc_coefficients.unsqueeze(-1),
                                          lmc_coefficients.unsqueeze(-2))
            lmc_mod = lmc_factor.unsqueeze(-3)
            self.lmcCoeff = delazify(lmc_mod)

            # Diagonal predictive covariance components
            variational_inducing_covar = self.model.variational_strategy.base_variational_strategy.variational_distribution.covariance_matrix
            self.middle_diag = self.model.variational_strategy.prior_distribution.lazy_covariance_matrix.mul(-1).representation()[0]
            self.middle_diag += torch.diagonal(
                variational_inducing_covar,
                offset = 0,dim1=-2,dim2=-1
                )

            # Noise components
            task_noises = DiagLazyTensor(self.likelihood.noise_covar.noise)
            noise = ConstantDiagLazyTensor(
                self.likelihood.noise,
                diag_shape=task_noises.shape[-1])
            self.noise = delazify(task_noises + noise)

            print(' - Done')

        else:

            print(' - Error, inconsistent number of variables')
            print(' N' + u'\u00B0' +' of variables:')
            print(' - in training data: {:>5}'.format(self.num_vars),
                  end="", flush=True)
            print(' - in test data: {:>5}'.format(testingData.shape[1]))

    def train(self, num_latents, num_inducing_pts, batchsize, epochs,
              learning_rate, shuffle = True):
        """
        Train the LMC surrogate on a pre-specified training dataset

        Note: users must have loaded the training data using the
        'setTrainingData' method prior to running the 'train' method

        Arguments:
            num_latents (int):
                Number of latent variables for the LMC
            num_inducing_pts (int):
                Number of inducing points for the variational strategy
            batchsize (int):
                Size of the minibatches drawn from the full training data within
                each epoch iteration
            epochs (int):
                Number of epochs for which to train the data. Each epoch will
                train over the full training data, broken into batches
            learning_rate (float):
                Learning rate of the Adam optimiser used to learn the LMC
                parameters.
            shuffle (bool):
                Flags reshuffling to training observation prior to batching in
                each training epoch. If false, each epoch will iterate on the
                same sequences of training data batches. Default is 'True'

        Returns:
            None
        """
        print(u'\u2500' * 75)
        print(' Training gaussian surrogate model')
        self.num_latents = num_latents
        self.num_inducing_pts = num_inducing_pts

        # CUDA check
        if self.haveGPU:

            print(' CUDA available',end="", flush=True)

            if self.useGPU:

                print(' - Using GPU', flush=True)

            else:

                print(' - Using CPU', flush=True)

        else:

            print(' CUDA not available - Using CPU')


        # Create data loader from training data
        train_dataset = TensorDataset(self.trainX, self.trainY)
        train_loader = DataLoader(train_dataset,
                                  batch_size = batchsize,
                                  shuffle = shuffle)

        # Initialise model and training likelihood in training mode
        self.model = begrsGPModel(self.num_vars,
                                  self.num_param,
                                  self.num_latents,
                                  self.num_inducing_pts)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks = self.num_vars)

        self.model.train()
        self.likelihood.train()

        # set Adam optimiser for the parameters
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
            ], lr=learning_rate)

        # ELBO loss function for parameter optimisation
        mll = gpytorch.mlls.VariationalELBO(self.likelihood,
                                            self.model,
                                            num_data = self.trainY.size(0))

        # Run optimisation
        self.losses = []
        self.KLdiv = []
        for i in range(epochs):

            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(train_loader,
                                  desc='Iteration {:<3}'.format(i+1),
                                  leave = True,
                                  file=sys.stdout)

            iterLoss = []
            iterKL = []
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = - mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                minibatch_iter.refresh()
                loss.backward()
                optimizer.step()

                # Save loss and KL divergence for diagnostics
                kl_term = self.model.variational_strategy.kl_divergence().div(
                    self.trainY.size(0))
                iterLoss.append(loss.item())
                iterKL.append(kl_term.item())

            minibatch_iter.close()
            self.losses.append(iterLoss)
            self.KLdiv.append(iterKL)


    def logP(self, theta_base, batch = False, batch_size = 40):
        """
        Evaluate the log-likelihood of the LMC surrogate on the empirical data

        This method additionally returns the gradient of the log-likelihood with
        respect to the parameters, calculated using the autograd feature of
        the torch implementation.

        Two options are available, controlled by the 'batch' argument:
        - Single: the contribution each empirical observation is evaluated
          separately (only the main diagonal of the LMC covariance is used).
          This corresponds to the derivations in the main paper.
        - Batched: The emprical observations are evaluated in batches. This is
          less correct (as transitions become correlated) but has the benefit of
          being faster for average-sized batches (~ 40-50).

        Note: users must have loaded the empirical data using the
        'setTestingData' method prior to running the 'logP' method. In addition
        this method assumes that the theta_base parameter vector is already
        centred.

        Arguments:
            theta_base (ndarray):
                1D numpy array containing a centred candidate parameterisation
            batch (bool):
                Flags that the log-likelihood should be evaluated over batches
                of the empirical data. Default is 'False'
            batch_size (int):
                Size of the batches for batched mode. Default is 40.

        Returns:
            returnValue (tuple), containing the following values:
                logP (numpy float64):
                    The log-likelihood evaluated at theta_base
                theta_grad (numpy float64):
                    The gradient of the log-likelihood evaluated at theta_base
        """
        theta = torch.from_numpy(theta_base[None,:]).float()
        theta.requires_grad_(True)

        if batch is False:

            logP = self.logPSingle(theta)

        else:

            logP = self.logPBatched(theta,batch_size)

        theta_grad = theta.grad

        if self.haveGPU is True and self.useGPU is True:

            returnValue = (np.float64(
                                logP.detach().cpu().numpy()),
                           np.float64(
                                theta_grad.detach().cpu().numpy().flatten()))

        else:

            returnValue = (np.float64(
                                logP.detach().numpy()),
                           np.float64(
                                theta_grad.detach().numpy().flatten()))

        return returnValue


    def softLogPrior(self, theta, k = 20):
        """
        Evaluate the logarithm of the soft minimal prior
        This corresponds to the double logistic minimal prior in the paper.

        Note: this method assumes that the theta parameter vector is already
        centred.

        Arguments:
            theta (ndarray):
                1D numpy array containing a centred candidate parameterisation
            k (int):
                Slope parameter of the logistic functions. Higher values lead to
                sharper boundary transitions. Default is 20.

        Returns:
            (tuple), containing the following values:
                (numpy float64):
                    The log-prior evaluated at theta
                theta_grad (numpy float64):
                    The gradient of the log-prior evaluated at theta
        """
        # convert to array, protect against overflows
        sample = np.clip(np.asarray(theta),
                         -20,
                         20)

        f_x = np.exp(-k*(sample + 3**0.5))
        g_x = np.exp( k*(sample - 3**0.5))

        prior = - np.log(1 + f_x) - np.log(1 + g_x)
        grad = k*(f_x/(1+f_x) - g_x/(1+g_x))

        return (sum(prior),grad)


    def logPSingle(self, theta):
        """
        (utility) Calculates the log-likelihood based on single transitions
        (diagonal covariance matrix)

        Note: this method is not meant to be used directly, users should call
        it through the 'logP' method.

        Arguments:
            theta (ndarray):
                1D numpy array containing a centred candidate parameterisation

        Returns:
            logP (Tensor):
                The log-likelihood evaluated at theta
        """
        paramInds = self.num_vars + self.num_param
        testX = self.testX.clone()
        testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))

        n = self.testY.shape[0]*self.testY.shape[1]
        batch = torch.cat([testX.unsqueeze(0)]*self.num_latents, dim=0)
        indpts = self.model.variational_strategy.base_variational_strategy.inducing_points

        preds = self.likelihood(self.model(batch))
        diff = self.testY.reshape([n]) - preds.mean.reshape([n])

        #--------------------------------
        # Uses the precomputed components set with the testing data
        # Covariance kernel components
        induc_data_covar = self.model.covar_module(indpts,batch).evaluate()
        data_data_covar = self.model.covar_module(batch,batch)

        # Diagonal predictive covariance
        interp_term = self.L.inv_matmul(induc_data_covar.double()).to(testX.dtype)
        predictive_covar_diag = (
            interp_term.pow(2) * self.middle_diag.unsqueeze(-1)
            ).sum(-2).squeeze(-2)
        predictive_covar_diag += torch.diagonal(
            delazify(data_data_covar.add_jitter(1e-4)),
            offset = 0,dim1=-2,dim2=-1
            )

        # Block diagonal LMC wrapping of diagonal predictive covariance
        covar_mat_blocks = torch.mul(self.lmcCoeff,
                                      predictive_covar_diag.unsqueeze(-1).unsqueeze(-1)
                                      ).sum(-4) + self.noise
        covar_mat = BlockDiagLazyTensor(covar_mat_blocks)

        # Gaussian likelihood calculation
        inv_quad_term = covar_mat.inv_quad(diff.unsqueeze(-1))
        logP = -0.5 * sum([inv_quad_term,
                           sum(covar_mat_blocks.det().log()),
                           diff.size(-1) * np.log(2 * np.pi)])

        logP.backward(retain_graph=True)

        return logP


    def logPBatched(self, theta, batch_size):
        """
        (utility) Calculates the log-likelihood based on batches of transitions

        Note: this method is not meant to be used directly, users should call
        it through the 'logP' method.

        Arguments:
            theta (ndarray):
                1D numpy array containing a centred candidate parameterisation
            batch_size (int):
                Size of the batches

        Returns:
            logP (Tensor):
                The log-likelihood evaluated at theta

        """
        paramInds = self.num_vars + self.num_param
        testX = self.testX.clone()
        testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))

        # Create data loader for batches
        test_dataset = TensorDataset(testX, self.testY)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        # Get log probability for each batch directly from GPytorch
        logP = 0
        for x_batch, y_batch in test_loader:

            predictions = self.likelihood(self.model(x_batch))
            logP += predictions.log_prob(y_batch)

        logP.backward(retain_graph=True)

        return logP


    def dlogP(self, theta, batch_size = 40):
        """
        (utility) Calculates the gradient of the batched likelihood using the
        analytical derivation.

        Note: this method is provided as a validation of the analytical gradient
        derivations of the paper. Its purpose is to verify that this matches the
        likelihood gradient obtained via the autograd feature of Torch. It
        should produce the same gradient as 'logP' in batched mode
        (batch='True') for a given theta and batch_size, but is significantly
        slower. As a result, users should not use this method for any other
        purpose than checking the gradient.

        Arguments:
            theta (ndarray):
                1D numpy array containing a centred candidate parameterisation
            batch_size (int):
                Size of the batches. Default is 40

        Returns:
            dL (numpy float64):
                The gradient of the likelihood evaluated at theta

        """
        # Hard-wired parameters
        latent_dim = 0
        num_dim = 2

        paramInds = self.num_vars+self.num_param
        self.testX[:, self.num_vars:paramInds] = torch.from_numpy(
            np.tile(theta[None,:],(self.N,1))).float()

        # Data loader for batches
        test_dataset = TensorDataset(self.testX, self.testY)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        # Common variables:
        #   inducing points, inducing values, L, middle term, LMC terms
        lenscl = self.model.covar_module.base_kernel.raw_lengthscale
        indpts = self.model.variational_strategy.base_variational_strategy.inducing_points
        inducing_values = self.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean
        variational_inducing_covar = self.model.variational_strategy.base_variational_strategy.variational_distribution.covariance_matrix
        induc_induc_covar = self.model.covar_module(indpts).add_jitter()
        L = cholesky_factor(induc_induc_covar)
        mod_batch_shape = indpts.shape[0]

        middle_term = self.model.variational_strategy.prior_distribution.lazy_covariance_matrix.mul(-1)
        middle_term += DiagLazyTensor(
                        variational_inducing_covar.diagonal(dim1=1,dim2=2))

        # LMC wrapping variables
        lmc_coefficients = self.model.variational_strategy.lmc_coefficients.expand(*[mod_batch_shape],
                                                                          self.model.variational_strategy.lmc_coefficients.size(-1))
        lmc_factor = MatmulLazyTensor(lmc_coefficients.unsqueeze(-1),
                                      lmc_coefficients.unsqueeze(-2))

        # Iterate over minibatches of data
        dL = np.zeros(len(theta))
        for x_batch, y_batch in test_loader:

            # Common batch likelihood variables
            preds = self.likelihood(self.model(x_batch))
            covar_mat = preds.lazy_covariance_matrix
            n = preds.covariance_matrix.shape[0]
            diff = y_batch.reshape([n]) -  preds.mean.reshape([n])

            induc_data_covar = self.model.covar_module(
                                indpts,x_batch).evaluate()
            interp_term = L.inv_matmul(
                            induc_data_covar.double()).to(indpts.dtype)

            K_mn = self.model.covar_module(indpts,x_batch).evaluate()

            # Parameter-specific gradient components
            # if self.useGPU is True and torch.cuda.is_available():
            if self.haveGPU is True and self.useGPU is True:
                dmean_tens = torch.tensor(()).new_empty((len(theta), n, 1),
                    dtype=indpts.dtype).cuda()
                dcovar_tens = torch.tensor(()).new_empty((len(theta), n, n),
                    dtype=indpts.dtype).cuda()
            else:
                dmean_tens = torch.tensor(()).new_empty((len(theta), n, 1),
                    dtype=indpts.dtype)
                dcovar_tens = torch.tensor(()).new_empty((len(theta), n, n),
                    dtype=indpts.dtype)

            for i in range(len(theta)):

                ind = self.num_vars + i

                # 1. derivative of the kernel-based covariance
                indpts_dev = indpts[:,:,ind].unsqueeze(-1) - x_batch[0,ind]
                dK_mn = indpts_dev*K_mn/(lenscl**2)    # uses Hadamard product

                # 2a. derivative of mean
                dinterp_term = L.inv_matmul(dK_mn.double()).to(x_batch.dtype)
                dpredictive_mean = (dinterp_term.transpose(-1, -2) @
                                    inducing_values.unsqueeze(-1)).squeeze(-1)

                dmean = dpredictive_mean.permute(*range(0, latent_dim),
                                                 *range(latent_dim + 1,
                                                        num_dim),
                                                 latent_dim)
                dmean = dmean @ lmc_coefficients
                dmean_tens[i,:,:] = dmean.reshape([n,1])

                # 2b. derivative of variance
                covar_component = MatmulLazyTensor(
                                        dinterp_term.transpose(-1, -2),
                                        middle_term @ interp_term)
                dpredictive_covar = SumLazyTensor(
                                        covar_component.transpose(-1, -2),
                                        covar_component)

                # LMC wrapping
                dcovar = KroneckerProductLazyTensor(dpredictive_covar,
                                                    lmc_factor)
                dcovar = dcovar.sum(latent_dim)
                # dcovar = dcovar.add_jitter(1e-6)
                dcovar_tens[i,:,:] = delazify(dcovar)

            # 3. Batched derivative of log likelihood
            # 3a. derivative of the log determinant
            covmat = torch.matmul(torch.inverse(delazify(covar_mat)).unsqueeze(-3),
                                  dcovar_tens) # FASTER!
            dlogdet = torch.diagonal(covmat, dim1=-2, dim2=-1).sum(-1)

            # 3b. derivative of the Mahanalobis term w.r.t. covariance
            dSinv = - torch.matmul(covmat,
                                    torch.inverse(delazify(covar_mat)
                                                  ).unsqueeze(-3))
            dS1_full = torch.matmul(torch.matmul(diff.unsqueeze(-1),
                                                  diff.unsqueeze(-2)
                                                  ).unsqueeze(-3),
                                            dSinv)
            dS1 = torch.diagonal(dS1_full, dim1=-2, dim2=-1).sum(-1)

            # derivative of the Mahanalobis term w.r.t. mean
            S1 = torch.matmul(torch.inverse(preds.covariance_matrix).unsqueeze(-3),
                              dmean_tens)
            dS2 = torch.matmul(diff.unsqueeze(-2).unsqueeze(-3),S1)

            # Final gradient
            dlobs = -0.5*(dlogdet + dS1) + dS2.squeeze(-2).squeeze(-1)

            # if self.useGPU is True and torch.cuda.is_available():
            if self.haveGPU is True and self.useGPU is True:

                returnValue = dlobs.cpu().detach().numpy()

            else:

                returnValue = dlobs.detach().numpy()

            dL += returnValue

        return dL

class begrsNutsSampler:
    """
    NUTS Sampler class for the package. See function-level help for more
    details.

    Attributes:

        begrsModel (begrs object)
            Instance of the begrs class
        logP (function)
            Function providing the log posterior
        mode (ndarray)
            vector of parameter values at the posterior mode
        nuts (sampyl.NUTS object)
            Instance of the sampyl NUTS sampler class

    Methods:
        __init__ :
            Initialises an empty instance of the class
        minESS:
            Calculates the effecgive sample size of a sample
        setup:
            Configure the NUTS smppler
        run:
            Run the NUTS sampler
    """

    def __init__(self,begrsModel,logP):
        """
        Initialises an empty instance of the class


        Arguments:
            begrsModel (begrs):
                An instange of a begrs surrogate model
            logP (function):
                A user-defined function providing the log posterior.

        """
        # Initialise empty fields
        self.begrsModel = begrsModel
        self.logP = logP
        self.mode = None
        self.nuts = None

    def minESS(self,array):
        """
        Calculates the effecgive sample size of a sample


        Arguments:
            array (ndarray):
                A posterior sample produced by the NUTS sampler

        Returns:
            float
                The effective sample size of the posterior sample.
        """

        essVec = np.zeros(array.shape[1])
        for i in range(array.shape[1]):
            essVec[i] = ess(array[:,i])

        return min(essVec)

    def setup(self, data, init):
        """
        Configure the NUTS sampler


        Arguments:
            data (ndarray):
                The empirical dataset required to calculate the likelihood.
            init (ndarray):
                A centered vector of initial values for the parameter vector.
        """

        print('NUTS sampler initialisation')

        # Set data as the BEGRS testing set
        self.begrsModel.setTestingData(data)

        # Find posterior mode
        print('Finding MAP vector')
        t_start = time.time()
        nll = lambda *args: tuple( -i for i in self.logP(*args))
        sampleMAP = minimize(nll, init,
                             method='L-BFGS-B',
                             bounds = self.begrsModel.num_param*[(-3**0.5,3**0.5)],
                             jac = True)
        self.mode = sampleMAP.x
        print(' {:10.4f} secs.'.format(time.time() - t_start))

        # Setup NUTS
        t_start = time.time()
        start = smp.state.State.fromfunc(self.logP)
        start.update({'sample': self.mode})
        E_cutoff = -2*self.logP(self.mode)[0]
        scale = np.ones_like(init)
        self.nuts = smp.NUTS(self.logP,
                        start,
                        scale = {'sample': scale},
                        grad_logp = True,
                        step_size = 0.01,
                        Emax = E_cutoff)

    def run(self, N, burn = 0):
        """
        Run the NUTS sampler

        Arguments:
            N (int):
                Number of NUTS samples to draw from the posterior
            burn (int)
                Numberr of burn-in samples to discard. The default is 0.

        Returns:
        -------
            posteriorSample (ndarray):
                (N - burn) posterior NUTS samples
        """

        print('NUTS sampling')

        # Draw samples from posterior
        t_start = time.time()
        chain = self.nuts.sample(N, burn=burn)

        # Process chain to uncenter samples
        posteriorSample = np.zeros([N-burn,self.begrsModel.num_param])
        for i, sampleCntrd in enumerate(chain.tolist()):
            posteriorSample[i,:] = self.begrsModel.uncenter(sampleCntrd)

        print(' {:10.4f} secs.'.format(time.time() - t_start))

        return posteriorSample

class begrsSbc:
    """
    Simulated Bayesian Computing class for the package. See function-level
    help for more details.

    Based on:

        Talts, Sean, Michael Betancourt, Daniel Simpson, Aki Vehtari, and
        Andrew Gelman (2018) “Validating Bayesian inference algorithms with
        simulation-based calibration,” arXiv preprint arXiv:1804.06788.

    Attributes:

        testSamples (ndarray)
            2-dimensional array of testing parameterizations
        testSamples (ndarray)
            3-dimensional array of simulated data
        hist (ndarray)
            Rank histogram
        histBins (ndarray)
            Bins for the rank histogram
        numParam (int)
            Number of parameters in a posterior sample
        numSamples (int)
            Number of samples in SBC analysis
        posteriorSampler (begrsNutsSampler)
            Instance of the begrsNutsSampler class used as the sampler
        posteriorSamplesMC (list of ndarrays)
            list of posterior samples for each testing parametrization
        posteriorSamplesESS (list of floats)
            list of effecive sample sizes for the posterior samples

    Methods:
        __init__ :
            Initialises an empty instance of the class
        saveData:
            Save the result of the SBC analysis
        setTestData:
            Set the testing samples and data for the SBC analysis
        setPosteriorSampler:
            Set the posterior samplet for the SBC analysis
        run:
            Run the SBC analysis
    """

    def __init__(self):
        """
        Initialises an empty instance of the class

        No input arguments

        """

        self.testSamples = None
        self.testData = None
        self.hist = None
        self.histBins = None
        self.numParam = None
        self.numSamples = None
        self.posteriorSampler = None
        self.posteriorSamplesMC = []
        self.posteriorSamplesESS = []

    def saveData(self, path):
        """
        Save the result of the SBC analysis

        Notes:
        - Data is saved as a pickled and zipped Dict
        - Existing files will be overwritten
        - Variables saved are:
            testSamples
            testData
            hist
            posteriorSamplesMC

        Arguments:
            path (str):
                A path to a file
        """

        print(u'\u2500' * 75)
        print(' Saving SBC run data to: {:s} '.format(path), end="", flush=True)

        # Check saving already exists
        if not os.path.exists(path):

            dirName,fileName = os.path.split(path)

            if not os.path.exists(dirName):
                os.makedirs(dirName,mode=0o777)

            saveDict = {'testSamples':self.testSamples,
                        'testData':self.testData,
                        'hist':self.hist,
                        'posteriorSamples':self.posteriorSamplesMC}

            fil = open(path,'wb')
            fil.write(zlib.compress(pickle.dumps(saveDict, protocol=2)))
            fil.close()

            print(' - Done')

        else:

            print('\n Cannot write to {:s}, file already exists'.format(path))

    def setTestData(self, testSamples, testData):
        """
        Set the testing samples and data for the SBC analysis

        Note the specific structure that the 2 arguments must take.

        Arguments:
            testSamples (ndarray)
                2D numpy array containing the parameter vectors drawn from the
                prior. Structure is:

                    numSamples x numParams

            testData (ndarray)
                3D numpy array containing the simulated data corresponding to
                each parameter sample. Structure is:

                    T x num_vars x numSamples
        """

        print(u'\u2500' * 75)
        print(' Setting test samples & data', end="", flush=True)

        if testSamples.shape[0] == testData.shape[-1]:

            self.testData = testData
            self.numSamples = testData.shape[-1]
            self.testSamples = testSamples
            self.numParam = testSamples.shape[-1]
            print(' - Done')

        else:
            print(' - Error, inconsistent number of samples')
            print(' N' + u'\u00B0' +' of samples:')
            print(' - in test samples: {:>5}'.format(testSamples.shape[0]))
            print(' - in test data: {:>5}'.format(testData.shape[-1]))

    def setPosteriorSampler(self, posteriorSampler):
        """
        Set the posterior samplet for the SBC analysis

        Arguments:
            posteriorSampler (list of ndarrays)
                Instance of the begrsNutsSampler class
        """

        self.posteriorSampler = posteriorSampler

    def run(self, N, burn, init, autoThin = True, essCutoff = 0):
        """
        Run the SBC analysis

        Produce a rank histogram with (N - burn + 1) bins by drawing (N-burn)
        samples for each parameter sample in the testing samples.

        Notes:
        - If auto-thin is active (true by default), the algoroithm will
          draw posterior samples until the effective sample size of the draws
          is (N-burn). This setting allows to control for autocorrelation in
          the posterior samples The histogram is then calculated from
          normalised counts of the entire set of posterior samples.
        - When auto-thin is active, the algorithm first draws (N - burn)
          posterior samples, then calculates the ESS. If the ESS is less than
          95% of the required length, the number of extra samples to draw is:

              (N - burn)/ESS - 1

          This continues iteratively until the ESS > 0.95 (N-burn) condition is
          met.

          If the ESS is very low (<5), then the size of the extra run will
          dramatically increase. This can happen for example if the burn-in
          period is too short. The essCutoff parameter protects against this by
          specifying a minimal size to the ESS used in the sample size
          calculation. This becomes:

              (N - burn)/max(ESS, essCutoff) - 1

          By default this is zet to 0, which is the same as specifying no
          cutoff.

        Arguments
        ----------
            N (int):
                Number of NUTS samples to draw from the posterior
            burn (int)
                Numberr of burn-in samples to discard.
            init (ndarray):
                A centered vector of initial values for the parameter vector.
            autoThin (boolean):
                Flag for using auto-thinning. The default is True.
            essCutoff (float)
                Minimal ESS cutoff used for autothin sample calculation. The
                default is 0.
        """

        print(u'\u2500' * 75)
        print(' Running SBC analysis', end="", flush=True)

        # Consistency checks here - make sure we have everything
        if self.testSamples is None:
            print(' - Error, no parameter samples provided')
            print(u'\u2500' * 75)
            return

        if self.testData is None:
            print(' - Error, no test data provided for samples')
            print(u'\u2500' * 75)
            return

        if self.posteriorSampler is None:
            print(' - Error, no posterior sampler provided')
            print(u'\u2500' * 75)
            return

        if self.hist is not None:
            print(' - Error, already run & histogram exists')
            print(u'\u2500' * 75)
            return

        # Run analysis if all checks passed
        L = N-burn
        for ind in range(self.testData.shape[-1]):

            print('Sample: {:d} of {:d}'.format(ind+1,
                                                self.numSamples))
            # Setup sampler
            data = self.testData[:,:,ind]
            testSample = self.testSamples[ind]

            self.posteriorSampler.setup(data, init)

            # Get a first sample and determine ESS
            posteriorSamples = self.posteriorSampler.run(N, burn)
            sampleESS = self.posteriorSampler.minESS(posteriorSamples)
            print('Minimal sample ESS: {:.2f}'.format(sampleESS))
            L_2 = posteriorSamples.shape[0]

            # If auto thin active and ESS insufficient, retake samples
            if autoThin:
                while sampleESS < 0.95*L:

                    # Adjust sample ESS if below specified cutoff
                    adjustedSampleESS = max(sampleESS,essCutoff)
                    thinRatio = L/adjustedSampleESS - 1

                    addSamples = self.posteriorSampler.run(
                                        np.ceil(L_2*thinRatio).astype(int))
                    posteriorSamples = np.concatenate((posteriorSamples,
                                                      addSamples),
                                                      axis = 0)
                    sampleESS = self.posteriorSampler.minESS(posteriorSamples)
                    print('Minimal sample ESS: {:.2f}'.format(sampleESS))
                    L_2 = posteriorSamples.shape[0]

            # Initialise histogram on first run
            if self.hist is None:
                self.histBins = np.arange(self.numParam)
                self.hist = np.zeros([L + 1, self.numParam])

            # Augment histogram and MC collection of posterior samples
            rankStatsRaw = sum(posteriorSamples < testSample)
            rankStats = np.floor(rankStatsRaw * L/L_2).astype(int)
            self.hist[rankStats,self.histBins]+=1
            self.posteriorSamplesMC.append(posteriorSamples)
            self.posteriorSamplesESS.append(sampleESS)

        print(' SBC analysis complete')
        print(u'\u2500' * 75)
