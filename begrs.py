# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 08:48:12 2021

@author: sb636
"""

import os
import sys
import numpy as np
import torch
import gpytorch
from tqdm import tqdm
from warnings import filterwarnings

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.lazy import delazify, TriangularLazyTensor, DiagLazyTensor
from gpytorch.lazy import SumLazyTensor, MatmulLazyTensor, KroneckerProductLazyTensor

#------------------------------------------------------------------------------
# Cholesky factorisation utility
def cholesky_factor(induc_induc_covar):

    L = psd_safe_cholesky(delazify(induc_induc_covar).double())

    return TriangularLazyTensor(L)

# Filter user warnings: catch CUDA initialisation warning if no GPU is present
filterwarnings("ignore", category = UserWarning)
#------------------------------------------------------------------------------
# Main classes
class begrsGPModel(gpytorch.models.ApproximateGP):

    # Multitask Gaussian process, with the batch dimension serving to store
    # the multiple tasks.

    def __init__(self, num_vars, num_param, num_latents, num_inducing_pts):

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

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class begrs:

    def __init__(self, useGPU = True):

        # Initialise empty fields
        self.model = None
        self.likelihood = None
        self.parameter_range = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
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

        mean = (self.parameter_range[:,0] + self.parameter_range[:,1])/2
        stdDev = (self.parameter_range[:,1] - self.parameter_range[:,0])/np.sqrt(12)

        sampleCntrd = (sample - mean)/stdDev

        return sampleCntrd


    def uncenter(self, sampleCntrd):

        mean = (self.parameter_range[:,0] + self.parameter_range[:,1])/2
        stdDev = (self.parameter_range[:,1] - self.parameter_range[:,0])/np.sqrt(12)

        sample = mean + stdDev*sampleCntrd

        return sample


    def save(self, path):

        print(u'\u2500' * 75)
        print(' Saving model to: {:s} '.format(path), end="", flush=True)

        # Check saving directory exists
        if not os.path.exists(path):

            if self.model is not None:
                os.makedirs(path,mode=0o777)

                saveDict = {'parameter_range':self.parameter_range,
                            'trainX':self.trainX,
                            'trainY':self.trainY,
                            'testX':self.testX,
                            'testY':self.testY,
                            'losses':self.losses,
                            'KLdiv':self.KLdiv,
                            'N':self.N,
                            'num_vars':self.num_vars,
                            'num_param':self.num_param,
                            'num_latents':self.num_latents,
                            'num_inducing_pts':self.num_inducing_pts}#,
                            # 'useGPU':self.useGPU}

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
                self.testX = begrs_dict['testX']
                self.testY = begrs_dict['testY']
                self.N = begrs_dict['N']
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


    def setTrainingData(self, trainingData, doeSamples, parameter_range):

        print(u'\u2500' * 75)
        print(' Setting training data set', end="", flush = True)

        # allocate self.num_vars and self.num_param from datasets
        numSamples = doeSamples.shape[0]

        if numSamples == trainingData.shape[2] and doeSamples.shape[1] == parameter_range.shape[0]:

            self.parameter_range = parameter_range
            self.num_param = doeSamples.shape[1]
            numObs = trainingData.shape[0]
            self.num_vars = trainingData.shape[1]
            paramInds = self.num_vars + self.num_param

            samples = self.center(doeSamples)

            train_x_array = np.zeros([numSamples*(numObs-1),paramInds])
            train_y_array = np.zeros([numSamples*(numObs-1),self.num_vars])

            # Repackage training data and samples
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
            # Convert to tensors and store
            self.trainX = torch.from_numpy(train_x_array).float()
            self.trainY = torch.from_numpy(train_y_array).float()

            print(' - Done', flush = True)
            print(' N' + u'\u00B0' + ' of parameters: {:>5}'.format(self.num_param))
            print(' N' + u'\u00B0' + ' of variables: {:>5}'.format(self.num_vars))
            print(' N' + u'\u00B0' + ' of parameter samples: {:>5}'.format(numSamples))

        else:

            print(' Error, inconsistent sample dimensions')
            print(' N' + u'\u00B0' +' of parameters: ' + ' '*5 +'N' + u'\u00B0' + \
                  ' of parameter samples: ')
            print(' - in range matrix: {:>5}'.format(parameter_range.shape[0]),
                  end="", flush=True)
            print(' - in DOE matrix: {:>5}'.format(doeSamples.shape[1]))
            print(' - in training data: {:>5}'.format(trainingData.shape[2]),
                  end="", flush=True)
            print(' - in DOE matrix: {:>5}'.format(doeSamples.shape[0]),
                  end="", flush=True)


    def setTestingData(self, testingData):

        print(u'\u2500' * 75)
        print(' Setting testing data set', end="", flush=True)

        # Check consistency with number of variables
        if testingData.shape[1] == self.num_vars:

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

            print(' - Done')
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

        print(u'\u2500' * 75)
        print(' Training gaussian surrogate model')
        self.num_latents = num_latents
        self.num_inducing_pts = num_inducing_pts

        # CUDA check
        if self.haveGPU:

            print(' CUDA availabe',end="", flush=True)

            if self.useGPU:

                print(' - Using GPU', flush=True)

            else:

                print(' - Using CPU', flush=True)

        else:

            print(' CUDA not availabe - Using CPU')


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



    def logP(self, theta_base, batch_size = 40):

        # paramInds = self.num_vars + self.num_param

        theta = torch.from_numpy(theta_base[None,:]).float()
        theta.requires_grad_(True)

        # testX = self.testX.clone()
        # testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))

        if batch is False:

            logP = self.logPSingle(theta)

        else:

            logP = self.logPBatched(theta,batch_size)

        # logP.backward(retain_graph=True)
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

    def logPSingle(self, theta):

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


    def softLogPrior(self, theta, k = 20):

        sample = np.asarray(theta)

        f_x = np.exp(-k*(sample + 3**0.5))
        g_x = np.exp( k*(sample - 3**0.5))

        prior = - np.log(1 + f_x) - np.log(1 + g_x)
        grad = k*(f_x/(1+f_x) - g_x/(1+g_x))

        return (sum(prior),grad)


    def dlogP(self, theta, batch_size = 40):

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
