#%% Imports
import numpy as np

class LogLikes:
    """
    .. _loglikes:
    
    Class with methods defining chi-squared functions using correlated and uncorrelated errors.
    """
    def __init__(self,data,propagation_config,errors='correlated',verbose=False):
        """
        Arguments:
            data (object): Data object that contains data correlations and uncorrelated errors (Eg: :ref:`AMS-02 Data <ams>`)
            propagation_config (object): Object of class :ref:`Propagation <propagation>`
        """
        self.verbose = verbose
        self.pbar_flux = data.pbar_flux
        self.start_index = data.start_index
        self.data_errors = data.errors
        if self.verbose: print('\n Inverse of data_correlations: ', self.data_errors)      
        self.cov_inv = (data.correlations+propagation_config.production_xsection_cov).I  
        if self.verbose: print('\n Inverse of data_correlations+nuclear_uncertainties covariance matrix: ',self.cov_inv)      

    def chi2_uncorr(self,phi_pred):
        """
        Chi-squared function with only uncorrelated data errors; same as the one used to sample 
        propagation parameters contained in the multinest sample. 
        """
        if phi_pred.ndim == 1:
            phi_pred = phi_pred
        diff = self.pbar_flux - phi_pred
        return np.sum((diff)**2 / self.data_errors**2,axis = -1)

    def chi2_corr(self,phi_pred):
        """
        Chi-squared function including data covariance 
        """
        if phi_pred.ndim == 1:
            phi_pred = phi_pred[np.newaxis,:]
        diff = self.pbar_flux - phi_pred
        return np.diag((diff) @ self.cov_inv @ (diff).T)


class LogLikeRatios:
    """
    .. _loglikeratios:
    
    Class with methods defining log-likelihood ratios using correlated and uncorrelated errors.
    """
    def __init__(self,drn,chi_squares,sol_mod,verbose=False):
        """
        Arguments:
            drn (object): Object of class :ref:`DRNet <drn>`
            chi_squares (object): Object of class :ref:`LogLikes <loglikes>`
            sol_mod (object): Eg. :ref:`Force field approximation <ffapprox>`
        """
        self.verbose = verbose
        self.chi2_uncorr = chi_squares.chi2_uncorr
        self.chi2_corr = chi_squares.chi2_corr
        self.chi2_CR_uncorr = self.chi2_uncorr(sol_mod.phi_CR_uncorr)
        if self.verbose: print('\n Uncorrelated chi2 for CR only case: ',self.chi2_CR_uncorr)
        self.chi2_CR_corr = self.chi2_corr(sol_mod.phi_CR_corr)
        if self.verbose: print('\n Correlated chi2 for CR only case: ',self.chi2_CR_corr)
        self.chi2_CR_diff = np.clip(self.chi2_CR_uncorr - self.chi2_CR_corr,  -500,500)
        self.CR_marginalized_likelihood = np.sum(np.exp(self.chi2_CR_diff/2),axis=-1)
        if self.verbose: print('\n Marginalized chi2 for CR only case: ',self.CR_marginalized_likelihood)
        self.pp_ur = drn.pp_ur

    def del_chi2_uncorr(self, phi_DMCR):
        """
        Calculates difference of marginalized chi-squared values between cases of with and without DM;
        using the only uncorrelated errors.
        """
        del_chi2 = []
        for i in range(len(phi_DMCR)):
            chi2_DMCR = self.chi2_uncorr(phi_DMCR[i])
            if self.verbose: print('\n Uncorrelated chi2 for DMCR: ',chi2_DMCR)
            chi2_diff = np.clip(self.chi2_CR_uncorr - chi2_DMCR,  -500,500)
            del_chi2_t = -2*np.log( 1/len(self.pp_ur) * np.sum(np.exp(chi2_diff/2),axis=-1) )
            del_chi2.append(del_chi2_t)
        return del_chi2

    def del_chi2_corr(self, phi_DMCR):
        """
        Calculates difference of marginalized chi-squared values between cases of with and without DM;
        using the total covariance matrix including correlated and uncorrelated errors.
        """
        delta_chi2_cov = []
        for i in range(len(phi_DMCR)):
            chi2_DMCR = self.chi2_corr(phi_DMCR[i])
            if self.verbose: print('\n Correlated chi2 for DMCR: ',chi2_DMCR)
            chi2_DM_diff = np.clip(self.chi2_CR_uncorr - chi2_DMCR,  -500,500)
            delta_chi2_t = -2*np.log(np.sum(np.exp(chi2_DM_diff/2),axis=-1) / self.CR_marginalized_likelihood)
            delta_chi2_cov.append(delta_chi2_t)
        return delta_chi2_cov


