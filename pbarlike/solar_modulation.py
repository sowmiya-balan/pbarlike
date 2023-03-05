#%% Imports
from pbarlike import m_p
import numpy as np
from iminuit import Minuit
from pbarlike.likelihoods import LogLikes

#%% DRN class

class ForceFieldApprox:
    """
    .. _ffapprox:
    
    Contains methods to help calculate the solar modulated antiproton flux measured at the 
    top of the atmosphere (TOA), which depends on solar modulation parameters.

    Note:
        - Always add new classes with same attributes for compatibility with rest of the code.
        - Solar modulation parameters are profiled over when using user-defined propagation parameters; and 
          marginalized over when using the multinest sample. 
    """
    def __init__(self,data, propagation_config, drn,verbose=False):
        """
        Arguments:
            propagation_config (object): Object of class :ref:`Propagation <propagation>`
            drn (object): Object of class :ref:`DRNet <drn>`
        """
        self.verbose = verbose
        self.E = data.energy
        self.E_drn = drn.E_drn
        self.prop_model = propagation_config.model
        self.mns = propagation_config.mns
        chi_squares = LogLikes(data,propagation_config)
        if drn.marginalization:
            self.solar_modulation_uc = self.solar_modulation_margdparams
            self.solar_modulation_c = self.solar_modulation_margdparams
        else:
            self.nuisance_uc_estimation = self.nuisance_estimation(chi_squares.chi2_uncorr)
            self.nuisance_c_estimation = self.nuisance_estimation(chi_squares.chi2_corr)
            self.solar_modulation_uc = self.solmod_uc_profiledparams
            self.solar_modulation_c = self.solmod_c_profiledparams
        self.phi_CR_LIS = drn.phi_CR_LIS
        self.phi_CR_uncorr = self.solar_modulation_uc(self.phi_CR_LIS) 
        if self.verbose: print('\n Solar modulated CR flux (sol mod params using uncorr chi2): ',self.phi_CR_uncorr)
        self.phi_CR_corr = self.solar_modulation_c(self.phi_CR_LIS) 
        if self.verbose: print('\n Solar modulated CR flux (sol mod params using corr chi2): ',self.phi_CR_corr)
        self.solmod_options = {'uncorrelated': self.solar_modulation_uc, 'correlated': self.solar_modulation_c}
    
    def solar_mod(self,phi_LIS, V, Z=-1., A=1., m=m_p ):
        """
        Modulates the given CR flux depending on the given solar modulation potential, charge and mass
        of the incoming nuclei; defaults are set for antiproton nuclei. 

        Arguments:
            phi_LIS (array): Antiproton flux at the local interstellar (LIS) region.
            V (float): Solar modulation potential
            Z (float): Atomic number of CR nuclei
            A (float): Mass number of CR nuclei
            m (float): Mass of CR nuclei
        
        Returns:
            array: Solar modulated CR flux
        """
        # E_LIS_ams(58,) - array of KE per nucleon values at LIS which after solar modulation reduce to E_ams
        E_LIS_data = self.E + np.abs(Z)/A * V
        # phi_LIS_interp - (n,58) array of flux values interpolated to the above E values.
        phi_LIS_interp = np.exp(np.interp(np.log(E_LIS_data),np.log(self.E_drn),np.log(phi_LIS)))
        # phi_earth(n,58) -  flux after solar modulation
        phi_earth = phi_LIS_interp * (self.E**2 + 2*self.E*m)/(E_LIS_data**2 + 2*E_LIS_data*m)
        return phi_earth

    def nuisance_estimation(self,chi2):
        """
        Finds solar modulation parameters that minimize the given chi-squared function.
        """
        def profiling(phi_LIS):
            def lsq(V):
                # V - solar modulation potential
                phi_pred = self.solar_mod(phi_LIS, V )
                chi2_temp = chi2(phi_pred)
                return chi2_temp
        
            profiling = Minuit.from_array_func(fcn   = lsq ,
                                                start = np.array([0.6]), 
                                                error = np.array([0.001]), 
                                                limit = np.array([[0.2, 0.9]]), 
                                                errordef=1) 
            
            profiling.migrad()
            # V_profiled,A-profiled - values of estimated parameters
            V_profiled = profiling.np_values()
            return self.solar_mod(phi_LIS, V_profiled)
        return profiling

    def solmod_uc_profiledparams(self, phi_LIS):
        return np.array([self.nuisance_uc_estimation(phi_LIS[i]) for i in range(len(phi_LIS))])
    
    def solmod_c_profiledparams(self, phi_LIS):
        return np.array([self.nuisance_c_estimation(phi_LIS[i]) for i in range(len(phi_LIS))])

    def solar_modulation_margdparams(self, phi_LIS):
        """
        Calculates solar modulated flux for solar modulation parameters contained in the multinest sample.
        """
        phi = {'DIFF.BRK': 12, 'INJ.BRK': 14}
        delta_phi_bar = {'DIFF.BRK': 13, 'INJ.BRK': 15}
        V = self.mns[:,phi[self.prop_model]] + self.mns[:,delta_phi_bar[self.prop_model]]
        if self.verbose: print('\n Solar modulation potential: ',V)
        return np.array([self.solar_mod(phi_LIS[i],V[i]) for i in range(len(phi_LIS))])

    def TOA_sim(self, phi_DM_LIS,errors='correlated'):
        """
        Calculates primary and secondary antiproton flux after solar modulation according to the force field approximation.
        """
        phi_LIS = self.phi_CR_LIS + phi_DM_LIS
        if self.verbose: print('\n phi_LIS: ',phi_LIS)
        phi_DMCR = np.array([self.solmod_options[errors](phi_LIS[i]) for i in range(len(phi_LIS))])
        if self.verbose: print('\n Flux after solar modulation, phi_DMCR: ',phi_DMCR)
        return phi_DMCR
