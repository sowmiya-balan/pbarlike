#%% Imports
from pbarlike import script_directory
import gc
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#%% DRN class

def create_INJ_BRK_parameters(gamma_1p = 1.72, gamma_1 = 1.73, R_0 = 6.43e3, s = 0.33, gamma_2p = 2.45, gamma_2 = 2.39, D_0 = 4.1e28, delta = 0.372, delta_h_delta = -0.09, R_1D = 2.34e5, v_0c = 0.64, v_A = 20.4):
    """
    Gives a set of propagation parameters obtained from a fit of proton and helium fluxes
    to AMS-02 and Voyager data, for the INJ.BRK model.
    """
    propagation_parameters = np.array([gamma_1p, gamma_1, R_0, s, gamma_2p, gamma_2, D_0, delta, delta_h_delta, R_1D, v_0c, v_A])
    return propagation_parameters

def create_DIFF_BRK_parameters(gamma_2p = 2.34, gamma_2 = 2.28, D0 = 3.78e28, delta_l = -0.66, delta = 0.52, delta_h_delta = -0.16, R_D0 = 3910, s_D = 0.41, R_D1 = 2.22e5, v_0c = 1.91):
    """
    Gives a set of propagation parameters obtained from a fit of proton and helium fluxes
    to AMS-02 and Voyager data, for the DIFF.BRK model.
    """
    propagation_parameters = np.array([gamma_2p, gamma_2, D0, delta_l, delta, delta_h_delta, R_D0, s_D, R_D1, v_0c])
    return propagation_parameters


class DRNet:
    """
    Contains methods that transform the user inputs (propagation parameters,
    DM mass, branching fractions) as required by DMNet (simulates primary antiproton flux) and sNet 
    (simulates secondary antiproton flux) and methods that postprocess network ouputs to user usable forms.
    """
    def __init__(self,propagation_config,propagation_parameters,prevent_extrapolation=True,verbose=False):
        """
        Arguments: 
            propagation_config (object): Object of class :ref:`Propagation <propagation>`
            propagation_parameters (list or array): propagation parameters
            prevent_extrapolation (bool): Whether network should be allowed to predict in untrained parameter regions; default - True
        """
        print("\n Initializing DRN ...")
        self.verbose = verbose
        self.pe = prevent_extrapolation
        self.dep_path = propagation_config.dep_path
        self.load_deps()
        self.load_pp_data(propagation_config)
        self.preprocessing_prop_params(propagation_parameters)
        self.phi_CR_LIS = self.CR_LIS_sim()
        print('\n The simulation tool has been initiated. ')

    def load_deps(self):
        """
        Loading propagation parameter transformations for primary and secondary antiprotons
        """
        # DM antiprotons    
        self.DM_trafos = np.load(self.dep_path + 'DM_trafos_x.npy', allow_pickle = True)
        # Secondary antiprotons        
        self.S_trafos = np.load(self.dep_path + 'S_trafos.npy', allow_pickle = True)
        self.E_drn = np.array(np.load(script_directory+'/dependencies/E.npy'))

    def load_pp_data(self, propagation_config):
        """
        .. _ppformat:

        Loads depth, names, priors and multinest samples for propagation parameters.
        """
        self.dep_path = propagation_config.dep_path
        self.propagation_model = propagation_config.model
        self.N_pp = propagation_config.N_pp
        self.param_names = propagation_config.param_names
        self.mnpp = propagation_config.mnpp
        # Loading priors
        mins_pp = {'DIFF.BRK' : [2.249, 2.194, 3.411e+28, -9.66635e-01, 4.794e-01, -2.000e-01, 3.044e+03, 3.127e-01, 1.217e+0, -1e-5],
                   'INJ.BRK': [1.59786, 1.60102, 4939.44, 0.221776, 2.41369, 2.36049,3.53e+28, 0.301255, -0.171395, 125612, -1e-5, 14.3322]}
        maxs_pp = {'DIFF.BRK' : [2.37e+00, 2.314e+00, 4.454e+28, -3.677e-01, 6.048e-01, -8.330e-02, 4.928e+03, 5.142e-01, 3.154e+05, 1.447e+01],
                   'INJ.BRK': [1.84643, 1.84721, 8765.77, 0.45543, 2.49947, 2.44248, 5.49E+28, 0.41704, -0.0398135, 413544, 8.61201, 29.206]}
        self.mins_pp = mins_pp[self.propagation_model]
        self.maxs_pp = maxs_pp[self.propagation_model]

    def preprocessing_prop_params(self, propagation_parameters):
        """
        Checks validity of given propagation parameters and defaults to multinest sample if conditions are not met.
        """
        # Preparing propagation parameters. Required transformations will be done within simulations (DM_sim, CR_sim)
        if type(propagation_parameters) == list :
            propagation_parameters = np.array(propagation_parameters)[np.newaxis,:]
        if propagation_parameters.ndim == 1:
            propagation_parameters = propagation_parameters[np.newaxis,:]
        if propagation_parameters.all == 0 : 
            self.pp = self.mnpp   
            self.marginalization = True  
        elif propagation_parameters.shape[1] != self.N_pp :
            print('\n The number of propagation parameters is not consistent with the propagation model. The default multinest sample will be used for marginalization.')
            self.pp = self.mnpp  
            self.marginalization = True 
        # Inputs that are outside regions in which the network is trained can cause ANNs to perform unreliably. If extrapolation is allowed, code will proceed with given
        # inputs, else multinest sample is used.
        elif self.pe:
            for i in range(self.N_pp):
                if np.min(propagation_parameters[:, i]) <= self.mins_pp[i] or np.max(propagation_parameters[:, i]) >= self.maxs_pp[i]:
                    print("\n At least one of the inputs for %s is outside the trained parameter ranges. The default multinest sample will be used for marginalization. " % (self.param_names[self.propagation_model])[i])
                    print(np.min(propagation_parameters[:, i]),np.max(propagation_parameters[:, i]))
                    self.pp = self.mnpp
                    self.marginalization = True
                    break
                else :
                    self.pp = propagation_parameters
                    self.marginalization = False
        else :
            self.pp = propagation_parameters
            self.marginalization = False   
        self.pp_ur = self.pp
        if self.verbose: print('\n Shape of propagation parameters: ', np.shape(self.pp))  

    def preprocessing_DMparams(self, DM_mass, br_fr, sigma_v):
        """
        Preprocesses DM masses, annihliation cross-sections and branching fractions as required by the DMNet.
        """
        # Preparing DM masses
        if type(DM_mass) == float or type(DM_mass)== np.float64 or type(DM_mass) == np.int64:
            DM_mass = np.array([DM_mass])
        elif type(DM_mass) == list :
            DM_mass = np.array(DM_mass)
        self.DM_mass = DM_mass
        if self.verbose: print('\n Length of given DM mass array: ',len(self.DM_mass))
        self.DM_mass_r = np.repeat(DM_mass,len(self.pp),axis=0)
        # Min-max standardization of DM masses
        m_DM = ((np.log10(DM_mass)+3) - np.log10(5e3)) / (np.log10(5e6)-np.log10(5e3))
        # Repeating mass for the given number of propagation parameter points since DMNet only accepts list of arrays of equal length.
        self.m_DM = np.repeat(m_DM,len(self.pp),axis=0)

        # Preparing branching fractions
        if type(br_fr)==list:
            br_fr = np.array([br_fr])
        if br_fr.ndim == 1:
            br_fr = br_fr[np.newaxis,:] 
        # Replacing zeros by 1e-5 and renormalizing
        masked_array = np.where(br_fr < 1e-5, 0, 1) # ones for every fs >= 1e-5
        masked_reversed = np.ones_like(masked_array) - masked_array # ones for every fs < 1e-5
        masked_rf = masked_array * br_fr # array with entries only >= 1e-5, else 0
        norm = np.sum(masked_rf, axis = -1)
        if norm==0.:
            norm=1
        scaling = (1-np.sum(masked_reversed, axis = -1)*1e-5)/norm # scaling for each >=1e-5 fs, while keeping relative fractions and normalizations
        bf_temp = masked_rf * scaling[:,None] + masked_reversed*1e-5 # scale fs >=1e-5 and set other to 1e-5
        # Preprocessing braching fractions 
        bf = (np.log10(bf_temp) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0])) 
        bf_r = np.repeat(bf,len(self.pp),axis=0)
        self.bf = bf_r

        # Repeating propagation parameters and branching fractions if necessary
        if len(self.DM_mass)>1:
            self.pp = np.tile(self.pp,(len(self.DM_mass),1))
            self.bf = np.repeat(bf_r,len(self.DM_mass),axis=0)
        
        # Preventing extrapolation
        self.stop_sim = False
        stop_DM = False
        if np.min(self.DM_mass) < 5 or np.max(self.DM_mass) > 5e3:
            print('\n At least one of the given DM masses is outside of the provided range (5 GeV to 5 TeV). DM antiproton flux cannot be predicted.')
            stop_DM = True
        if np.min(bf) < 1e-5 or np.max(bf) > 1 :
            print('Given branching fractions: ', bf)
            print('\n The given branching fractions were not in the range of trained parameters or not normalized to one. Values below 1e-5 were mapped to 1e-5 and the remaining fractions normalized accordingly.')
            stop_DM = True 
        if stop_DM:
            self.stop_sim = True
            
        # Preparing thermally averaged annihilation cross-section (sigma_v)
        if type(sigma_v) == float or type(sigma_v)== np.float64 or type(sigma_v) == np.int64:
            sigma_v = np.array([sigma_v])
        if type(sigma_v) == list:
            sigma_v = np.array(sigma_v)
        self.sv = sigma_v
        
    def DM_sim(self): 
        # Transforming propagation parameters
        pp = ((self.pp - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0,1]))
        # x - (n,40) array of x values at which network predicts output ; x = E/m_DM
        min_x = -0.1 # Necessary for model without reacceleration (DM antiproton spectra diverge for E -> m_DM, x -> 0)
        x_temp = 10**(np.linspace(-3.7, min_x, 40))
        x = np.repeat([x_temp],len(self.DM_mass_r),axis=0)
        # E_dmnet - (n,40) array of kinetic energy(KE) per nucleon values at which network predicts output
        E_dmnet = x*self.DM_mass_r[:,None]
        # y_DM_x - (n,40) array of y values predicted by the DMNet for different x values; y(x) = log10(m^3 x phi(E))        
        DM_model = tf.keras.models.load_model(self.dep_path + 'DM_model_x.h5')

        if self.verbose: print('\n Preprocessed mass: ',self.m_DM)
        if self.verbose: print('\n Preprocessed branching fractions: ',self.bf)
        if self.verbose: print('\n Preprocessed prop params for DMNet: ',pp)

        y_DM_x = DM_model([self.m_DM,self.bf,pp])

        if self.verbose: print('\n DMNet Ouput: ',y_DM_x)

        # Releasing memory
        tf.keras.backend.clear_session()
        del DM_model    
        gc.collect()
        # phi_dmnet - (n,40) array of flux values predicted by the DMNet for different KE per nucleon values
        phi_dmnet = 10**(y_DM_x) / (self.DM_mass_r[:,None]**3 * x)
        # phi_DM_LIS - (n,28) array of flux values interpolated to obtain fluxes at the same KE per nucleon values as in E_drn so that it can
        # be added to phi_CR_LIS. Only after solar modulation, the flux is to be interpolated to E_ams values.
        phi_DM_LIS = np.zeros((len(self.DM_mass_r),len(self.E_drn)))
        # Flux predicted by DMNet is only reliable in the allowed x range, i.e. for only those KE per nucleon in E_drn 
        # which for a given DM mass fall within the allowed x*m values. Thus we make a list of these allowed E_drn values, interpolate
        # the flux only at these values and set all other flux values at other E_drn to zero.
        for i in range(len(self.DM_mass_r)):
            E_drn_allowed = []
            indices = []
            for j in range(len(self.E_drn)):
                if self.E_drn[j]/self.DM_mass_r[i] >= 10**(-3.7) and self.E_drn[j]/self.DM_mass_r[i] <= 1:
                    E_drn_allowed.append(self.E_drn[j])
                    indices.append(j)
            phi_DM_LIS[i,indices]  = np.exp(np.interp(np.log(E_drn_allowed), np.log(E_dmnet[i]), np.log(phi_dmnet[i])))
        if self.verbose: print('\n Interpolated DMNet Output: ',phi_DM_LIS)
        return phi_DM_LIS 

    def CR_LIS_sim(self):  
        # Preprocessing propagation parameters
        pp = ((self.pp - np.array(self.S_trafos[0]))/np.array(self.S_trafos[1]))
        if self.verbose: print('Preprocessed prop params for SNet: ',pp)
        # y_CR - (28,) array of y values predicted by the sNet at different KE per nucleon values in self.E_drn ; y(E) = log10(E^2.7 phi(E))
        S_model =tf.keras.models.load_model(self.dep_path + 'S_model.h5')
        y_CR = S_model(pp)
        if self.verbose: print('\n SNet Output: ',y_CR)
        # Releasing memory
        tf.keras.backend.clear_session()
        del S_model
        gc.collect()
        # phi_CR_LIS - (28,) array of flux values predicted by the sNet at different KE per nucleon values in E_drn
        phi_CR_LIS = 10**(y_CR)/self.E_drn**2.7
        if self.verbose: print('\n Energy at which SNet produces output: ',self.E_drn)
        if self.verbose: print('\n phi_CR_LIS: ',phi_CR_LIS)
        return phi_CR_LIS

    def DM_LIS_sim(self):
        """
        Simulates primary and secondary antiproton fluxes at the local interstellar region using DMNet and sNet respectively.
        """
        if self.stop_sim :
            print('\n The DM antiproton flux cannot be predicted by DRN due to atleast one parameter outside the region in which the network is trained.')
            phi_DM_LIS = None
        else:
            DRN_output = self.DM_sim()
            phi_DM_LIS = np.array([self.sv[i]/10**(-25.5228)*DRN_output for i in range(len(self.sv))])
            phi_DM_LIS = np.reshape(phi_DM_LIS,(len(self.sv)*len(self.DM_mass),len(self.pp_ur),len(self.E_drn)))
        return phi_DM_LIS
