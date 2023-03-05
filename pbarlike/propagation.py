from pbarlike import script_directory
import numpy as np

class Propagation:
    """
    Class to configure propagation model, load marginalization sample and 
    antiproton production cross-section uncertainties.
    """
    def __init__(self, prop_model="DIFF.BRK", production_xsection_cov=True, include_low_energy=False):
        """
        Arguments:
            prop_model (str): "DIFF.BRK" or "INJ.BRK"; default - "DIFF.BRK" 
            production_xsection_cov (bool): 
                Whether or not to include antiproton production cross-section uncertainties.
            include_low_energy (bool): 
                Whether or not to include low energy region affected by 
                solar modulation; default - False.

        Note:
            - Antiproton production cross-section uncertainties are included as data correlations.
              This covariance matrix is obtained by propagating cross-section uncertainties through 
              antiproton flux simulations and calculating the sample covariance of antiproton flux.
              For more details, see section 3.1 in :ref:`[4] <4>`. 
            - The sample of propagation parameters is obtained from a multinest fit to AMS-02 and Voyager data.
              This sample will be used for marginalization over the propagation parameters and is henceforth
              referred to as the *multinest sample*. For more details, see section 3.2 in :ref:`[4] <4>`.
        """
        model_options = ['DIFF.BRK', 'INJ.BRK']
        if prop_model in model_options:
            self.model = prop_model
        else:
            print('\n The propagation model "%s" is not a part of pbarlike. \
            DRN cannot emulate antiproton flux for custom models. To include this model, \
            extend pbarlike by including a simulator for the new model. \
            For the current run, the model will be set to default (DIFF.BRK). '%prop_model)
            self.model = 'DIFF.BRK'

        self.dep_path = script_directory + '/dependencies/' + self.model + '/'
        self.load_pxs_cov_mat(production_xsection_cov, include_low_energy)
        self.load_pp_data()
    
    def load_pxs_cov_mat(self, production_xsection_cov=True, include_low_energy=False):
        # Coviariance matrix arising from production cross-section uncertainties, given in AMS-02 energy bins 
        start_index = 0 if include_low_energy else 14
        if production_xsection_cov:
            self.production_xsection_cov = np.matrix(np.genfromtxt(self.dep_path+'CovMatrix_AMS02_pbar_CrossSection.txt'))[start_index:,start_index:]
        elif not production_xsection_cov :
            self.production_xsection_cov = np.matrix(np.zeros((58-start_index,58-start_index)))
        else :
            pass 

    def load_pp_data(self):
        # Setting length of propagation parameters according to diffusion model
        N_pp = {'DIFF.BRK': 10, 'INJ.BRK': 12}
        self.N_pp = N_pp[self.model]
        # Defining names of parameters
        self.param_names = {'DIFF.BRK' : ['gamma_2,p', 'gamma_2', 'D0', 'delta_l', 'delta', 'delta_h - delta', 'R_D,0', 's_D', 'R_D,1', 'v_0,c'],
                            'INJ.BRK': ['gamma_1,p', 'gamma_1', 'R_0', 's', 'gamma_2,p', 'gamma_2', 'D_0', 'delta', 'delta_h - delta', 'R_1,D', 'v_0', 'v_A']} 
        # MultiNest sample for propagation parameters using AMS-02 7y data+Galprop simulation+purely secondary antiprotons
        self.mns = (np.genfromtxt(self.dep_path + 'multinest_sample.dat'))
        self.mnpp = self.mns[:,:self.N_pp]