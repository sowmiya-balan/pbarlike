#%% Imports
import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from pbarlike.data import ams02Data
from pbarlike.propagation import Propagation
from pbarlike.drn_interface import DRNet
from pbarlike.solar_modulation import ForceFieldApprox
from pbarlike.likelihoods import LogLikes,LogLikeRatios

#%% C++ branching fractions dictionary from GAMBIT --> DRN branching fractions array
def br_fr(branching_fractions, sigma_v=1.0):
    """
    Recasts dictionary of key-value pairs of channels and branching fractions obtained from GAMBIT to 
    format accepted by DarkRayNet

    Arguments:
        branching_fractions (dictionary): Dictionary containing branching fractions for different channels obtained from gambit
    
    Returns:
        array: branching fraction of channels accepted by DarkRayNet
    """
    # To avoid 0/0 during normalization
    if sigma_v==0.:
        sigma_v=1
    # Normalizing bf 
    factorized_bf = {key: branching_fractions[key] / sigma_v
                        for key in branching_fractions.keys()}

    # DRBF - Dark Ray net Branching Fraction dictionary
    # Positions of channels in DRN input-bf-array
    DRBF = {"q qbar":0, "c cbar":1, "b bbar":2, "t tbar":3, "W+ W-":4, "Z0 Z0":5, "g g":6, "h h":7}
    bf = np.zeros((1,8))

    # Template for the DRN input-bf-array entry   -    Gambit annihilation channel : DRN annihilation channel(location in DRN input-bf-array )
    keys_to_location = {
                    "u_1 ubar_1":DRBF['q qbar'],"u_2 ubar_2":DRBF['c cbar'],"u_3 ubar_3":DRBF['t tbar'],
                    "ubar_1 u_1":DRBF['q qbar'],"ubar_2 u_2":DRBF['c cbar'],"ubar_3 u_3":DRBF['t tbar'],
                    "d_1 dbar_1":DRBF['q qbar'],"d_2 dbar_2":DRBF['q qbar'],"d_3 dbar_3":DRBF['b bbar'],
                    "dbar_1 d_1":DRBF['q qbar'],"dbar_2 d_2":DRBF['q qbar'],"dbar_3 d_3":DRBF['b bbar'],
                    "W+ W-":DRBF['W+ W-'], "W- W+":DRBF['W+ W-'],
                    "Z0 Z0":DRBF['Z0 Z0'], "g g":DRBF['g g'], 
                    "h0_1 h0_1":DRBF['h h']
                        }
    
    # For all possible Gambit annihilation channels, enter corresponding factorized_bf(if present, otherwise 0) in DRN input-bf-array in appropriate location
    for i in keys_to_location.keys() :
        bf[0,keys_to_location[i]] += factorized_bf.get(i,0)
    return bf

#%% Initializing DRN class, LIS simulation, solar modulation and delta-chi2 calculation

def DRN_initialization(propagation_parameters,prop_model='DIFF.BRK',prevent_extrapolation=False,include_low_energy=False, production_xsection_cov=True, verbose=False):
    """
    Initializes DRNet, calculates secondary antiproton flux for the given propagation model for 
    the entire multinest sample and initializes likelihoods for AMS-02 dataset; used at scan-level initialization.

    Arguments:
        propagation_parameters (array): 
            Usually dummy input from GAMBIT initialization file to make DRN default to multinest sample.
        
        prop_model (str): "DIFF.BRK" or "INJ.BRK"; default - "DIFF.BRK" 

        prevent_extrapolation(bool):  decides if DRN should be allowed to predict in parameter regions outside trained region; default-False

        include_low_energy (bool): 
                Whether or not to include low energy region affected by 
                solar modulation; default - False.

        production_xsection_cov(bool): decides if covariance arising from antiproton production cross-section uncertainties should be included; default - True

        verbose: default - False
    
    Returns:
        objects: drn,sol_mod,loglike_ratios

    Note:
        - GAMBIT currently uses pbarlike to obtain AMS-02 antiproton likelihoods by marginalizing over 
          multinest sample.
        - Propagation parameters and propagation model are set in the GAMBIT initialization file.
        - The default choice is to marginalize over the multinest sample by setting faulty propagation
          parameters in the GAMBIT initialization file.
    """
    from pbarlike import banner
    data = ams02Data(include_low_energy)
    propagation_config = Propagation(prop_model, production_xsection_cov, include_low_energy)
    drn = DRNet(propagation_config,propagation_parameters,prevent_extrapolation,verbose)
    sol_mod = ForceFieldApprox(data,propagation_config,drn,verbose)
    chi_squares = LogLikes(data,propagation_config,verbose)
    loglike_ratios = LogLikeRatios(drn,chi_squares,sol_mod,verbose)
    return [drn,sol_mod,loglike_ratios]
    
def py_pbar_logLikes(obj_list, DM_mass, brfr, sigma_v = 10**(-25.5228)):
    """
    Used at every point in GAMBIT scan to obtain AMS-02 antiproton likelihood for antiproton production 
    from DM annihilation and secondary origin.
    
    Arguments:
        drn (object): Object of class :ref:`DRNet <drn>`
        sol_mod (object): Eg. :ref:`Force field approximation <ffapprox>`
        loglike_ratios (object): Object of class :ref:`LogLikeRatios <loglikeratios>`
        DM_mass(int, float, list, or 1D array): dark matter mass in GeV 
        brfr(list or array): branching fractions to specify the DM annihilation channel; format - [q qbar, c cbar, b bbar, t tbar, W+ W-, Z0 Z0, g g, h h]
        sigma_v(int, float, list or 1D array): thermally averaged annihilation cross-section in :math:`cm^3s^{-1}`

    Returns:
        dictionary: key-value pairs of log-likelihood ratios with correlated and uncorrelated errors.
        
    """
    drn = obj_list[0]
    sol_mod = obj_list[1]
    loglike_ratios = obj_list[2]
    bf = br_fr(brfr,sigma_v)
    drn.preprocessing_DMparams(DM_mass, bf, sigma_v)
    phi_DM_LIS = drn.DM_LIS_sim()
    phi_DMCR = sol_mod.TOA_sim(phi_DM_LIS)
    del_chi2_uncorr = loglike_ratios.del_chi2_uncorr(phi_DMCR)
    del_chi2_corr = loglike_ratios.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2_uncorr[0] , 'correlated' : del_chi2_corr[0]}
    return result
 