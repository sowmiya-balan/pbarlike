#%% Imports
import numpy as np
from pbarlike.data import ams02Data
from pbarlike.propagation import Propagation
from pbarlike.drn_interface import DRNet
from pbarlike.solar_modulation import ForceFieldApprox
from pbarlike.likelihoods import LogLikes,LogLikeRatios

def loglike_recipe(DM_mass, bf, sigma_v, propagation_parameters,prop_model='DIFF.BRK', Data=ams02Data,include_low_energy=False, production_xsection_cov=True,prevent_extrapolation= False, verbose=False):
    """
    Calculates difference of marginalized chi-squared value between cases of with and without DM.

    Arguments:
        DM_mass(int, float, list, or 1D array): dark matter mass in GeV 

        brfr(list or array): branching fractions to specify the DM annihilation channel; format - [q qbar, c cbar, b bbar, t tbar, W+ W-, Z0 Z0, g g, h h]

        sigma_v(int, float, list or 1D array): thermally averaged annihilation cross-section in :math:`cm^3s^{-1}`

        propagation_parameters(list or array): propagation parameters (for format and ranges see DRN.load_pp_data()); for marginalization, pass faulty propagation parameters (Eg: [0.])

        propagation_model(str): "DIFF.BRK" or "INJ.BRK"

        prevent_extrapolation(bool):  decides if DRN should be allowed to predict in parameter regions outside trained region; default-False

        data(1D array): antiproton flux measurements in :math:(m^2 sr s GeV)^{-1} at energies E; default - 7 year AMS-02 data

        E(1D array): kinetic energy per nucleon values in GeV at which antiproton measurements are given; default - E_ams

        errors(1D array): statistical errors at corresponding kinetic energy per nucleon values; default - errors_ams
        
        data_cov(2D array): systematic errors at corresponding kinetic energy per nucleon values; default - ams_7y_cov

        xsection_cov(bool): decides if covariance arising from antiproton production cross-section uncertainties should be included; default - True

        verbose: default - False

    Returns:
        dictionary: marginalized chi-squared differences using correlated and uncorrelated errors
    """
    data = Data(include_low_energy)
    propagation_config = Propagation(prop_model, production_xsection_cov, include_low_energy)
    drn = DRNet(propagation_config,propagation_parameters,prevent_extrapolation,verbose)
    sol_mod = ForceFieldApprox(data, propagation_config,drn,verbose)
    chi_squares = LogLikes(data,propagation_config,verbose)
    loglike_ratios = LogLikeRatios(drn,chi_squares,sol_mod,verbose)
    drn.preprocessing_DMparams(DM_mass, bf, sigma_v)
    phi_DM_LIS = drn.DM_LIS_sim()
    phi_DMCR_uc = sol_mod.TOA_sim(phi_DM_LIS,'uncorrelated')
    phi_DMCR_c = sol_mod.TOA_sim(phi_DM_LIS,'correlated')
    del_chi2_uncorr = loglike_ratios.del_chi2_uncorr(phi_DMCR_uc)
    del_chi2_corr = loglike_ratios.del_chi2_corr(phi_DMCR_c)
    result = {'uncorrelated' : del_chi2_uncorr , 'correlated' : del_chi2_corr}
    return result
