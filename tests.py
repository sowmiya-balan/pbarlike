#%% Imports
import numpy as np
from pbarlike.data import ams02Data
from pbarlike.drn_interface import create_DIFF_BRK_parameters,create_INJ_BRK_parameters
from pbarlike.convenience_functions import loglike_recipe

#%% Inputs
m_DM = np.array([10**(5.018181818181818 - 3)]) 
bf = np.array([1.000e-05, 1.000e-05, 9.993e-01, 1.000e-05, 1.000e-05, 1.000e-05,  1.000e-05, 1.000e-05])
sv = 3e-26
pp_db_ib = [create_DIFF_BRK_parameters(), create_INJ_BRK_parameters()]
pm_options = ['DIFF.BRK','INJ.BRK']
pp = 0 # 0- DIFF.BRK; 1 - INJ.BRK
include_low_energy_arg = False
production_xsection_cov_arg = True
pe_arg = True
verbose_arg=False

#%% Results
print('\n Calling pbarlike...')

print('\n Test Simulation:1; Propagation Model: DIFF.BRK')
results = loglike_recipe(m_DM, bf, sv, pp_db_ib[pp], prop_model=pm_options[pp], \
                               Data=ams02Data,include_low_energy=include_low_energy_arg, \
                               production_xsection_cov=production_xsection_cov_arg,prevent_extrapolation= pe_arg, verbose=verbose_arg)

print('\n del_chi2 = ', results["uncorrelated"])
print('\n del_chi_cov = ', results["correlated"])

print('\n Test Simulation:2; Propagation Model: INJ.BRK')
pp = 1
results = loglike_recipe(m_DM, bf, sv, pp_db_ib[pp], prop_model=pm_options[pp],\
                               Data=ams02Data,include_low_energy=include_low_energy_arg, \
                               production_xsection_cov=production_xsection_cov_arg,prevent_extrapolation= pe_arg, verbose=verbose_arg)

print('\n del_chi2 = ', results["uncorrelated"])
print('\n del_chi_cov = ', results["correlated"])

print('\n ------------------------------------------------------------'\
      '\n pbarlike performed likelihood calculation without errors! :)'\
      '\n ------------------------------------------------------------')