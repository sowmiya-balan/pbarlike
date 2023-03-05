import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
script_directory = os.path.dirname(os.path.realpath(__file__))
m_p = 0.9382720881604903  # Mass of proton in GeV (938.2720881604903 MeV)

from pbarlike.convenience_functions import loglike_recipe

