from pbarlike import script_directory, m_p
import numpy as np

def R_to_Ekin(R,z = -1, A = 1, m = m_p ):
    """
    Converts R in GV to kinetic energy per nucleon in GeV, 

    .. math::
        E_k / A = \sqrt{(RZ/A)^2 + m^2}
    """
    Z = np.abs(z)
    return np.sqrt(R**2 * (Z/A)**2 + m**2) - m

def flux_in_Ekin(flux_in_R,R,z=-1,A=1,m=m_p):
    """
    Converting flux :math:`(m^2 \ sr \ s \ GV)^{-1}` in R to flux :math:`(m^2 \ sr \ s \ GeV)^{-1}` in E,

    .. math::
        \\frac{d \phi}{d R} \\frac{d R}{d (E_k/A)} = \\frac{d \phi}{d R} \\frac{1}{R} (\\frac{A}{Z})^2 \sqrt{( \\frac{RA}{Z})^2 + m^2}
    """
    Z = np.abs(z)
    return flux_in_R /R * (A/Z)**2 * np.sqrt((R*Z/A)**2 + m**2)

class ams02Data:
    """
    .. _ams:
    
    Class to load and process AMS-02 7-year dataset. 
    
    Note:
        - Always add new classes with same attributes for compatibility with rest of the code.
        - Measurements below :math:`10 \ GeV` are neglected by default, due to uncertainties 
          induced by approximate modeling of solar modulation.
        - AMS-02 data correlations have not been made public by the AMS collaboration. The data correlations
          included here were modeled in :ref:`[5] <5>`. 

    Attributes:
        pbar_flux (1D array): Antiproton flux values in :math:`(m^2 \ sr \ s \ GeV)^{-1}`
        energy (1D array): :math:`GeV` energies at which antiproton flux values are measured.
        errors (1D array): Uncorrelated data errors.
        correlations (2D array): Data correlations.
    """
    def __init__(self, include_low_energy = False):
        """
        Arguments:
            include_low_energy (bool): 
                Whether or not to include low energy region affected by 
                solar modulation; default - False.
        """                                

        # 7 year ams02_data (for column description, see pbar_ams02_7y.txt file)
        ams_data = np.genfromtxt(script_directory+'/dependencies/pbar_ams02_7y.txt')
        R_ams = (ams_data[:,0]+ams_data[:,1])/2 
        phi_in_R = ams_data[:,10] * ams_data[:,13]
        error_R_ams = np.sqrt(ams_data[:,11]**2+ams_data[:,12]**2)*ams_data[:,13]

        # E_ams -(58,) array of KE per nucleon values at which AMS02 flux measurements are recorded for pbar
        E_ams = np.array(R_to_Ekin(R_ams))
        # phi_ams - (58,) array of flux values measured at E_ams
        phi_ams = np.array(flux_in_Ekin(phi_in_R,R_ams))
        # error_ams - (58,) array of error values of flux values measured at E_ams
        error_ams = np.array(flux_in_Ekin(error_R_ams,R_ams))
        # Estimated ams02 covariance matrix (58,58) for the 7-year antiproton data:
        ams_7y_cov = np.matrix(np.genfromtxt(script_directory+'/dependencies/CovMatrix_AMS02_pbar_7y.txt'))

        self.start_index = 0 if include_low_energy else 14

        self.energy = E_ams[self.start_index:]
        self.pbar_flux = phi_ams[self.start_index:]
        self.errors = error_ams[self.start_index:]
        self.correlations = ams_7y_cov[self.start_index:,self.start_index:]
        # self.correlations = np.matrix(np.diag(self.errors**2))

    

