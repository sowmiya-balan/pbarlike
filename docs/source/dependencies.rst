dependencies
------------

- ``pbar_ams02_7y.txt``: AMS-02 7-year antiproton data; description of columns included in same file.

- ``E.npy``: Energies at which sNet makes anitproton flux predictions.

- ``CovMatrix_AMS02_pbar_7y.txt``: Data correlations in AMS-02 antiproton data in the same energy
  bins as the antiproton flux measurements, as modeled in :ref:`[5] <5>`.

- ``<propagation_model>/DM_model_x.h5``: DMNet for ``<propagation_model>``

- ``<propagation_model>/S_model.h5``: sNet for ``<propagation_model>``

- ``<propagation_model>/CovMatrix_AMS02_pbar_CrossSection.txt``: Covariance matrix from antiproton production cross-section uncertainties as described in section 3.1 of :ref:`[4] <4>`.

- ``<propagation_model>/multinest_sample.dat``: 
        Posterior sample of propagation parameters from fit to AMS-02 and Voyager data, see section 3.2 of :ref:`[4] <4>`. For description of CR parameters, refer DRN `documentation <https://github.com/kathrinnp/DarkRayNet>`_.
        
        DIFF.BRK:

        ============  =======================  ======================================  
        Column no.    DRN parameter name       Description
        ============  =======================  ======================================   
        0             gamma 2,p	               Slope of injection spectrum for p	
        1             gamma_2                  Slope of injection spectrum for He	
        2             D_0                      Normalization of diffusion coefficient
        3             delta_l                  Slope below low rigidity break in diffusion coefficient	
        4             delta                    Slope between low and high rigidity breaks in diffusion coefficient	
        5             delta_h - delta          delta_h: Slope above high rigidity break in diffusion coefficient	
        6             R_D,0                    Value of low rigidity at which break occurs in diffusion coefficient  	
        7             s_D                      Smoothening parameter for low-rigidity break in diffusion coefficient	
        8             R_D,1                    Value of high rigidity at which break occurs in diffusion coefficient	
        9             v_0,c                    Constant convection velocity	
        10            A_XS,3He                 Nuisance parameter for 3He production	
        11            delta_XS3He              Nuisance parameter for 3He production	
        12            V_p                      Solar modulation potential	for p
        13            V_pbar-V_p               V_pbar: Solar modulation	potential for pbar
        14            A_p                      Overall normalization for p
        15            Abd_He                   Helium 4 isotopic abundance
        ============  =======================  ======================================	

        INJ.BRK+vA:

        ============  =======================  ======================================  
        Column no.    DRN parameter name       Description
        ============  =======================  ======================================   
        0             gamma_1,p                Slope of injection spectrum below rigidity break for p
        1             gamma_1                  Slope of injection spectrum below rigidity break
        2             R_0                      Value of rigidity at break in injection spectrum
        3             s                        Break smoothening parameter for injection spectrum
        4             gamma_2,p                Slope of injection spectrum above rigidity break for p
        5             gamma_2                  Slope of injection spectrum above rigidity break
        6             D_0                      Normalization of diffusion coefficient
        7             delta                    Slope below rigidity break in diffusion coefficient	
        8             delta_h - delta          delta_h: Slope above rigidity break in diffusion coefficient	
        9             R_1,D                    Value of rigidity at which break occurs in diffusion coefficient
        10            v_0                      Constant convection velocity
        11            v_A                      Alfven velocity
        12            A_XS,3He                 Nuisance parameter for 3He production
        13            delta_XS3He              Nuisance parameter for 3He production
        14            V_p                      Solar modulation potential for p
        15            V_pbar-V_p               V_pbar: Solar modulation	potential for pbar
        16            A_p                      Overall normalization
        17            Abd_He                   Helium 4 isotopic abundance
        ============  =======================  ======================================
