def sel_em_TPV(t_d,t_m,t_ARC,N1,N2,N3,EQE):

    from MESH import SimulationPlanar
    import numpy as np
    from InAs_TPV import InAsTPV

    #Constants
    h_bar = 1.054571817E-34
    k_b = 1.380649E-23
    c = 2.99792458E8

    T_e = 1100

    #Preprocessing
    dw = 1E12
    W = [dw*(i+1) for i in range(1999)]
    q = 0
    q_w = np.zeros(1999)
    theta = np.zeros(len(W))
    q_b_w = np.zeros(len(W))
    emissivity = np.zeros(len(W))
    eps_S = np.zeros(shape=(1999,3))
    eps_a = np.zeros(shape=(1999,3))
    eps_spacer = np.zeros(shape=(1999,3))
    eps_b = np.zeros(shape=(1999,3))
    eps_ARC = np.zeros(shape=(1999,3))
    eps_vac = np.zeros(shape=(1999,3))

    #Drude Model
    def silver(w):
        e = 1.60217663e-19
        w_p = 9.61*e/h_bar
        gamma_d = 0.079*e/h_bar
        A1 = 23.9*(e/h_bar)**2
        gamma1 = 0.84*e/h_bar
        E1 = 4.58*e/h_bar
        eps = 3.08 - w_p**2/(w**2+1j*gamma_d*w) + A1/(E1**2-(h_bar*w)**2-1j*gamma1*h_bar*w)
        return eps

    #Doped Silicon
    def dopedSi(w,T,N):
        eps_inf = 11.7

        m0 = 9.1093837e-31
        mstar = 0.37*m0
        eps0 = 8.85e-12
        e = 1.60217663e-19

        N_A = N*1e6
        phi = T/300

        mu1 = 44.9e-4
        mu_max = 470.5e-4
        mu2 = 29e-4

        C_r = 2.23e17*1e6
        C_s = 6.1e20*1e6

        alpha = 0.719
        beta = 2
        p_c = 9.23e16*1e6

        A = 0.2364*phi**-1.474
        N_0 = 1.577e18*phi**0.46

        if N_A < N_0:
            B = 0.433*phi**0.2213
        else:
            B = 1.268-0.338*phi

        N_h = N_A*(1-A*np.exp(-(B*np.log(N_A/N_0))**2))
        mu = phi**1.5*(mu1*np.exp(-p_c/N_h) + mu_max/(1+(N_h/C_r)**alpha) - mu2/(1+(C_s/N_h)**beta))
        omega_p = (N_h*e**2/mstar/eps0)**0.5
        gamma = e/mstar/mu
        eps = eps_inf - omega_p**2/w/(w+1j*gamma)
        return eps

    #Lorentz Model
    def SiO2(w):
        eps = 2.1*(w**2-2e14**2+1j*5e12*w)/(w**2-1.5e14**2+1j*5e12*w)
        return eps

    #MATERIAL PROPERTIES --------------------------------------------------------------------------

    for i in range(1999):
        w = W[i]
        theta[i] = h_bar*w/(np.exp(h_bar*w/k_b/T_e)-1)

        eps_S[i,0] = w
        eps_S[i,1] = np.real(silver(w))
        eps_S[i,2] = np.imag(silver(w))

        eps_a[i,0] = w
        eps_a[i,1] = np.real(dopedSi(w,1100,N1))
        eps_a[i,2] = np.imag(dopedSi(w,1100,N1))

        eps_b[i,0] = w
        eps_b[i,1] = np.real(dopedSi(w,1100,N2))
        eps_b[i,2] = np.imag(dopedSi(w,1100,N2))

        eps_ARC[i,0] = w
        eps_ARC[i,1] = np.real(dopedSi(w,1100,N3))
        eps_ARC[i,2] = np.imag(dopedSi(w,1100,N3))

        eps_spacer[i,0] = w
        eps_spacer[i,1] = np.real(SiO2(w))
        eps_spacer[i,2] = np.imag(SiO2(w))

        eps_vac[i,0] = w
        eps_vac[i,1] = 1
        eps_vac[i,2] = 1e-18

    np.savetxt("Substrate.txt",eps_S)
    np.savetxt("Dielectric_a.txt",eps_a)
    np.savetxt("Dielectric_b.txt",eps_b)
    np.savetxt("ARC.txt",eps_ARC)
    np.savetxt("Vacuum.txt",eps_vac)
    np.savetxt("Spacer.txt",eps_spacer)

    #Setting up MESH ----------------------------------------------------------------------------------

    sim = SimulationPlanar()
    sim.AddMaterial('Substrate','Substrate.txt')
    sim.AddMaterial('Dielectric_a','Dielectric_a.txt')
    sim.AddMaterial('Spacer','Spacer.txt')
    sim.AddMaterial('Dielectric_b','Dielectric_b.txt')
    sim.AddMaterial('ARC','ARC.txt')
    sim.AddMaterial('Vacuum','Vacuum.txt')

    sim.AddLayer(layer_name = 'Substrate', thickness = 0, material_name = 'Substrate')
    sim.AddLayer(layer_name = 'Dielectric_a_1', thickness = t_d, material_name = 'Dielectric_a')
    sim.AddLayer(layer_name = 'Spacer_1', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_b_1', thickness = t_m, material_name = 'Dielectric_b')
    sim.AddLayer(layer_name = 'Spacer_2', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_a_2', thickness = t_d, material_name = 'Dielectric_a')
    sim.AddLayer(layer_name = 'Spacer_3', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_b_2', thickness = t_m, material_name = 'Dielectric_b')
    sim.AddLayer(layer_name = 'Spacer_4', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_a_3', thickness = t_d, material_name = 'Dielectric_a')
    sim.AddLayer(layer_name = 'Spacer_5', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_b_3', thickness = t_m, material_name = 'Dielectric_b')
    sim.AddLayer(layer_name = 'Spacer_6', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_a_4', thickness = t_d, material_name = 'Dielectric_a')
    sim.AddLayer(layer_name = 'Spacer_7', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_b_4', thickness = t_m, material_name = 'Dielectric_b')
    sim.AddLayer(layer_name = 'Spacer_8', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_a_5', thickness = t_d, material_name = 'Dielectric_a')
    sim.AddLayer(layer_name = 'Spacer_9', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'Dielectric_b_5', thickness = t_m, material_name = 'Dielectric_b')
    sim.AddLayer(layer_name = 'Spacer_10', thickness = 5e-9, material_name = 'Spacer')
    sim.AddLayer(layer_name = 'ARC', thickness = t_ARC, material_name = 'ARC')
    sim.AddLayer(layer_name = 'Vacuum', thickness = 0, material_name = 'Vacuum')

    sim.OptUseQuadgk()
    sim.SetKParallelIntegral(1)
    sim.SetThread(num_thread = 48)

    #Analysis ----------------------------------------------------------------------------------

    sim.SetProbeLayer(layer_name = 'Vacuum')

    sim.SetSourceLayer(layer_name = 'Substrate')
    sim.SetSourceLayer(layer_name = 'Dielectric_a_1')
    sim.SetSourceLayer(layer_name = 'Spacer_1')
    sim.SetSourceLayer(layer_name = 'Dielectric_b_1')
    sim.SetSourceLayer(layer_name = 'Spacer_2')
    sim.SetSourceLayer(layer_name = 'Dielectric_a_2')
    sim.SetSourceLayer(layer_name = 'Spacer_3')
    sim.SetSourceLayer(layer_name = 'Dielectric_b_2')
    sim.SetSourceLayer(layer_name = 'Spacer_4')
    sim.SetSourceLayer(layer_name = 'Dielectric_a_3')
    sim.SetSourceLayer(layer_name = 'Spacer_5')
    sim.SetSourceLayer(layer_name = 'Dielectric_b_3')
    sim.SetSourceLayer(layer_name = 'Spacer_6')
    sim.SetSourceLayer(layer_name = 'Dielectric_a_4')
    sim.SetSourceLayer(layer_name = 'Spacer_7')
    sim.SetSourceLayer(layer_name = 'Dielectric_b_4')
    sim.SetSourceLayer(layer_name = 'Spacer_8')
    sim.SetSourceLayer(layer_name = 'Dielectric_a_5')
    sim.SetSourceLayer(layer_name = 'Spacer_9')
    sim.SetSourceLayer(layer_name = 'Dielectric_b_5')
    sim.SetSourceLayer(layer_name = 'Spacer_10')
    sim.SetSourceLayer(layer_name = 'ARC')

    sim.InitSimulation()
    sim.IntegrateKParallel()
    phi = sim.GetPhi()

    q_w = np.zeros(1999)

    for i in range(1999):
        q_w[i] = phi[i]*theta[i]
        q_b_w = theta[i]*W[i]**2/(4*np.pi**2*c**2)
        emissivity[i] = q_w[i]/q_b_w

    eff,P_r = InAsTPV(T_e,emissivity,EQE)

    print(eff)

    return emissivity