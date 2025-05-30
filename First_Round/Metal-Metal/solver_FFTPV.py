def sel_em_TPV(t_a,t_b,t_ARC,wp_S,gam_S,wp_a,gam_a,wp_b,gam_b,epsinf_ARC,wLO_ARC,wTO_ARC,gam_ARC,EQE):

    from MESH import SimulationPlanar
    import numpy as np
    from InAs_TPV import InAsTPV

    #Constants
    h_bar = 1.054571817E-34
    k_b = 1.380649E-23
    c = 2.99792458E8

    T_e = 1100

    #Preprocessing
    wLO_ARC = wLO_ARC*wTO_ARC
    dw = 1E12
    W = [dw*(i+1) for i in range(1999)]
    q = 0
    q_w = np.zeros(1999)
    theta = np.zeros(len(W))
    q_b_w = np.zeros(len(W))
    emissivity = np.zeros(len(W))
    eps_S = np.zeros(shape=(1999,3))
    eps_a = np.zeros(shape=(1999,3))
    eps_b = np.zeros(shape=(1999,3))
    eps_ARC = np.zeros(shape=(1999,3))
    eps_vac = np.zeros(shape=(1999,3))

    #Drude Model
    def Drude(w,wp,gam):
        eps = 1-wp**2/(w*(w+1j*gam))
        return eps

    #Lorentz Model
    def Lorentz(w,epsinf,wLO,wTO,gam):
        eps = epsinf*(w**2-wLO**2+1j*gam*w)/(w**2-wTO**2+1j*gam*w) + 1j*1e-18
        return eps

    #MATERIAL PROPERTIES --------------------------------------------------------------------------

    for i in range(1999):
        w = W[i]
        theta[i] = h_bar*w/(np.exp(h_bar*w/k_b/T_e)-1)

        eps_S[i,0] = w
        eps_S[i,1] = np.real(Drude(w,wp_S,gam_S))
        eps_S[i,2] = np.imag(Drude(w,wp_S,gam_S))

        eps_a[i,0] = w
        eps_a[i,1] = np.real(Drude(w,wp_a,gam_a))
        eps_a[i,2] = np.imag(Drude(w,wp_a,gam_a))

        eps_b[i,0] = w
        eps_b[i,1] = np.real(Drude(w,wp_b,gam_b))
        eps_b[i,2] = np.imag(Drude(w,wp_b,gam_b))

        eps_ARC[i,0] = w
        eps_ARC[i,1] = np.real(Lorentz(w,epsinf_ARC,wLO_ARC,wTO_ARC,gam_ARC))
        eps_ARC[i,2] = np.imag(Lorentz(w,epsinf_ARC,wLO_ARC,wTO_ARC,gam_ARC))

        eps_vac[i,0] = w
        eps_vac[i,1] = 1
        eps_vac[i,2] = 1e-18

    np.savetxt("Substrate.txt",eps_S)
    np.savetxt("Metal_a.txt",eps_a)
    np.savetxt("Metal_b.txt",eps_b)
    np.savetxt("ARC.txt",eps_ARC)
    np.savetxt("Vacuum.txt",eps_vac)

    #Setting up MESH ----------------------------------------------------------------------------------

    sim = SimulationPlanar()
    sim.AddMaterial('Substrate','Substrate.txt')
    sim.AddMaterial('Metal_a','Metal_a.txt')
    sim.AddMaterial('Metal_b','Metal_b.txt')
    sim.AddMaterial('ARC','ARC.txt')
    sim.AddMaterial('Vacuum','Vacuum.txt')

    sim.AddLayer(layer_name = 'Substrate', thickness = 0, material_name = 'Substrate')
    sim.AddLayer(layer_name = 'Metal_a_1', thickness = t_a, material_name = 'Metal_a')
    sim.AddLayer(layer_name = 'Metal_b_1', thickness = t_b, material_name = 'Metal_b')
    sim.AddLayer(layer_name = 'Metal_a_2', thickness = t_a, material_name = 'Metal_a')
    sim.AddLayer(layer_name = 'Metal_b_2', thickness = t_b, material_name = 'Metal_b')
    sim.AddLayer(layer_name = 'Metal_a_3', thickness = t_a, material_name = 'Metal_a')
    sim.AddLayer(layer_name = 'Metal_b_3', thickness = t_b, material_name = 'Metal_b')
    sim.AddLayer(layer_name = 'Metal_a_4', thickness = t_a, material_name = 'Metal_a')
    sim.AddLayer(layer_name = 'Metal_b_4', thickness = t_b, material_name = 'Metal_b')
    sim.AddLayer(layer_name = 'Metal_a_5', thickness = t_a, material_name = 'Metal_a')
    sim.AddLayer(layer_name = 'Metal_b_5', thickness = t_b, material_name = 'Metal_b')
    sim.AddLayer(layer_name = 'ARC', thickness = t_ARC, material_name = 'ARC')
    sim.AddLayer(layer_name = 'Vacuum', thickness = 0, material_name = 'Vacuum')

    sim.OptUseQuadgk()
    sim.SetKParallelIntegral(1)
    sim.SetThread(num_thread = 48)

    #Analysis ----------------------------------------------------------------------------------

    sim.SetProbeLayer(layer_name = 'Vacuum')

    sim.SetSourceLayer(layer_name = 'Substrate')
    sim.SetSourceLayer(layer_name = 'Metal_a_1')
    sim.SetSourceLayer(layer_name = 'Metal_b_1')
    sim.SetSourceLayer(layer_name = 'Metal_a_2')
    sim.SetSourceLayer(layer_name = 'Metal_b_2')
    sim.SetSourceLayer(layer_name = 'Metal_a_3')
    sim.SetSourceLayer(layer_name = 'Metal_b_3')
    sim.SetSourceLayer(layer_name = 'Metal_a_4')
    sim.SetSourceLayer(layer_name = 'Metal_b_4')
    sim.SetSourceLayer(layer_name = 'Metal_a_5')
    sim.SetSourceLayer(layer_name = 'Metal_b_5')
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