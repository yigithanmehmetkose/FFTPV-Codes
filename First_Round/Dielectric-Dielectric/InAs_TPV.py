def InAsTPV(T_e,E_e,InAsEQE):
    import numpy as np

    #Constants
    h_bar = 1.054571817E-34
    k_b = 1.380649E-23
    sigma = 5.670374419E-8
    q = 1.60217663E-19
    c = 2.99792458E8

    #InAs material properties
    E_g = 0.354*q
    N_i = 1e15
    N_a = 5e19
    D_e = 1e3
    tau_e = 30e-9
    N_d = 1e19
    D_h = 13
    tau_h = 3e-6

    #Inputs
    T_c = 300
    N = 666
    #T_e = 1100

    #Preprocessing
    dw = 1E12
    W = [dw*(i+1) for i in range(3*N+1)]
    I_j = []
    I_p = []

    #Integration
    for p in range(N):
        w = [W[3*p], W[3*p+1], W[3*p+2], W[3*p+3]]
        e = [E_e[3*p], E_e[3*p+1], E_e[3*p+2], E_e[3*p+3]]
        EQE = [InAsEQE[3*p], InAsEQE[3*p+1], InAsEQE[3*p+2], InAsEQE[3*p+3]]
        P_w = []
        N_w = []
        for i in range(4):
            P_w.append(e[i]*h_bar*w[i]**3/(4*np.pi**2*c**2*(np.exp(h_bar*w[i]/(k_b*T_e))-1)))
            N_w.append(e[i]*EQE[i]*w[i]**2/(4*np.pi**2*c**2*(np.exp(h_bar*w[i]/(k_b*T_e))-1)))
        I_p.append(3.*dw/8.*(P_w[0]+3*P_w[1]+3*P_w[2]+P_w[3]))
        I_j.append(3.*dw/8.*(N_w[0]+3*N_w[1]+3*N_w[2]+N_w[3]))

    P_r = np.sum(I_p)
    J_ph = q*np.sum(I_j)
    J_s = q*N_i**2*(np.sqrt(D_e/tau_e)/N_a + np.sqrt(D_h/tau_h)/N_d)*1e4
    V_oc = k_b*T_c/q*np.log(J_ph/J_s+1)

    #Secant

    dV = V_oc/1000
    P_e = 0

    for i in range(1000):
        V = i*dV
        P2 = V*(J_ph-J_s*(np.exp(q*V/k_b/300)-1))
        if P2 > P_e:
            P_e = P2

    eff = P_e/P_r

    return eff, P_e