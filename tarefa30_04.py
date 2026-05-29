import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collisional_tools import *

#Parametros
Omega = 1.0
beta  = 1.0
g     = 1.0
lam_list = [0.0, 0.2, 0.5, 1.0]
tmax = 20

#Operadores
Hs = Omega * (sp @ sm) #Hamiltoniana do sistema
chiA = sx #contribuição coerente da ancilla

#Estado térmico da ancilla
Z = np.exp(+beta*Omega/2) + np.exp(-beta*Omega/2)

rhoA_th = np.array([
    [np.exp(+beta*Omega/2)/Z, 0],
    [0, np.exp(-beta*Omega/2)/Z]
], dtype=complex)

#probabilidades

pg = np.real(rhoA_th[0,0])
pe = np.real(rhoA_th[1,1])

#equação mestra
def master_equation(t, rho_vec, lam):
    rho = vec2hermitian(rho_vec) # vector -> matrix
    # dissipative term
    D_down = dissipator_entry(
    g,
    sm,
    sp,
    rho,
    rhoA_th
    )

    D_up = dissipator_entry(
    g,
    sp,
    sm,
    rho,
    rhoA_th
    )

    D = D_down + D_up

    # coherent term
    C_down = coherent_entry(
    g,
    sm,
    sp,
    rho,
    lam * chiA
    )

    C_up = coherent_entry(
    g,
    sp,
    sm,
    rho,
    lam * chiA
    )
    
    C = C_down + C_up

    # total evolution
    drho = -1j * commutator(Hs, rho) + D + C

    # matrix -> vector
    return hermitian2vec(drho)

#Evolução do estado
def evolve(rho0, lam):
    y0 = np.array(
        hermitian2vec(rho0),
        dtype=complex
    )

    sol = solve_ivp(
        master_equation,
        [0, tmax],
        y0,
        args=(lam,),
        t_eval=[tmax]
    )

    rho_t = vec2hermitian(sol.y[:, -1])

    return rho_t

#estados de energia
g_state = np.array([[1],[0]], dtype=complex)
e_state = np.array([[0],[1]], dtype=complex)

rho_g = g_state @ dagger(g_state)
rho_e = e_state @ dagger(e_state)

#TPM protcol
print(" TWO-POINT MEASUREMENT ")

Delta_list = []

for lam in lam_list:
    #evolução dos estados iniciais
    rho_g_t = evolve(rho_g, lam)
    rho_e_t = evolve(rho_e, lam)

    #probabilidades de transição:
    # g -> e
    P_ge = np.real(
        (
            dagger(e_state)
            @ rho_g_t
            @ e_state
        )[0,0]
    )

    # e -> g
    P_eg = np.real(
        (
            dagger(g_state)
            @ rho_e_t
            @ g_state
        )[0,0]
    )

    #distribuição de probabilidades
    P_plus  = pg * P_ge
    P_minus = pe * P_eg

    eps = 1e-14

    P_plus  += eps
    P_minus += eps

    #flutuação
    Sigma = np.log(float(P_plus / P_minus))

    # "?"
    Delta_chi = Sigma - 2 * beta * Omega

    Delta_list.append(np.real(Delta_chi))

    #saida
    print(f"\nλ = {lam}")
    print(f"P(+Ω) = {P_plus:.6e}")
    print(f"P(-Ω) = {P_minus:.6e}")
    print(f"Σ = {Sigma:.6f}")
    print(f"? = Δχ = {Delta_chi:.6f}")

#graficos
plt.figure(figsize=(7,5))
plt.plot(
    lam_list,
    Delta_list,
    'o-',
    linewidth=2,
    markersize=8
)
plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel(r'$\Delta_\chi$', fontsize=14)
plt.title(
    r'Contribuição coerente $\Delta_\chi$ vs. $\lambda$',
    fontsize=14
)
plt.grid(True)
plt.tight_layout()
plt.show()