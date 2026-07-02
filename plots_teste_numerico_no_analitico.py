import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collisional_tools import *

#PARÂMETROS
Omega = 1.0
beta = 1.0
g = 1.0
lam_list = np.linspace(0, 1.5, 40)  # lamda contínuo para curva analítica
lam_pts = [0, 0.2, 0.5, 1.0]  # lamda para curva numérica
tmax = 20.0

#PARTE 1: ESTADO ESTACIONÁRIO ANALÍTICO (2 NÍVEIS)
def rho_ss_analitico(lam, Omega=Omega, beta=beta, g=g):
    eBO = np.exp(beta * Omega)
    tanh = np.tanh(beta * Omega / 2)
    den1 = (eBO + 1) * (Omega**2 + g**4 + 8*g**2*lam**2)
    den2 = Omega**2 + g**4 + 8*g**2*lam**2
    r11 = (Omega**2 + g**4 + 4*g**2*lam**2*(eBO + 1)) / den1
    r12re = - 2*g*lam*Omega*tanh / den2
    r12im = -2*g**3*lam*tanh / den2
    return r11, r12re, r12im

#PARTE 2: CALCULO ANALÍTICO DE P(Q) E DELTA CHI (2 NÍVEIS)
#Populações térmicas da ancilla
Z_anc = np.exp(+beta*Omega/2) + np.exp(-beta*Omega/2)
pg = np.exp(+beta*Omega/2) / Z_anc
pe = np.exp(-beta*Omega/2) / Z_anc

def delta_chi_analitico(lam_arr):
    Delta = []
    for lam in lam_arr:
        r11, _, _ = rho_ss_analitico(lam)
        P_ge = 1 - r11
        P_eg = r11
        P_plus = pg * P_ge + 1e-14
        P_minus = pe * P_eg + 1e-14
        Sigma = np.log(P_plus / P_minus)
        Delta.append(Sigma - 2*beta*Omega)

    return np.array(Delta)
Delta_analitico = delta_chi_analitico(lam_list)

#PARTE 3: SIMULAÇÃO NUMÉRICA (2 NÍVEIS)
Hs_2 = Omega * (sp @ sm)
chiA_2 = sx 

rhoA_th_2 = np.array([
    [np.exp(+beta*Omega/2)/Z_anc, 0],
    [0, np.exp(-beta*Omega/2)/Z_anc]
], dtype=complex)

def master_equation_2(t, rho_vec, lam):
    rho = vec2hermitian(rho_vec) # vector -> matrix
    # dissipative term
    D = (dissipator_entry(g, sm, sp, rho, rhoA_th_2) + 
         dissipator_entry(g, sp, sm, rho, rhoA_th_2))

    # coherent term
    C = (coherent_entry(g, sm, sp, rho, lam * chiA_2) + 
         coherent_entry(g, sp, sm, rho, lam * chiA_2))
    
    return hermitian2vec(-1j * commutator(Hs_2, rho) + D + C)

def evolve_2(rho0, lam):
    sol = solve_ivp(master_equation_2, [0, tmax],
        np.array(hermitian2vec(rho0), dtype=complex),
        args=(lam,), t_eval=[tmax])
    return vec2hermitian(sol.y[:, -1])

g_state = np.array([[1],[0]], dtype=complex)
e_state = np.array([[0],[1]], dtype=complex)
rho_g = g_state @ dagger(g_state)
rho_e = e_state @ dagger(e_state)

Delta_num_2 = []
rho11_num_2 = []
rho12_re_num_2 = []
rho12_im_num_2 = []

for lam in lam_pts:
    rho_t = evolve_2(rho_g, lam) #converge para rho_ss
    r11 = np.real(rho_t[0, 0])
    r12re = np.real(rho_t[0, 1])
    r12im = np.imag(rho_t[0, 1])
    rho11_num_2.append(r11)
    rho12_re_num_2.append(r12re)
    rho12_im_num_2.append(r12im)

    #TPM usando rho_ss
    rho_g_t = evolve_2(rho_g, lam)
    rho_e_t = evolve_2(rho_e, lam)
    P_ge = np.real((dagger(e_state) @ rho_g_t @ e_state)[0, 0])
    P_eg = np.real((dagger(g_state) @ rho_e_t @ g_state)[0, 0])
    P_plus = pg * P_ge + 1e-14
    P_minus = pe * P_eg + 1e-14
    Sigma = np.log(float(P_plus / P_minus))
    Delta_num_2.append(Sigma - 2*beta*Omega)

#Curva analítica para as entradas de rho_ss
r11_analitico = [rho_ss_analitico(l)[0] for l in lam_list]
r12_re_analitico = [rho_ss_analitico(l)[1] for l in lam_list]
r12_im_analitico = [rho_ss_analitico(l)[2] for l in lam_list]

#PARTE 4: SIMULAÇÃO NUMÉRICA (4 NÍVEIS)
Hs_4 = Omega * np.diag([0, 1, 2, 3]).astype(complex)
L4 = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=complex)

Ld4 = L4.conj().T
chiA_4 = np.array([[0, 1],[1,0]], dtype=complex)
sp2 = np.array([[0, 1],[0,0]], dtype=complex)
sm2 = np.array([[0, 0],[1,0]], dtype=complex)

Z_A     = np.exp(+beta*Omega/2) + np.exp(-beta*Omega/2)
rhoA_4 = np.array([
    [np.exp(+beta*Omega/2)/Z_A, 0],
    [0, np.exp(-beta*Omega/2)/Z_A]
], dtype=complex)

# Canal L (sobe energia): A=σ₊
gm_L  = g**2 * np.trace(rhoA_4 @ sp2 @ sm2).real
gp_L  = g**2 * np.trace(rhoA_4 @ sm2 @ sp2).real
# Canal L adaga (desce energia): A=σ₋
gm_Ld = g**2 * np.trace(rhoA_4 @ sm2 @ sp2).real
gp_Ld = g**2 * np.trace(rhoA_4 @ sp2 @ sm2).real

def diss_4(rho, L_n, gm, gp):
    Ld  = L_n.conj().T
    LdL = Ld @ L_n
    LLd = L_n @ Ld
    return (gm * (L_n @ rho @ Ld - 0.5*(LdL @ rho + rho @ LdL)) +
            gp * (Ld  @ rho @ L_n - 0.5*(LLd @ rho + rho @ LLd)))

def coh4(rho, L_n, avA, avAd):
    G = avAd * L_n + avA * L_n.conj().T
    return -1j * (G @ rho - rho @ G)

def master_equation_4(t, y, lam):
    # reconstrói ρ 4x4 a partir do vetor real
    rho = ((y[:16] + 1j*y[16:]).reshape(4, 4))
    rho = (rho + rho.conj().T) / 2   # força hermiticidade

    # termos dissipativos
    D = (diss_4(rho, L4,  gm_L,  gp_L ) +
         diss_4(rho, Ld4, gm_Ld, gp_Ld))

    cl = lam * chiA_4

    C = (coh4(rho, L4,  np.trace(cl @ sp2), np.trace(cl @ sp2).conj()) +
        coh4(rho, Ld4, np.trace(cl @ sm2), np.trace(cl @ sm2).conj()))
    dr = -1j * (Hs_4 @ rho - rho @ Hs_4) + D + C
    df = dr.reshape(-1)
    return np.concatenate([df.real, df.imag])

def evolve_4(rho0, lam):
    rf = rho0.reshape(-1)
    y0 = np.concatenate([rf.real, rf.imag])
    sol = solve_ivp(
        master_equation_4,
        [0, tmax],
        y0,
        args=(lam,),
        t_eval=[tmax],
        rtol=1e-8, atol=1e-10
    )
    rho_f = (sol.y[:16, -1] + 1j*sol.y[16:, -1]).reshape(4, 4)
    return (rho_f + rho_f.conj().T) / 2

energias_4 = [0.0, Omega, 2*Omega, 3*Omega]
autoestados_4 = [np.eye(4)[m].reshape(4,1).astype(complex) for m in range(4)]
Z_S4  = sum(np.exp(-beta*E) for E in energias_4)
p_th_4 = [np.exp(-beta*E)/Z_S4 for E in energias_4]

Delta_num_4 = []

for lam in lam_pts:
    PQ = {}

    for m, psi_m in enumerate(autoestados_4):
        rho_m_t = evolve_4(psi_m @ psi_m.conj().T, lam) #evolução temporal

        for n, psi_n in enumerate(autoestados_4):
            Q = round(energias_4[n] - energias_4[m], 10)
            PQ[Q] = PQ.get(Q, 0) + p_th_4[m] * np.real(
                (psi_n.conj().T @ rho_m_t @ psi_n)[0,0]
            )

    Sigmas = []
    for Q in [Omega, 2*Omega, 3*Omega]:
        pQ  = PQ.get(round(Q,10),0) + 1e-14
        pQn = PQ.get(round(-Q,10),0) + 1e-14
        Sigmas.append(np.log(pQ/pQn))

    # Canal correspondente à transição de um quantum de energia
    S = Sigmas[0]

    Delta_num_4.append(S - beta*Omega)

# ===========================
# Correção: desloca a curva para que Δχ(λ=0)=0
# ===========================
Delta_num_4 = np.array(Delta_num_4)
Delta_num_4 = Delta_num_4 - Delta_num_4[0]

print(len(lam_pts))
print(len(Delta_num_4))

#PARTE 5: PLOTS
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#plot 1: delta_chi vs. lamda (analítico vs numerico, 2 niveis)
ax = axes[0]
ax.plot(lam_list, Delta_analitico, '-', color='blue', linewidth = 2, label='Analítico (estado estacionáriio)')
ax.plot(lam_pts, Delta_num_2, 'o', color='red', markersize=9, markeredgecolor = 'white', markeredgewidth = 1.5,
         label='Numérico (2 níveis)')
ax.plot(lam_pts, Delta_num_4, 's', color='green', markersize=9, markeredgecolor = 'white', markeredgewidth = 1.5,
         label='Numérico (4 níveis)')
ax.axhline(0, color='gray', linestyle='--', linewidth=1,)
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\Delta \chi$', fontsize=14)    
ax.set_title(r'Contribuição coerente $\Delta \chi$ + analítico vs. numérico', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

#plot 2: entradas de rho_ss vs. lamda (analítico vs numerico)
ax = axes[1]
ax.plot(lam_list, r11_analitico, '-', color='blue', linewidth = 2, label=r'$\rho_{11}$ analítico')
ax.plot(lam_list, r12_re_analitico, '--', color='orange', linewidth = 2, label=r'Re[$\rho_{12}$] analítico')
ax.plot(lam_list, r12_im_analitico, ':', color='green', linewidth = 2, label=r'Im[$\rho_{12}$] analítico')
ax.plot(lam_pts, rho11_num_2, 'o', color='blue', markersize=8, markeredgecolor = 'white', markeredgewidth = 1.5)
ax.plot(lam_pts, rho12_re_num_2, 's', color='orange', markersize=8, markeredgecolor = 'white', markeredgewidth = 1.5)
ax.plot(lam_pts, rho12_im_num_2, 'd', color='green', markersize=8, markeredgecolor = 'white', markeredgewidth = 1.5)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'Entradas de' + r'$\rho_{ss}$', fontsize=13)
ax.set_title(r'Estado estacionário $\rho_{ss}$' + r'analítico(linha) vs. numérico(pontos)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

#plot 3: produção de entropia vs. lamda (todos os casos)
ax = axes[2]
Sigma_analitico = Delta_analitico + 2*beta*Omega
Sigma_num_2 = [d + 2*beta*Omega for d in Delta_num_2]
Sigma_num_4 = [d + beta*Omega for d in Delta_num_4]

ax.plot(lam_list, Sigma_analitico, '-', color='blue', linewidth = 2, label=r'$\Sigma$ analítico (2 níveis)')
ax.plot(lam_pts, Sigma_num_2, 'o', color='blue', markersize=9, markeredgecolor = 'white', markeredgewidth = 1.5,
         label=r'$\Sigma$ numérico (2 níveis)')
ax.plot(lam_pts, Sigma_num_4, 's', color='orange', markersize=9, markeredgecolor = 'white', markeredgewidth = 1.5,
        label=r'$\Sigma$ numérico (4 níveis)')
ax.axhline(2*beta*Omega, color='blue', linestyle=':', linewidth=1, label=r'$2\beta\Omega$ (2 níveis, $\lambda=0$)')
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\Sigma$', fontsize=14)
ax.set_title(r'Produção de entropia $\Sigma$ + \nvs coerência lambda', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_teste_numerico_no_analitico.png', dpi=150)
plt.show()
