from collisional_tools import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#parâmetros físicos
g     = sympy.symbols("g", real=True)
omega = sympy.symbols("om", real=True)
lamb  = sympy.symbols("lam", real=True)
Hs = sz * omega / 2

# coerência da ancilla
Xa = sx * lamb

# operadores de salto
L, A = sp, sp

# estado térmico da ancilla
rhoA = np.array([
    [one/3, 0],
    [0, 2*one/3]
])

#estado simbólico do sistema
rhoS = build_rho(2)

#equação mestra
D = dissipator_entry(g, L, A, rhoS, rhoA)
coherent = coherent_entry(g, L, A, rhoS, Xa)
liouville = -I * commutator(Hs, rhoS)
drhoS = simple_expand(liouville + coherent + D)

#dinâmica com e sem coerência
params_base = {
    g: 1,
    omega: 2
}

#lambda = 0
f_sem = make_odeint_function(
    drhoS,
    {**params_base, lamb: 0}
)

#lambda = 0.5
f_com = make_odeint_function(
    drhoS,
    {**params_base, lamb: 0.5}
)

#estado inicial
rhoS0 = np.array([
    [1, 0],
    [0, 0]
])

rhoS0_vec = hermitian2vec(rhoS0)

#tempo de integração
time = 10
t = np.linspace(0, time, 300)

#integração
sol_sem = odeint(f_sem, rhoS0_vec, t)
sol_com = odeint(f_com, rhoS0_vec, t)

#operador sz numérico
sz_num = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

#função para calcular observáveis
def observaveis(solucao):
    sz_vals   = []
    pop0_vals = []
    pop1_vals = []
    coh_vals  = []

    for vec in solucao:
        rho = np.array(
            vec2hermitian(vec),
            dtype=complex
        )

        # <σz>
        sz = np.trace(sz_num @ rho).real

        # elementos diagonais
        pop0 = rho[0,0].real
        pop1 = rho[1,1].real

        # coerência |rho01|
        coh = np.abs(rho[0,1])

        sz_vals.append(sz)
        pop0_vals.append(pop0)
        pop1_vals.append(pop1)
        coh_vals.append(coh)

    return (
        np.array(sz_vals),
        np.array(pop0_vals),
        np.array(pop1_vals),
        np.array(coh_vals)
    )

#calcula observáveis
sz_sem, p0_sem, p1_sem, coh_sem = observaveis(sol_sem)
sz_com, p0_com, p1_com, coh_com = observaveis(sol_com)

#funções termodinâmicas
#operadores numéricos
sx_num = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sp_num = np.array([
    [0, 1],
    [0, 0]
], dtype=complex)

sm_num = np.array([
    [0, 0],
    [1, 0]
], dtype=complex)

Hs_num = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

#dissipador numérico
def dissipador_numerico(rho):
    gamma_up = 1/3
    gamma_down = 2/3

    termo1 = (
        sp_num @ rho @ sm_num
        - 0.5 * (
            sm_num @ sp_num @ rho
            + rho @ sm_num @ sp_num
        )
    )

    termo2 = (
        sm_num @ rho @ sp_num
        - 0.5 * (
            sp_num @ sm_num @ rho
            + rho @ sp_num @ sm_num
        )
    )

    D_num = (
        gamma_up * termo1
        + gamma_down * termo2
    )
    return D_num

#calor incoerente
def calor_incoerente(rho):
    D_num = dissipador_numerico(rho)
    Q = np.trace(Hs_num @ D_num)
    return np.real(Q)

#trabalho coerente
def trabalho_coerente(rho, lamb_val):
    G = sx_num
    comm = G @ Hs_num - Hs_num @ G
    W = 1j * lamb_val * np.trace(comm @ rho)
    return np.real(W)

#calcula fluxos termodinâmicos
Q_vals = []
W_vals = []

for vec in sol_com:
    rho = np.array(
        vec2hermitian(vec),
        dtype=complex
    )
    Q = calor_incoerente(rho)
    W = trabalho_coerente(rho, 0.5)
    Q_vals.append(Q)
    W_vals.append(W)
Q_vals = np.array(Q_vals)
W_vals = np.array(W_vals)

#gráfico: calor incoerente
plt.figure(figsize=(8,4))
plt.plot(t, Q_vals)
plt.xlabel("Tempo")
plt.ylabel("Q_inc")
plt.title("Fluxo de calor incoerente")
plt.grid()
plt.tight_layout()
plt.show()

#ráfgico: trabalho coerente
plt.figure(figsize=(8,4))
plt.plot(t, W_vals)
plt.xlabel("Tempo")
plt.ylabel("W_C")
plt.title("Trabalho coerente")
plt.grid()
plt.tight_layout()
plt.show()