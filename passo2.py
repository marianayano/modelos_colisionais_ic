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

        #elementos diagonais
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

#estado estacionário
print("\nEstado estacionário SEM coerência:\n")
print(
    np.round(
        np.array(vec2hermitian(sol_sem[-1]), dtype=complex),
        4
    )
)
print("\nEstado estacionário COM coerência:\n")
print(
    np.round(
        np.array(vec2hermitian(sol_com[-1]), dtype=complex),
        4
    )
)

#gráfico 1: <sz>(t)
plt.figure(figsize=(8,4))
plt.plot(
    t,
    sz_sem,
    label="λ = 0"
)
plt.plot(
    t,
    sz_com,
    "--",
    label="λ = 0.5"
)
plt.xlabel("Tempo")
plt.ylabel("⟨σz⟩")
plt.title("Evolução de ⟨σz⟩")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#gráfico 2: elementos diagonais 
plt.figure(figsize=(8,4))
plt.plot(
    t,
    p0_com,
    label="ρ00"
)
plt.plot(
    t,
    p1_com,
    label="ρ11"
)
plt.xlabel("Tempo")
plt.ylabel("Elementos diagonais")
plt.title("Elementos diagonais do sistema")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#gráfico 3: coerência
plt.figure(figsize=(8,4))
plt.plot(
    t,
    coh_sem,
    label="λ = 0"
)
plt.plot(
    t,
    coh_com,
    "--",
    label="λ = 0.5"
)
plt.xlabel("Tempo")
plt.ylabel("|ρ01|")
plt.title("Coerência quântica")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()