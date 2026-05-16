from collisional_tools import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

#parâmetros físicos
g     = sympy.symbols("g",   real=True)
omega = sympy.symbols("om",  real=True)
lamb  = sympy.symbols("lam", real=True)

Hs   = sz * omega / 2        # hamiltoniana do sistema
Xa   = sx * lamb             # parte coerente da ancilla (X_A = sx)
L, A = sp, sp                # operadores de salto
rhoA = np.array([[one/3, 0], [0, 2*one/3]])   # ancilla térmica
rhoS = build_rho(2)          # estado simbólico do sistema

#equação mestra (Eq. 6 do artigo)
D         = dissipator_entry(g, L, A, rhoS, rhoA)
coherent  = coherent_entry(g, L, A, rhoS, Xa)
liouville = -I * commutator(Hs, rhoS)
drhoS     = simple_expand(liouville + coherent + D)

#duas funções: com e sem coerência
params_base = {g: 1, omega: 2}

f_sem = make_odeint_function(drhoS, {**params_base, lamb: 0})    # lambda=0
f_com = make_odeint_function(drhoS, {**params_base, lamb: 0.5})  # lambda=0.5

#integração 
rhoS0     = np.array([[1, 0], [0, 0]])   # começa em |0><0|
rhoS0_vec = hermitian2vec(rhoS0)

time = 10
t    = np.linspace(0, time, 300)

sol_sem = odeint(f_sem, rhoS0_vec, t)
sol_com = odeint(f_com, rhoS0_vec, t)

# calcula <sz>(t) = Tr(sz ρ(t))
sz_num = np.array([[1, 0], [0, -1]], dtype=complex)

def sz_medio(solucao):
    resultado = []
    for vec in solucao:
        rho = vec2hermitian(vec)
        # converte de simbólico para numérico
        val = sum(sz_num[i][i] * complex(rho[i][i]) for i in range(2))
        resultado.append(val.real)
    return np.array(resultado)

sz_sem = sz_medio(sol_sem)
sz_com = sz_medio(sol_com)

#estado estacionário 
print("Estado estacionário SEM coerência (λ=0):")
print(np.round(vec2hermitian(sol_sem[-1]), 4))

print("\nEstado estacionário COM coerência (λ=0.5):")
print(np.round(vec2hermitian(sol_com[-1]), 4))

#plot
plt.figure(figsize=(8, 4))
plt.plot(t, sz_sem, label="λ = 0  (sem coerência)", color="steelblue")
plt.plot(t, sz_com, label="λ = 0.5 (com coerência)", color="coral", linestyle="--")
plt.xlabel("Tempo")
plt.ylabel("⟨σz⟩(t)")
plt.title("Evolução de ⟨σz⟩: efeito da coerência da ancilla")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("passo1_sz.png", dpi=150)
plt.show()
print("Gráfico salvo em passo1_sz.png")