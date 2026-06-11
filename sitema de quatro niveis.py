import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collisional_tools import *

# PARTE 1 — Parâmetros
Omega    = 1.0
beta     = 1.0
g        = 1.0
lam_list = [0.0, 0.2, 0.5, 1.0]
tmax     = 20.0

# PARTE 2 — Operadores 4x4
# Hamiltoniana: Hs = omega·diag(0,1,2,3)
Hs = Omega * np.diag([0, 1, 2, 3]).astype(complex)

# Operador de salto L: AUMENTA energia (σ₊ generalizado)
# L|0>=|1>, L|1>=|2>, L|2>=|3>
L4 = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=complex)

Ld4 = L4.conj().T   # L adaga: REDUZ energia (σ₋ generalizado)
                    # L adaga|1>=|0>, L adaga|2>=|1>, L†|3>=|2>

print("Verificação de L:")
basis = [np.array([1,0,0,0]), np.array([0,1,0,0]),
         np.array([0,0,1,0]), np.array([0,0,0,1])]
for i, v in enumerate(basis):
    result = L4 @ v
    print(f"  L|{i}⟩ = {result}")

print("Verificação de L†:")
for i, v in enumerate(basis):
    result = Ld4 @ v
    print(f"  L†|{i}⟩ = {result}")

# PARTE 3 — Ancilla: qubit 
Z_A     = np.exp(+beta*Omega/2) + np.exp(-beta*Omega/2)
rhoA_th = np.array([
    [np.exp(+beta*Omega/2)/Z_A, 0],
    [0, np.exp(-beta*Omega/2)/Z_A]
], dtype=complex)

chiA = np.array([[0, 1], [1, 0]], dtype=complex)   # σx

pg = np.real(rhoA_th[0, 0])
pe = np.real(rhoA_th[1, 1])
print(f"\npg={pg:.4f}, pe={pe:.4f}")

# Operadores da ancilla (2x2)
sp2 = np.array([[0,1],[0,0]], dtype=complex)
sm2 = np.array([[0,0],[1,0]], dtype=complex)

# Taxas γ
# Canal L (sobe energia): A=σ₊
gamma_m_L  = g**2 * np.trace(rhoA_th @ sp2 @ sm2).real
gamma_p_L  = g**2 * np.trace(rhoA_th @ sm2 @ sp2).real
# Canal L adaga (desce energia): A=σ₋
gamma_m_Ld = g**2 * np.trace(rhoA_th @ sm2 @ sp2).real
gamma_p_Ld = g**2 * np.trace(rhoA_th @ sp2 @ sm2).real

print(f"γ⁻_L={gamma_m_L:.4f},  γ⁺_L={gamma_p_L:.4f}")
print(f"γ⁻_L†={gamma_m_Ld:.4f}, γ⁺_L†={gamma_p_Ld:.4f}")

# PARTE 4 — Funções auxiliares
def dissipador_4x4(rho, L_n, gamma_m, gamma_p):
    Ld  = L_n.conj().T
    LdL = Ld @ L_n
    LLd = L_n @ Ld
    return (gamma_m * (L_n @ rho @ Ld - 0.5*(LdL @ rho + rho @ LdL)) +
            gamma_p * (Ld  @ rho @ L_n - 0.5*(LLd @ rho + rho @ LLd)))

def coerente_4x4(rho, L_n, avA, avAd):
    G = avAd * L_n + avA * L_n.conj().T
    return -1j * (G @ rho - rho @ G)

# PARTE 5 — Equação mestra
def master_equation(t, y, lam):
    # reconstrói ρ 4x4 a partir do vetor real
    rho = (y[:16] + 1j*y[16:]).reshape(4, 4)
    rho = (rho + rho.conj().T) / 2   # força hermiticidade

    # termos dissipativos
    D = (dissipador_4x4(rho, L4,  gamma_m_L,  gamma_p_L ) +
         dissipador_4x4(rho, Ld4, gamma_m_Ld, gamma_p_Ld))

    # termos coerentes
    chiA_lam = lam * chiA
    avA_sp   = np.trace(chiA_lam @ sp2)   # ⟨σ₊⟩chi
    avA_sm   = np.trace(chiA_lam @ sm2)   # ⟨σ₋⟩chi

    C = (coerente_4x4(rho, L4,  avA_sp, avA_sp.conj()) +
         coerente_4x4(rho, Ld4, avA_sm, avA_sm.conj()))

    drho = -1j * (Hs @ rho - rho @ Hs) + D + C

    drho_flat = drho.reshape(-1)
    return np.concatenate([drho_flat.real, drho_flat.imag])

def evolve(rho0, lam):
    rho_flat = rho0.reshape(-1)
    y0 = np.concatenate([rho_flat.real, rho_flat.imag])
    sol = solve_ivp(
        master_equation,
        [0, tmax],
        y0,
        args=(lam,),
        t_eval=[tmax],
        method="RK45",
        rtol=1e-8, atol=1e-10
    )
    rho_f = (sol.y[:16, -1] + 1j*sol.y[16:, -1]).reshape(4, 4)
    return (rho_f + rho_f.conj().T) / 2

# PARTE 6 — Energias, autoestados e populações térmicas
energias = [0.0, Omega, 2*Omega, 3*Omega]
autoestados = [np.eye(4)[m].reshape(4,1).astype(complex) for m in range(4)]

Z_S  = sum(np.exp(-beta*E) for E in energias)
p_th = [np.exp(-beta*E)/Z_S for E in energias]

print("\nPopulações térmicas do sistema:")
for m, (E, p) in enumerate(zip(energias, p_th)):
    print(f"  p(|{m}⟩, E={E:.1f}Ω) = {p:.4f}")

# PARTE 7 — Protocolo TPM generalizado
print("\n" + "="*55)
print("  TPM — sistema de 4 níveis")
print("="*55)

Delta_list = []
PQ_list    = []

for lam in lam_list:
    print(f"\n── λ = {lam} ──")

    # Matriz de transição 4x4: P[m,n] = prob de m -> n
    P_trans = np.zeros((4, 4))
    for m, psi_m in enumerate(autoestados):
        rho_m   = psi_m @ psi_m.conj().T
        rho_m_t = evolve(rho_m, lam)
        for n, psi_n in enumerate(autoestados):
            P_trans[m, n] = np.real(
                (psi_n.conj().T @ rho_m_t @ psi_n)[0, 0]
            )

    print("Matriz P[m→n]:")
    print(np.round(P_trans, 4))

    # Distribuição de calor P(Q) 
    PQ = {}
    for m in range(4):
        for n in range(4):
            Q = round(energias[n] - energias[m], 10)
            PQ[Q] = PQ.get(Q, 0) + p_th[m] * P_trans[m, n]

    print("\nDistribuição P(Q):")
    for Q in sorted(PQ):
        print(f"  P(Q={Q:+.1f}) = {PQ[Q]:.6f}")

    # Razão P(Q)/P(-Q) e Σ para cada Q>0
    print("\nRazões de flutuação:")
    eps      = 1e-14
    Q_vals   = [Omega, 2*Omega, 3*Omega]
    Sigmas   = []
    for Q in Q_vals:
        pQ  = PQ.get( round( Q, 10), 0) + eps
        pQn = PQ.get( round(-Q, 10), 0) + eps
        Sigma_Q = np.log(pQ / pQn)
        Sigmas.append(Sigma_Q)
        print(f"  Q={Q:+.2f}: Σ={Sigma_Q:.4f},  β·Q={beta*Q:.4f},  Δχ=Σ-β·Q={Sigma_Q-beta*Q:.4f}")

    # delta_chi médio sobre os Q positivos
    Delta_chi = np.mean([S - beta*Q for S, Q in zip(Sigmas, Q_vals)])
    Delta_list.append(Delta_chi)
    PQ_list.append(PQ)
    print(f"  ⟨Δχ⟩ médio = {Delta_chi:.6f}")

# PARTE 8 — Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: P(Q) para cada λ
Qs_vals = sorted(PQ_list[0].keys())
x       = np.arange(len(Qs_vals))
width   = 0.2
cores   = ["steelblue", "coral", "seagreen", "purple"]

for i, (lam, PQ) in enumerate(zip(lam_list, PQ_list)):
    probs = [PQ.get(Q, 0) for Q in Qs_vals]
    axes[0].bar(x + i*width, probs, width,
                label=f"λ={lam}", color=cores[i], alpha=0.85)

axes[0].set_xticks(x + width*1.5)
axes[0].set_xticklabels([f"{Q:+.0f}Ω" for Q in Qs_vals], fontsize=9)
axes[0].set_xlabel("Q (calor)", fontsize=13)
axes[0].set_ylabel("P(Q)", fontsize=13)
axes[0].set_title("Distribuição de calor P(Q)\nsistema de 4 níveis", fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# Plot 2: Δχ vs λ
axes[1].plot(lam_list, Delta_list, 'o-', color="coral",
             linewidth=2, markersize=8)
axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[1].set_xlabel(r"$\lambda$", fontsize=14)
axes[1].set_ylabel(r"$\langle\Delta_\chi\rangle$", fontsize=14)
axes[1].set_title("Contribuição coerente média\nvs λ", fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tarefa_4niveis.png", dpi=150)
plt.show()

"Matriz de transição: todas as linhas são iguais para cada λ. Isso significa que o estado estacionário é o mesmo independente do estado inicial "
"— o sistema esquece de onde veio, o que é esperado para uma dinâmica markoviana com tmax grande."
"Distribuição P(Q): para lambda=0, P(Q) é assimétrica — "
"há muito mais probabilidade de calor positivo (Q=+3 omega tem 41%) do que negativo (Q=−3omega tem 0.1%). "
"Conforme lambda aumenta, a distribuição se torna mais simétrica — a coerência equilibra as probabilidades."
"O resultado Δχ para λ=0:"
"Para λ=0, delta_chi = 1, 2, 3 para Q = omega, 2omega, 3omega. Isso não é zero, "
"logo significa que a fórmula Σ = β·Q não é satisfeita mesmo sem coerência. Então com a coerencia: Σ = β·Q + Δχ(λ=0) + contribuição de λ"