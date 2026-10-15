import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# (o import de collisional_tools não era usado neste script, então foi removido)

# PARTE 1 — Parâmetros
Omega    = 1.0
beta_S   = 1.0     # temperatura inversa inicial do sistema S
beta_A   = 1.0     # temperatura inversa inicial da ancilla A
delta_beta = beta_S - beta_A   # δβ = β_S − β_A  (papel: Σ = δβ·Q)
g        = 1.0
lam_list = [0.0, 0.2, 0.5, 1.0]
tmax     = 20.0

# PARTE 2 — Operadores 4x4
# Hamiltoniana: Hs = Omega·diag(0,1,2,3)
Hs = Omega * np.diag([0, 1, 2, 3]).astype(complex)

# Operador de salto L: AUMENTA energia do sistema
# L|0>=|1>, L|1>=|2>, L|2>=|3>
L4 = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=complex)

Ld4 = L4.conj().T   # L†: REDUZ energia do sistema

print("Verificação de L:")
basis = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]),
         np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
for i, v in enumerate(basis):
    print(f"  L|{i}⟩ = {L4 @ v}")

print("Verificação de L†:")
for i, v in enumerate(basis):
    print(f"  L†|{i}⟩ = {Ld4 @ v}")

# PARTE 3 — Ancilla: qubit
# Convenção: índice 0 = FUNDAMENTAL (energia −Ω/2), índice 1 = EXCITADO (+Ω/2)
Z_A = np.exp(+beta_A*Omega/2) + np.exp(-beta_A*Omega/2)
rhoA_th = np.array([
    [np.exp(+beta_A*Omega/2)/Z_A, 0],
    [0, np.exp(-beta_A*Omega/2)/Z_A]
], dtype=complex)

chiA = np.array([[0, 1], [1, 0]], dtype=complex)   # σx

pg = np.real(rhoA_th[0, 0])   # população do fundamental
pe = np.real(rhoA_th[1, 1])   # população do excitado
print(f"\npg={pg:.4f}, pe={pe:.4f}")

# *** CORREÇÃO PRINCIPAL ***
# Antes: sp2 = |0><1| e sm2 = |1><0|. Com |0> = fundamental, isso fazia
# "σ₊" ABAIXAR a ancilla (e σ₋ subir), invertendo as taxas: o sistema
# subia com taxa ∝ pg (> pe) -> inversão de população, P(Q) assimétrica
# mesmo com β_S = β_A.
# Agora: σ₊ = |e><g| = |1><0| (sobe), σ₋ = |g><e| = |0><1| (desce).
sp2 = np.array([[0, 0], [1, 0]], dtype=complex)   # σ₊
sm2 = np.array([[0, 1], [0, 0]], dtype=complex)   # σ₋

# Taxas γ
# Canal L (sistema sobe => ancilla desce, precisa da ancilla excitada): ∝ pe
gamma_m_L  = g**2 * np.trace(rhoA_th @ sp2 @ sm2).real   # = g² pe
gamma_p_L  = g**2 * np.trace(rhoA_th @ sm2 @ sp2).real   # = g² pg
# Canal L† (sistema desce => ancilla sobe, precisa da ancilla fundamental): ∝ pg
gamma_m_Ld = g**2 * np.trace(rhoA_th @ sm2 @ sp2).real   # = g² pg
gamma_p_Ld = g**2 * np.trace(rhoA_th @ sp2 @ sm2).real   # = g² pe

# Taxa efetiva de subida/descida do sistema
# subida  (L ρ L†):  gamma_m_L  + gamma_p_Ld = 2 g² pe
# descida (L† ρ L):  gamma_p_L  + gamma_m_Ld = 2 g² pg
# razão subida/descida = pe/pg = exp(-β_A Ω)  -> balanço detalhado em β_A
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
    rho = (y[:16] + 1j*y[16:]).reshape(4, 4)
    rho = (rho + rho.conj().T) / 2   # força hermiticidade

    D = (dissipador_4x4(rho, L4,  gamma_m_L,  gamma_p_L) +
         dissipador_4x4(rho, Ld4, gamma_m_Ld, gamma_p_Ld))

    chiA_lam = lam * chiA
    avA_sp = np.trace(chiA_lam @ sp2)
    avA_sm = np.trace(chiA_lam @ sm2)

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
autoestados = [np.eye(4)[m].reshape(4, 1).astype(complex) for m in range(4)]

# Estado inicial do SISTEMA: térmico em β_S  (P(E1) = e^{-β_S E1}/Z1)
Z_S  = sum(np.exp(-beta_S*E) for E in energias)
p_th = [np.exp(-beta_S*E)/Z_S for E in energias]

print("\nPopulações térmicas do sistema (β_S):")
for m, (E, p) in enumerate(zip(energias, p_th)):
    print(f"  p(|{m}⟩, E={E:.1f}Ω) = {p:.4f}")

# PARTE 7 — Protocolo TPM generalizado
print("\n" + "="*55)
print("  TPM — sistema de 4 níveis")
print(f"  β_S={beta_S}, β_A={beta_A}, δβ=β_S−β_A={delta_beta}")
print("="*55)

Delta_list = []
PQ_list    = []

for lam in lam_list:
    print(f"\n── λ = {lam} ──")

    # Matriz de transição: P[m,n] = prob de m -> n
    P_trans = np.zeros((4, 4))
    for m, psi_m in enumerate(autoestados):
        rho_m   = psi_m @ psi_m.conj().T
        rho_m_t = evolve(rho_m, lam)
        for n, psi_n in enumerate(autoestados):
            P_trans[m, n] = np.real((psi_n.conj().T @ rho_m_t @ psi_n)[0, 0])

    print("Matriz P[m→n]:")
    print(np.round(P_trans, 4))

    # Distribuição de calor P(Q), Q = E_n − E_m (energia ganha pelo sistema)
    PQ = {}
    for m in range(4):
        for n in range(4):
            Q = round(energias[n] - energias[m], 10)
            PQ[Q] = PQ.get(Q, 0) + p_th[m] * P_trans[m, n]

    print("\nDistribuição P(Q):")
    for Q in sorted(PQ):
        print(f"  P(Q={Q:+.1f}) = {PQ[Q]:.6f}")

    # Razão de flutuação:  Σ = ln[P(Q)/P(−Q)]
    # Teorema (papel):  Σ = (β_S − β_A)·Q  quando não há coerência
    # Δχ = Σ − δβ·Q  mede o desvio devido à coerência
    print("\nRazões de flutuação:")
    eps    = 1e-14
    Q_vals = [Omega, 2*Omega, 3*Omega]
    Sigmas = []
    for Q in Q_vals:
        pQ  = PQ.get(round(Q, 10), 0) + eps
        pQn = PQ.get(round(-Q, 10), 0) + eps
        Sigma_Q = np.log(pQ / pQn)
        Sigmas.append(Sigma_Q)
        print(f"  Q={Q:+.2f}: Σ={Sigma_Q:.4f},  δβ·Q={delta_beta*Q:.4f},  "
              f"Δχ=Σ−δβ·Q={Sigma_Q - delta_beta*Q:.4f}")

    Delta_chi = np.mean([S - delta_beta*Q for S, Q in zip(Sigmas, Q_vals)])
    Delta_list.append(Delta_chi)
    PQ_list.append(PQ)
    print(f"  ⟨Δχ⟩ médio = {Delta_chi:.6f}")

# PARTE 8 — PLOTS
Qs_vals = sorted(PQ_list[0].keys())
x = np.arange(len(Qs_vals))
width = 0.2
cores = ["steelblue", "coral", "seagreen", "purple"]

plt.figure(figsize=(7, 5))
for i, (lam, PQ) in enumerate(zip(lam_list, PQ_list)):
    probs = [PQ.get(Q, 0) for Q in Qs_vals]
    plt.bar(x + i*width, probs, width, label=f"λ={lam}", color=cores[i], alpha=0.85)

plt.xticks(x + width*1.5, [f"{Q:+.0f}Ω" for Q in Qs_vals], fontsize=9)
plt.xlabel("Q (calor)", fontsize=13)
plt.ylabel("P(Q)", fontsize=13)
plt.title("Distribuição de calor P(Q)\nsistema de 4 níveis", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("plot_distribuicao_calor_4niveis.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(lam_list, Delta_list, 'o-', color="coral", linewidth=2, markersize=8)
plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.xlabel(r"$\lambda$", fontsize=14)
plt.ylabel(r"$\langle\Delta_\chi\rangle$", fontsize=14)
plt.title("Contribuição coerente média\nvs. λ", fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_delta_chi_4niveis.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# Notas (resultados esperados após a correção)
# - Matriz P[m→n]: linhas iguais (estado estacionário independe do inicial).
# - λ=0 e β_S=β_A: P(Q)=P(−Q), Σ=0 e Δχ=0 (equilíbrio, razão = e^0 = 1).
# - λ>0: Δχ ≠ 0 é a contribuição da coerência: Σ = δβ·Q + Δχ(λ).
# - Para β_S ≠ β_A e λ=0: Σ = (β_S−β_A)·Q, sem Δχ.
# ------------------------------------------------------------------