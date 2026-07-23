import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collisional_tools import *

# ================================================================
# PARÂMETROS
# ================================================================
Omega = 1.0
beta  = 1.0
g     = 1.0
tau   = 20.0

Z_A     = np.exp(+beta*Omega/2) + np.exp(-beta*Omega/2)
rhoA_th = np.array([
    [np.exp(+beta*Omega/2)/Z_A, 0],
    [0, np.exp(-beta*Omega/2)/Z_A]
], dtype=complex)

sx_n = np.array([[0,1],[1,0]], dtype=complex)
sy_n = np.array([[0,-1j],[1j,0]], dtype=complex)
sz_n = np.array([[1,0],[0,-1]], dtype=complex)
paulis = [sx_n, sy_n, sz_n]

def build_rhoA1A2(lam):
    rho = np.kron(rhoA_th, rhoA_th)
    for sigma in paulis:
        rho += lam * np.kron(sigma, sigma)
    return rho

def partial_trace_1(rho4):
    return rho4[:2,:2] + rho4[2:,2:]

def partial_trace_2(rho4):
    return rho4[::2,::2] + rho4[1::2,1::2]

# ================================================================
# PARTE 1 — ENCONTRAR λ_max (positividade de ρ_{A1A2})
# ================================================================
print("=== ENCONTRANDO λ_max ===")
lam_scan = np.linspace(0, 1.5, 1000)
lam_max  = 0.0
for lam in lam_scan:
    evals = np.linalg.eigvalsh(build_rhoA1A2(lam))
    if np.all(evals >= -1e-10):
        lam_max = lam
    else:
        break

print(f"λ_max = {lam_max:.4f}  (acima disso ρ_{{A1A2}} tem autovalor negativo)")

# Calcula também λ_max analiticamente:
# O menor autovalor de ρ^th ⊗ ρ^th é pe² = (e^{-βΩ}/Z)²
# O menor autovalor de Σ_k σ_k ⊗ σ_k é −3 (soma dos 3 Paulis)
# Então λ_max ≈ pe²/3
pe = np.exp(-beta*Omega/2)/Z_A * np.exp(-beta*Omega/2)/Z_A  # pe²
print(f"Estimativa analítica λ_max ≈ pe²/3 = {pe/3:.4f}")
print(f"pg² = {(np.exp(+beta*Omega/2)/Z_A)**2:.4f}")

# ================================================================
# PARTE 2 — EVOLUÇÃO COM DUAS ANCILLAS
# ================================================================
Hs = Omega * (sp @ sm)

def evolve_step(rho0_S, rhoA_eff):
    chi_eff = rhoA_eff - np.diag(np.diag(rhoA_eff))
    def meq(t, rv):
        rho = vec2hermitian(rv)
        D = (dissipator_entry(g, sm, sp, rho, rhoA_eff) +
             dissipator_entry(g, sp, sm, rho, rhoA_eff))
        C = (coherent_entry(g, sm, sp, rho, chi_eff) +
             coherent_entry(g, sp, sm, rho, chi_eff))
        return hermitian2vec(-1j*commutator(Hs, rho) + D + C)
    sol = solve_ivp(meq, [0, tau],
                    np.array(hermitian2vec(rho0_S), dtype=complex),
                    t_eval=[tau], rtol=1e-8, atol=1e-10)
    return vec2hermitian(sol.y[:,-1])

def evolve_two_ancillas(rho0_S, lam):
    rhoA12    = build_rhoA1A2(lam)
    rhoA1_eff = partial_trace_2(rhoA12)
    rho_S_mid = evolve_step(rho0_S, rhoA1_eff)
    rhoA2_eff = partial_trace_1(rhoA12).copy()
    for sigma in paulis:
        corr = lam * np.trace(sigma @ rho_S_mid).real
        rhoA2_eff += corr * sigma
    evals, evecs = np.linalg.eigh(rhoA2_eff)
    evals = np.maximum(evals, 0)
    rhoA2_eff = evecs @ np.diag(evals) @ evecs.conj().T
    rhoA2_eff /= np.trace(rhoA2_eff)
    rho_S_final = evolve_step(rho_S_mid, rhoA2_eff)
    return rho_S_final

def evolve_one_ancilla(rho0_S, lam):
    chiA = lam * sx_n
    def meq(t, rv):
        rho = vec2hermitian(rv)
        D = (dissipator_entry(g, sm, sp, rho, rhoA_th) +
             dissipator_entry(g, sp, sm, rho, rhoA_th))
        C = (coherent_entry(g, sm, sp, rho, chiA) +
             coherent_entry(g, sp, sm, rho, chiA))
        return hermitian2vec(-1j*commutator(Hs, rho) + D + C)
    sol = solve_ivp(meq, [0, tau],
                    np.array(hermitian2vec(rho0_S), dtype=complex),
                    t_eval=[tau], rtol=1e-8, atol=1e-10)
    return vec2hermitian(sol.y[:,-1])

# ================================================================
# PARTE 3 — TPM DENTRO DO REGIME FÍSICO VÁLIDO
# ================================================================
Z_S  = 1 + np.exp(-beta*Omega)
pg_S = 1.0 / Z_S
pe_S = np.exp(-beta*Omega) / Z_S

g_state = np.array([[1],[0]], dtype=complex)
e_state = np.array([[0],[1]], dtype=complex)
rho_g   = g_state @ dagger(g_state)
rho_e   = e_state @ dagger(e_state)

# Varredura contínua até λ_max
lam_arr   = np.linspace(0, lam_max, 5)
lam_pts   = [l for l in [0.0, 0.02, 0.05, 0.08] if l <= lam_max]

Sigma_2anc = []
Delta_2anc = []
rho11_2anc = []
Sigma_1anc = []
Delta_1anc = []

print("\n=== TPM — regime físico válido (λ ≤ λ_max) ===")
for lam in lam_arr:
    # 2 ancillas
    rg2 = evolve_two_ancillas(rho_g, lam)
    re2 = evolve_two_ancillas(rho_e, lam)
    P_ge2 = np.real((dagger(e_state) @ rg2 @ e_state)[0,0])
    P_eg2 = np.real((dagger(g_state) @ re2 @ g_state)[0,0])
    P_ge2 = float(np.real(P_ge2))
    P_eg2 = float(np.real(P_eg2))
    S2    = np.log(pg_S*P_ge2/pe_S/P_eg2 + 1e-14)
    Sigma_2anc.append(S2)
    Delta_2anc.append(S2 - 2*beta*Omega)
    rho11_2anc.append(float(np.real(rg2[0,0])))

    # 1 ancilla (referência)
    rg1 = evolve_one_ancilla(rho_g, lam)
    re1 = evolve_one_ancilla(rho_e, lam)
    P_ge1 = float(np.real((dagger(e_state) @ rg1 @ e_state)[0,0]))
    P_eg1 = float(np.real((dagger(g_state) @ re1 @ g_state)[0,0]))
    S1    = np.log(pg_S*P_ge1/pe_S/P_eg1 + 1e-14)
    Sigma_1anc.append(S1)
    Delta_1anc.append(S1 - 2*beta*Omega)

# ================================================================
# PARTE 4 — PLOTS
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Σ vs λ
ax = axes[0]
ax.plot(lam_arr, Sigma_2anc, '-', color='steelblue', linewidth=2,
        label='2 ancillas correlacionadas')
ax.plot(lam_arr, Sigma_1anc, '--', color='coral', linewidth=2,
        label='1 ancilla (referência)')
ax.axvline(lam_max, color='gray', linestyle=':', linewidth=1.5,
           label=f'λ_max = {lam_max:.3f}')
ax.axhline(2*beta*Omega, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\Sigma$', fontsize=14)
ax.set_title(r'Produção de entropia $\Sigma$ vs. $\lambda$' +
             f'\n(regime físico: λ ≤ {lam_max:.3f})', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Δχ − Δχ(0) vs λ
ax = axes[1]
d0_2 = Delta_2anc[0]
d0_1 = Delta_1anc[0]
ax.plot(lam_arr, [d - d0_2 for d in Delta_2anc], '-', color='steelblue',
        linewidth=2, label='2 ancillas correlacionadas')
ax.plot(lam_arr, [d - d0_1 for d in Delta_1anc], '--', color='coral',
        linewidth=2, label='1 ancilla (referência)')
ax.axvline(lam_max, color='gray', linestyle=':', linewidth=1.5,
           label=f'λ_max = {lam_max:.3f}')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\Delta\chi - \Delta\chi(\lambda=0)$', fontsize=13)
ax.set_title(r'Contribuição coerente $\Delta\chi$ (offset subtraído)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: ρ₁₁_ss vs λ
ax = axes[2]
ax.plot(lam_arr, rho11_2anc, '-', color='steelblue', linewidth=2,
        label='2 ancillas correlacionadas')
ax.axhline(pg_S, color='gray', linestyle=':', linewidth=1,
           label=f'$p_g$ térmico = {pg_S:.3f}')
ax.axvline(lam_max, color='gray', linestyle=':', linewidth=1.5,
           label=f'λ_max = {lam_max:.3f}')
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\rho_{11}^{ss}$', fontsize=14)
ax.set_title(r'Estado estacionário $\rho_{11}^{ss}$ vs. $\lambda$', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Duas ancillas correlacionadas — regime físico válido',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('duas_ancilas_corrigido.png', dpi=150)
plt.show()
print(f"\nGráfico salvo. λ_max = {lam_max:.4f}")