# %% [markdown]
# # Eigenvalue Distribution and CG Convergence
#
# This script illustrates how the distribution of eigenvalues of the system matrix
# $A$ affects the convergence rate of the Conjugate Gradient method.
#
# **Key idea:** The standard convergence bound depends on the condition number
# $\kappa(A) = \lambda_{\max} / \lambda_{\min}$, but the *actual* convergence
# depends on the full eigenvalue distribution. Clustered eigenvalues lead
# to faster convergence.

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'slides', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# ## Theoretical Bound
#
# For Conjugate Gradient applied to $A x = b$ with $A$ SPD, the error in the
# $A$-norm satisfies:
#
# $$\frac{\|x_k - x^*\|_A}{\|x_0 - x^*\|_A} \leq
#   2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$$
#
# where $\kappa = \lambda_{\max} / \lambda_{\min}$ is the condition number.
#
# However, if eigenvalues are well-clustered, CG can converge *much faster*
# than this worst-case bound.

# %%
# Define two matrices with the SAME condition number but DIFFERENT
# eigenvalue distributions

np.random.seed(123)

# Matrix 1: Uniformly distributed eigenvalues in [1, 100]
# κ = 100 in both cases
n = 50
lambda1 = np.linspace(1, 100, n)  # evenly spread

# Matrix 2: Clustered eigenvalues — 45 eigenvalues in [1, 2], 5 in [95, 100]
lambda2 = np.concatenate([
    np.linspace(1, 2, n - 5),
    np.linspace(95, 100, 5)
])

kappa = lambda1[-1] / lambda1[0]
print(f"Condition number: κ = {kappa:.1f}")
print(f"Matrix 1 eigenvalues: {n} values from 1 to 100 (uniform)")
print(f"Matrix 2 eigenvalues: {n-5} values in [1,2] + 5 values in [95,100] (clustered)")

# %% [markdown]
# ## Build Random SPD Matrices with Specified Eigenvalues
#
# We use a random orthogonal matrix Q and set A = Q Λ Q^T.

# %%
def build_spd_matrix(eigenvalues, seed=42):
    """Build a random SPD matrix with specified eigenvalues."""
    n = len(eigenvalues)
    rng = np.random.default_rng(seed)
    # Generate a random orthogonal matrix via QR decomposition
    H = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(H)
    return Q @ np.diag(eigenvalues) @ Q.T

A1 = build_spd_matrix(lambda1, seed=42)  # uniform eigenvalues
A2 = build_spd_matrix(lambda2, seed=42)  # clustered eigenvalues

# %% [markdown]
# ## Conjugate Gradient Implementation

# %%
def conjugate_gradient(A, b, max_iter=100, tol=1e-10):
    """CG for Ax = b with A SPD. Returns error history."""
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    errors = [np.sqrt(r @ r)]  # residual norm

    for k in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        errors.append(np.linalg.norm(r_new))
        if errors[-1] < tol:
            break
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    return np.array(errors)

# %%
# Random right-hand side
b1 = np.random.randn(n)
b2 = b1.copy()  # same b for fair comparison

errors1 = conjugate_gradient(A1, b1, max_iter=80)
errors2 = conjugate_gradient(A2, b2, max_iter=80)

# Also compute the theoretical bound for comparison
sqrt_kappa = np.sqrt(kappa)
bound_factor = (sqrt_kappa - 1) / (sqrt_kappa + 1)
k_vals = np.arange(len(errors1))
theory_bound = 2 * bound_factor**k_vals * errors1[0]

# %% [markdown]
# ## Plot: Eigenvalue Distributions and Convergence Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Eigenvalue distribution: Uniform ---
ax = axes[0, 0]
ax.stem(lambda1, np.ones(n), linefmt='#e74c3c-', markerfmt='o', basefmt=' ')
ax.set_xlabel('Eigenvalue $\lambda$', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Uniform Eigenvalue Distribution\n$(\\kappa = 100)$', fontsize=13)
ax.set_ylim(0, 1.5)
ax.grid(True, alpha=0.3)

# --- Eigenvalue distribution: Clustered ---
ax = axes[0, 1]
ax.stem(lambda2, np.ones(n), linefmt='#3498db-', markerfmt='o', basefmt=' ')
ax.set_xlabel('Eigenvalue $\lambda$', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Clustered Eigenvalue Distribution\n$(\\kappa = 100)$', fontsize=13)
ax.set_ylim(0, 1.5)
ax.grid(True, alpha=0.3)

# --- Convergence: semilog plot ---
ax = axes[1, 0]
ax.semilogy(errors1 / errors1[0], linewidth=2, color='#e74c3c',
            label='Uniform eigenvalues')
ax.semilogy(errors2 / errors2[0], linewidth=2, color='#3498db',
            label='Clustered eigenvalues')
ax.semilogy(theory_bound / errors1[0], '--', linewidth=1.5, color='gray',
            label='Worst-case bound')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel(r'Relative Residual $\|r_k\| / \|r_0\|$', fontsize=11)
ax.set_title('CG Convergence: Same $\kappa$, Different Distributions', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# --- Zoomed: first 30 iterations ---
ax = axes[1, 1]
ax.semilogy(errors1[:30] / errors1[0], linewidth=2, color='#e74c3c',
            label='Uniform eigenvalues')
ax.semilogy(errors2[:30] / errors2[0], linewidth=2, color='#3498db',
            label='Clustered eigenvalues')
ax.semilogy(theory_bound[:30] / errors1[0], '--', linewidth=1.5, color='gray',
            label='Worst-case bound')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel(r'Relative Residual $\|r_k\| / \|r_0\|$', fontsize=11)
ax.set_title('First 30 Iterations (Zoom)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.suptitle(
    'Eigenvalue Distribution Effect on CG Convergence\n'
    'Both matrices have $\kappa = 100$, but clustered eigenvalues converge much faster',
    fontsize=15, y=1.01)
plt.tight_layout()

outpath = os.path.join(fig_dir, 'eigenvalue_illustration.pdf')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Figure saved to {outpath}")
plt.show()

# %% [markdown]
# ## Why Does This Matter for MRI?
#
# In multi-coil MRI reconstruction (SENSE), the normal equations matrix is
# $A = \sum_i S_i^H F^H M F S_i$. The eigenvalue distribution depends on:
#
# 1. **Coil geometry**: Well-separated coils with distinct sensitivity profiles
#    produce a better-conditioned system,
# 2. **Undersampling factor** $R$: Higher $R$ worsens conditioning,
# 3. **Regularization**: Adding $\lambda I$ shifts all eigenvalues by $\lambda$,
#    improving conditioning at the cost of slight bias.
#
# This is why CG works well for SENSE — the eigenvalues of the normal equations
# are often well-clustered due to the physics of coil sensitivity profiles.
