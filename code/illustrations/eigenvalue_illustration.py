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
# **$A$-norm** satisfies:
#
# $$\frac{\|x_k - x^*\|_A}{\|x_0 - x^*\|_A} \leq
#   2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$$
#
# where $\kappa = \lambda_{\max} / \lambda_{\min}$ is the condition number.
#
# **IMPORTANT:** This bound is for the *A-norm error*, NOT the residual norm.
# The A-norm error $\|x_k - x^*\|_A = \sqrt{(x_k - x^*)^T A (x_k - x^*)}$
# measures energy-weighted distance to the solution.
#
# If eigenvalues are well-clustered, CG can converge *much faster*
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
#
# This implementation tracks BOTH the residual norm AND the A-norm error
# so we can compare the theoretical bound against the correct quantity.

# %%
def conjugate_gradient(A, b, max_iter=100, tol=1e-10):
    """
    CG for Ax = b with A SPD.

    Returns:
    - residuals: residual norm ||r_k|| at each iteration
    - a_norm_errors: A-norm error ||x_k - x*||_A at each iteration
      (the quantity the theoretical bound actually applies to)
    """
    n = len(b)
    # Ground truth solution for A-norm error computation
    x_star = np.linalg.solve(A, b)
    A_norm_initial_sq = (x_star) @ A @ (x_star)  # since x0 = 0

    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    residuals = [np.linalg.norm(r)]
    a_norm_errors = [np.sqrt(A_norm_initial_sq)]  # ||0 - x*||_A = sqrt(x*^T A x*)

    for k in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap

        # Residual norm
        residuals.append(np.linalg.norm(r_new))

        # A-norm error: ||x_k - x*||_A
        err = x - x_star
        a_norm_err = np.sqrt(err @ A @ err)
        a_norm_errors.append(a_norm_err)

        if residuals[-1] < tol:
            break

        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    return np.array(residuals), np.array(a_norm_errors)

# %%
# Random right-hand side
rng = np.random.default_rng(42)
b1 = rng.standard_normal(n)
b2 = b1.copy()  # same b for fair comparison

residuals1, a_errs1 = conjugate_gradient(A1, b1, max_iter=80)
residuals2, a_errs2 = conjugate_gradient(A2, b2, max_iter=80)

# Also compute the theoretical bound for comparison
# NOTE: This bound applies to the A-norm error, NOT the residual.
sqrt_kappa = np.sqrt(kappa)
bound_factor = (sqrt_kappa - 1) / (sqrt_kappa + 1)
k_vals = np.arange(len(a_errs1))
theory_bound = 2 * bound_factor**k_vals * a_errs1[0]

# %% [markdown]
# ## Plot: Eigenvalue Distributions and Convergence Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Eigenvalue distribution: Uniform ---
ax = axes[0, 0]
markerline, stemlines, baseline = ax.stem(lambda1, np.ones(n),
                                          linefmt='C3-', markerfmt='o',
                                          basefmt=' ')
plt.setp(stemlines, color='#e74c3c')
plt.setp(markerline, markerfacecolor='#e74c3c', markeredgecolor='#e74c3c')
ax.set_xlabel(r'Eigenvalue $\lambda$', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(r'Uniform Eigenvalue Distribution\\$(\kappa = 100)$', fontsize=13)
ax.set_ylim(0, 1.5)
ax.grid(True, alpha=0.3)

# --- Eigenvalue distribution: Clustered ---
ax = axes[0, 1]
markerline2, stemlines2, baseline2 = ax.stem(lambda2, np.ones(n),
                                              linefmt='C0-', markerfmt='o',
                                              basefmt=' ')
plt.setp(stemlines2, color='#3498db')
plt.setp(markerline2, markerfacecolor='#3498db', markeredgecolor='#3498db')
ax.set_xlabel(r'Eigenvalue $\lambda$', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(r'Clustered Eigenvalue Distribution\\$(\kappa = 100)$', fontsize=13)
ax.set_ylim(0, 1.5)
ax.grid(True, alpha=0.3)

# --- Convergence: A-norm error (semilog plot) ---
ax = axes[1, 0]
ax.semilogy(a_errs1 / a_errs1[0], linewidth=2, color='#e74c3c',
            label='Uniform eigenvalues')
ax.semilogy(a_errs2 / a_errs2[0], linewidth=2, color='#3498db',
            label='Clustered eigenvalues')
ax.semilogy(theory_bound / a_errs1[0], '--', linewidth=1.5, color='gray',
            label=r'Worst-case bound $2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$')
ax.set_xlabel(r'Iteration $k$', fontsize=11)
ax.set_ylabel(r'Relative $A$-norm Error $\|x_k - x^*\|_A / \|x_0 - x^*\|_A$', fontsize=11)
ax.set_title(r'CG Convergence: Same $\kappa$, Different Distributions', fontsize=13)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# --- Zoomed: first 30 iterations ---
ax = axes[1, 1]
ax.semilogy(a_errs1[:30] / a_errs1[0], linewidth=2, color='#e74c3c',
            label='Uniform eigenvalues')
ax.semilogy(a_errs2[:30] / a_errs2[0], linewidth=2, color='#3498db',
            label='Clustered eigenvalues')
ax.semilogy(theory_bound[:30] / a_errs1[0], '--', linewidth=1.5, color='gray',
            label=r'Worst-case bound $2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$')
ax.set_xlabel(r'Iteration $k$', fontsize=11)
ax.set_ylabel(r'Relative $A$-norm Error $\|x_k - x^*\|_A / \|x_0 - x^*\|_A$', fontsize=11)
ax.set_title('First 30 Iterations (Zoom)', fontsize=13)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.suptitle(
    r'Eigenvalue Distribution Effect on CG Convergence' + '\n'
    r'Both matrices have $\kappa = 100$, but clustered eigenvalues converge much faster'
    + '\n' + r'Error measured in $A$-norm $\|x_k - x^*\|_A$ = theoretical bound quantity',
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
