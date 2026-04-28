# %% [markdown]
# # Steepest Descent vs Conjugate Gradient on a 2D Quadratic Form
#
# This script compares how steepest descent and conjugate gradient navigate
# the contours of a 2D quadratic function $f(x) = \frac{1}{2} x^T A x - b^T x$.
#
# **Key observations:**
# - Steepest descent zigzags along the gradient directions
# - CG uses A-orthogonal (conjugate) directions and converges in at most n steps
#
# Both methods minimize the same function, but CG chooses *search directions*
# that don't interfere with each other.

# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Ensure figures directory exists
fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'slides', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# ## Define the Quadratic Problem
#
# We construct a 2×2 symmetric positive definite matrix A with condition number
# $\kappa \approx 10$. The optimal solution is $x^* = A^{-1}b = (1, 1)$.

# %%
# Build a 2x2 SPD matrix with controllable condition number
np.random.seed(42)
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta),  np.cos(theta)]])
# Eigenvalues: λ1=1, λ2=10 → condition number κ = 10
Lambda = np.diag([1.0, 10.0])
A = R @ Lambda @ R.T  # A is SPD, condition number ~10

b = A @ np.array([1.0, 1.0])  # so that x* = (1,1)
x0 = np.array([-1.5, 1.8])    # initial guess

# Quadratic function f(x) = 0.5 * x^T A x - b^T x
def f(x, y):
    v = np.array([x, y])
    return 0.5 * v @ A @ v - b @ v

# Gradient ∇f(x) = A x - b = -(b - A x) = -residual
def grad(v):
    return A @ v - b

# %% [markdown]
# ## Steepest Descent
#
# At each step $k$:
# - Search direction: $p_k = -\nabla f(x_k) = r_k$
# - Step size: $\alpha_k = \frac{r_k^T r_k}{r_k^T A r_k}$
# - Update: $x_{k+1} = x_k + \alpha_k p_k$

# %%
def steepest_descent(A, b, x0, max_iter=30, tol=1e-8):
    """Steepest descent with exact line search for Ax = b."""
    x = x0.copy()
    trajectory = [x.copy()]
    r = b - A @ x  # residual (negative gradient)

    for _ in range(max_iter):
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)  # exact line search
        x = x + alpha * r
        trajectory.append(x.copy())
        r = b - A @ x
        if np.linalg.norm(r) < tol:
            break
    return np.array(trajectory)

# %% [markdown]
# ## Conjugate Gradient
#
# At each step $k$:
# - Search direction $p_k$ is A-orthogonal to all previous directions
# - $p_0 = r_0$ (initial residual)
# - $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$
# - $x_{k+1} = x_k + \alpha_k p_k$
# - $r_{k+1} = r_k - \alpha_k A p_k$
# - $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$
# - $p_{k+1} = r_{k+1} + \beta_k p_k$

# %%
def conjugate_gradient(A, b, x0, max_iter=30, tol=1e-8):
    """Conjugate Gradient for solving Ax = b with A SPD."""
    x = x0.copy()
    trajectory = [x.copy()]
    r = b - A @ x
    p = r.copy()

    for _ in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        trajectory.append(x.copy())
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        if np.linalg.norm(r) < tol:
            break
    return np.array(trajectory)

# %%
# Run both algorithms from the same starting point
traj_sd = steepest_descent(A, b, x0, max_iter=20)
traj_cg = conjugate_gradient(A, b, x0, max_iter=20)

print(f"Steepest descent took {len(traj_sd)-1} iterations")
print(f"Conjugate gradient took {len(traj_cg)-1} iterations")
print(f"CG converges in at most n={A.shape[0]} steps for exact arithmetic!")

# %% [markdown]
# ## Plot Side-by-Side Comparison

# %%
# Create contour grid
x_range = np.linspace(-2.2, 2.2, 200)
y_range = np.linspace(-1.0, 2.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = np.vectorize(lambda xv, yv: f(xv, yv))(X, Y)

# Exponential contour levels for better visualization of the quadratic
levels = np.logspace(np.log10(Z.min() - Z.min() + 1e-10),
                      np.log10(Z.max() + 1e-10), 20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Steepest Descent ---
ax1.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.5, linewidths=0.5)
ax1.plot(traj_sd[:, 0], traj_sd[:, 1], 'o-', color='#e74c3c', linewidth=2,
         markersize=7, markerfacecolor='white', markeredgewidth=2, label='SD iterates')
ax1.plot(traj_sd[0, 0], traj_sd[0, 1], 's', color='black', markersize=10, label='Start $x_0$')
ax1.plot(1, 1, '*', color='#2ecc71', markersize=15, markeredgewidth=1.5,
         label='Optimum $x^*$')
ax1.set_xlabel('$x_1$', fontsize=12)
ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title('Steepest Descent\n(zigzag in narrow valley)', fontsize=14)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# --- Conjugate Gradient ---
ax2.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.5, linewidths=0.5)
ax2.plot(traj_cg[:, 0], traj_cg[:, 1], 'o-', color='#3498db', linewidth=2,
         markersize=7, markerfacecolor='white', markeredgewidth=2, label='CG iterates')
ax2.plot(traj_cg[0, 0], traj_cg[0, 1], 's', color='black', markersize=10, label='Start $x_0$')
ax2.plot(1, 1, '*', color='#2ecc71', markersize=15, markeredgewidth=1.5,
         label='Optimum $x^*$')
ax2.set_xlabel('$x_1$', fontsize=12)
ax2.set_ylabel('$x_2$', fontsize=12)
ax2.set_title('Conjugate Gradient\n(direct path in 2 steps)', fontsize=14)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.suptitle('Steepest Descent vs Conjugate Gradient\n'
             r'Minimizing $f(x) = \frac{1}{2}x^T A x - b^T x$ with $\kappa(A) \approx 10$',
             fontsize=15, y=1.01)
plt.tight_layout()

outpath = os.path.join(fig_dir, 'steepest_descent_vs_cg.pdf')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Figure saved to {outpath}")
plt.show()

# %% [markdown]
# ## Error Norm Comparison
#
# We also plot the error $\|x_k - x^*\|_2$ vs iteration for both methods
# to show the speed difference quantitatively.

# %%
x_star = np.array([1.0, 1.0])
errors_sd = np.linalg.norm(traj_sd - x_star, axis=1)
errors_cg = np.linalg.norm(traj_cg - x_star, axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(errors_sd, 'o-', color='#e74c3c', linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, label='Steepest Descent')
ax.semilogy(errors_cg, 's-', color='#3498db', linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, label='Conjugate Gradient')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel(r'Error $\|x_k - x^*\|_2$', fontsize=12)
ax.set_title('Convergence Speed: SD vs CG', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

outpath2 = os.path.join(fig_dir, 'sd_vs_cg_error.pdf')
plt.savefig(outpath2, dpi=150, bbox_inches='tight')
print(f"Figure saved to {outpath2}")
plt.show()
