"""
2D Level-Set PINN Proof of Concept
==================================
Solves the level-set equation

    phi_t + R * |grad(phi)| = 0,   (x, y, t) in [0,1]^2 x [0, T]

with constant spread rate R and a circular ignition:

    IC: phi(x, y, 0) = sqrt((x-x0)^2 + (y-y0)^2) - r0   (signed distance)

Analytical solution (R constant): the front stays a circle of radius
r(t) = r0 + R*t, and phi remains a signed distance function:

    phi_true(x, y, t) = sqrt((x-x0)^2 + (y-y0)^2) - (r0 + R*t).

No boundary-condition loss is used: r(T) = 0.4 < 0.5, the distance from
the ignition center to the domain edge, so the front never reaches the
boundary during the simulation.

Only phi is learned (one PINN). R is a fixed constant here -- the
computed-input side of the design, standing in for the Rothermel raster
the real model will consume.

Run:  python levelset_pinn_poc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Reproducibility ----------------
np.random.seed(0)
tf.random.set_seed(0)

# ---------------- PoC parameters ----------------
R = 0.5              # constant rate of spread (model units)
X0, Y0 = 0.5, 0.5    # ignition center
R0 = 0.1             # ignition radius
T_FINAL = 0.6        # final time -> front radius r0 + R*T = 0.4
W_IC = 10.0          # IC loss weight (mirrors the 1D 10x baseline)

N_COLLOC = 2000      # interior collocation points (fixed set, cf. 1D code)
N_IC = 500           # initial-condition points at t = 0
EPOCHS = 10000
LR = 1e-3


# ---------------- Model ----------------
def create_pinn_model():
    model = keras.Sequential()
    model.add(layers.Input(shape=(3,)))  # [x, y, t]
    model.add(layers.Dense(50, activation="tanh"))
    model.add(layers.Dense(50, activation="tanh"))
    model.add(layers.Dense(1))           # phi(x, y, t)
    return model


# ---------------- Physics ----------------
def pde_residual(model, x, y, t):
    """r = phi_t + R*|grad(phi)| at the given points.

    Only first derivatives are needed (cheaper than the 1D heat code's
    second derivative). tf.gradients does not exist in TF2 -- use a
    persistent GradientTape and watch the input tensors explicitly.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        tape.watch(t)
        phi = model(tf.concat([x, y, t], axis=1))
    phi_t = tape.gradient(phi, t)
    phi_x = tape.gradient(phi, x)
    phi_y = tape.gradient(phi, y)
    del tape  # persistent tapes must be freed manually
    grad_mag = tf.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)
    return phi_t + R * grad_mag


def signed_distance_to_ignition(x, y):
    """Analytical IC: signed distance to the ignition circle."""
    return tf.sqrt((x - X0) ** 2 + (y - Y0) ** 2) - R0


# ---------------- Data ----------------
def generate_collocation(n):
    x = tf.random.uniform((n, 1), 0, 1)
    y = tf.random.uniform((n, 1), 0, 1)
    t = tf.random.uniform((n, 1), 0, T_FINAL)
    return x, y, t


def generate_ic_points(n):
    x = tf.random.uniform((n, 1), 0, 1)
    y = tf.random.uniform((n, 1), 0, 1)
    t = tf.zeros((n, 1))
    return x, y, t


# ---------------- Training ----------------
def train_pinn(model, x_c, y_c, t_c, x_ic, y_ic, t_ic, epochs=EPOCHS):
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    hist = {"total": [], "pde": [], "ic": []}

    for epoch in range(epochs):
        with tf.GradientTape() as tape:  # outer tape: the Adam step
            r = pde_residual(model, x_c, y_c, t_c)  # inner tape: the PDE
            loss_pde = tf.reduce_mean(tf.square(r))

            phi_ic = model(tf.concat([x_ic, y_ic, t_ic], axis=1))
            loss_ic = tf.reduce_mean(
                tf.square(phi_ic - signed_distance_to_ignition(x_ic, y_ic)))

            loss = loss_pde + W_IC * loss_ic

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        hist["total"].append(float(loss))
        hist["pde"].append(float(loss_pde))
        hist["ic"].append(float(loss_ic))

        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d}  total={float(loss):.3e}  "
                  f"pde={float(loss_pde):.3e}  ic={float(loss_ic):.3e}")

    return hist


# ---------------- Validation (the PoC's actual deliverable) ----------------
def analytical_phi(x, y, t):
    return np.sqrt((x - X0) ** 2 + (y - Y0) ** 2) - (R0 + R * t)


def relative_l2_error(model, n_grid=101, times=(0.0, 0.2, 0.4, 0.6)):
    """Relative L2 of phi vs the analytical expanding circle, pooled over times."""
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(xs, ys)
    num, den = 0.0, 0.0
    for t in times:
        pts = np.stack([X.ravel(), Y.ravel(),
                        np.full(X.size, t)], axis=1).astype(np.float32)
        phi_pred = model.predict(pts, verbose=0).reshape(X.shape)
        phi_true = analytical_phi(X, Y, t)
        num += np.sum((phi_pred - phi_true) ** 2)
        den += np.sum(phi_true ** 2)
    return float(np.sqrt(num / den))


def plot_fronts(model, times=(0.0, 0.2, 0.4, 0.6),
                save="pinn_front_snapshots.png"):
    """Predicted phi=0 contour (red) vs analytical circle (black dashed)."""
    xs = np.linspace(0, 1, 201)
    ys = np.linspace(0, 1, 201)
    X, Y = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
    for ax, t in zip(axes, times):
        pts = np.stack([X.ravel(), Y.ravel(),
                        np.full(X.size, t)], axis=1).astype(np.float32)
        PHI = model.predict(pts, verbose=0).reshape(X.shape)
        ax.contour(X, Y, PHI, levels=[0], colors="red", linewidths=2)
        ax.add_patch(plt.Circle((X0, Y0), R0 + R * t, fill=False,
                                color="black", linestyle="--"))
        ax.set_aspect("equal")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Level-set PINN: predicted front (red) vs analytical (dashed)")
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.close(fig)


def plot_losses(hist, save="loss_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = np.arange(len(hist["total"]))
    ax.plot(ep, hist["total"], label="total")
    ax.plot(ep, hist["pde"], label="PDE residual")
    ax.plot(ep, hist["ic"], label="IC")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training losses (2D level-set PINN)")
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.close(fig)


# ---------------- Main ----------------
if __name__ == "__main__":
    x_c, y_c, t_c = generate_collocation(N_COLLOC)
    x_ic, y_ic, t_ic = generate_ic_points(N_IC)

    model = create_pinn_model()
    hist = train_pinn(model, x_c, y_c, t_c, x_ic, y_ic, t_ic)

    err = relative_l2_error(model)
    print(f"\nrelative L2 error vs analytical solution: "
          f"{err:.4f} ({100 * err:.2f}%)")

    plot_fronts(model)
    plot_losses(hist)
    print("wrote pinn_front_snapshots.png and loss_curves.png")
