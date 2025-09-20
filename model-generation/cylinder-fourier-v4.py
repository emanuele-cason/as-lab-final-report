import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import pyvista as pv
import time
import sys
import os

# Geometric parameters
L = 0.5  # lenght (axial)
R = 0.25  # Radius
t = 0.0005  # Deformations scale factor (equal wall thickness, see final report)

ntheta = 196  # Discretization along circumference (number of nodes)
nx = 63  # Discretization along axis (number of nodes)

x = np.linspace(0, L, nx)
theta = np.linspace(0, 2 * np.pi, ntheta)
X_grid, Theta_grid = np.meshgrid(x, theta)

# Gaussian matrix construction, centered in i0, j0 (prevalent mode)
N = 10  # matrix size
i0, j0 = 5, 7
sigma_i = sigma_j = 1.3
random_global_weight = 0
random_local_weight = 0
scheme = "slope_equal"

I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
G_gauss = np.exp(-((I - i0) ** 2 / (2 * sigma_i**2) + (J - j0) ** 2 / (2 * sigma_j**2)))

rng = np.random.default_rng(int(time.time()))
G_rand_global = rng.standard_normal((N, N))
sign_matrix = np.sign(G_rand_global)

rng = np.random.default_rng(int(time.time()))
G_rand_local = rng.uniform(
    low=1 - random_local_weight, high=1 + random_local_weight, size=(N, N)
)

G = sign_matrix * (
    random_global_weight * np.abs(G_rand_global)
    + (1 - random_global_weight) * np.abs(G_gauss) * G_rand_local
)

S = np.zeros((N, N), dtype=float)
for k in range(N):
    kx = k / (2 * L)
    for l in range(N):
        kt = l / (2 * np.pi * R)
        q2 = kx**2 + kt**2
        if scheme == "displacement_equal":
            s = 1.0
        elif scheme == "slope_equal":
            s = 1.0 / np.sqrt(q2) if q2 > 0 else 0.0
        elif scheme == "energy_equal":
            s = 1.0 / q2 if q2 > 0 else 0.0
        S[k, l] = s
S[0, 0] = 0.0
# S = S * 1 / np.max(np.abs(S))
S = S * 1 / (np.mean(np.abs(S[:10, :10])))

A = np.random.rand(*G.shape) * G
B = G - A

A = A * S
B = B * S

# Matrix plot
fig_matrix, ax = plt.subplots()
x_edges = np.arange(N + 1)
y_edges = np.arange(N + 1)
c = ax.pcolormesh(x_edges, y_edges, G, cmap="viridis", shading="auto")
fig_matrix.colorbar(c, ax=ax, label="Coefficient value")
ax.set_xticks(np.arange(N) + 0.5)
ax.set_yticks(np.arange(N) + 0.5)
ax.set_xticklabels(np.arange(N))
ax.set_yticklabels(np.arange(N))
ax.set_xticks(np.arange(N + 1), minor=True)
ax.set_yticks(np.arange(N + 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
ax.invert_yaxis()
plt.show()

print("\nMatrix A")
print(np.round(A, 3))
print("\n\nMatrix S")
print(np.round(S, 3))

# Computation of the radial deformation field: w (x, theta)
W = np.zeros_like(X_grid)
for k in range(N):
    for l in range(N):
        W += np.cos(k * np.pi * X_grid / L) * (
            A[k, l] * np.cos(l * Theta_grid) + B[k, l] * np.sin(l * Theta_grid)
        )
W *= t

R_deformed = R + W
Y = R_deformed * np.cos(Theta_grid)
Z = R_deformed * np.sin(Theta_grid)

# Lateral deformed surface plot
fig_dfield, ax = plt.subplots(figsize=(8, 4))
Theta_length = Theta_grid * R
c = ax.pcolormesh(Theta_length, X_grid, W * 1000, cmap="viridis", shading="auto")
cbar = fig_dfield.colorbar(
    c, ax=ax, orientation="horizontal", fraction=0.08, pad=0.15, aspect=40
)
cbar.set_label("Deformation field [mm]")
ax.set_xlabel("Circumference [m]")
ax.xaxis.set_label_position("top")
ax.tick_params(
    axis="x", which="both", bottom=False, top=True, labelbottom=False, labeltop=True
)
ax.set_ylabel("Axis [m]")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# Mesh generation (.STL)
faces = []
for i in range(ntheta - 1):
    for j in range(nx - 1):
        p1 = [Y[i, j], Z[i, j], X_grid[i, j]]
        p2 = [Y[i + 1, j], Z[i + 1, j], X_grid[i + 1, j]]
        p3 = [Y[i + 1, j + 1], Z[i + 1, j + 1], X_grid[i + 1, j + 1]]
        p4 = [Y[i, j + 1], Z[i, j + 1], X_grid[i, j + 1]]
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

faces = np.array(faces)
cylinder_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    cylinder_mesh.vectors[i] = f

file_base = input("Insert filename (no extension): ").strip()
if file_base:

    os.makedirs(file_base, exist_ok=True)
    stl_fname = os.path.join(file_base, f"{file_base}.stl")
    log_name = os.path.join(file_base, f"{file_base}.txt")
    matrix_img_fname = os.path.join(file_base, f"{file_base}-matrix.jpg")
    dfield_img_fname = os.path.join(file_base, f"{file_base}-dfield.jpg")

    cylinder_mesh.save(stl_fname)
    print(f"STL file exported: {stl_fname}")

    # Salva immagini plot
    fig_matrix.savefig(matrix_img_fname, dpi=300, bbox_inches="tight", pad_inches=0)
    fig_dfield.savefig(dfield_img_fname, dpi=300, bbox_inches="tight", pad_inches=0)

    # Logging parametri e matrici
    with open(log_name, "w") as f:
        f.write("=== Parameters ===\n")
        f.write(f"L = {L}\nR = {R}\nt = {t}\n")
        f.write(f"ntheta = {ntheta}\nnx = {nx}\n")
        f.write(
            f"N = {N}, sigma = {sigma_i}, random_global_weight = {random_global_weight}\n"
        )
        f.write(f"i0 = {i0}, j0 = {j0}\n\n")
        f.write("=== MATRIX A ===\n")
        np.savetxt(f, A, fmt="%12.6f", delimiter="   ")
        f.write("\n\n=== MATRIX B ===\n")
        np.savetxt(f, B, fmt="%12.6f", delimiter="   ")
        f.write("\n\n=== MATRIX S ===\n")
        np.savetxt(f, S, fmt="%5.1e", delimiter="   ")
        f.write("\n=== MATRIX W ===\n")
        f.write("\n=== Stats ===\n")
        f.write(f"sum = {np.sum(W)}\n")
        f.write(f"min = {np.min(W)}\n")
        f.write(f"max = {np.max(W)}\n")
        f.write(f"mean = {np.mean(W)}\n")
        f.write(f"max-min = {np.abs(np.max(W) - np.min(W))}\n")
        f.write(f"rms = {np.sqrt(np.mean(W**2))}\n")
        f.write("\n=== Raw data ===\n")
        np.savetxt(f, W, fmt="%.5f")
    print(f"Log file saved: {log_name}")
else:
    print("No file saved.")

# 3D plot
if file_base:
    pv_mesh = pv.read(stl_fname)
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv_mesh,
        scalars=W.ravel(order="C"),
        cmap="viridis",
        show_edges=False,
        smooth_shading=True,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

if file_base:
    mesh_loaded = pv.read(stl_fname)
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh_loaded, color="lightblue", show_edges=True, edge_color="black"
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
