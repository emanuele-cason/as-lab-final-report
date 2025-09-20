import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from tkinter import filedialog, Tk
import pyvista as pv

# User inputs
print("")
print("Cylinder and dimple parameters: ")
Rc = float(input("Radius of the cylinder Rc [meters]: "))
H = float(input("Heigth of the cylinder H [meters]: "))
a_len = float(
    input("Amplitude of the dimple along circumferential direction a [meters]: ")
)
b = float(input("Amplitude of the dimple along axial direction b [meters]: "))
wb = float(input("Maximum depth of the dimple wb [meters]: "))
z_d = float(input("Vertical position of the dimple z_d [meters]: "))

print("Mesh/discretization parameters: ")
n_theta = int(input("Nodes along circumference n_theta: "))
n_z = int(input("Nodes along the axis n_z: "))

a = a_len / Rc


# Radial deformation computation (see final report)
def calc_dr(theta, z):
    theta_diff = theta - (np.pi - a / 2)
    zeta = z - (z_d - b / 2)

    if theta_diff < 0 or theta_diff > a or zeta < 0 or zeta > b:
        return 0.0

    if theta_diff > np.pi:
        theta_diff = 2 * np.pi - theta_diff

    arc = Rc * theta_diff
    dr = (
        -(wb / 4)
        * (1 - np.cos((2 * np.pi * theta_diff) / a))
        * (1 - np.cos((2 * np.pi * zeta) / b))
    )
    return dr


# Mesh generation and 3D plot
theta_vals = np.linspace(0, 2 * np.pi, n_theta)
z_vals = np.linspace(0, H, n_z)
Theta, Z = np.meshgrid(theta_vals, z_vals)
Dr = np.vectorize(calc_dr)(Theta, Z)
R_deformed = Rc + Dr
X = R_deformed * np.cos(Theta)
Y = R_deformed * np.sin(Theta)

show_plot = input("Show a 3D plot? [Y]/[N]: ")

if show_plot == "Y":
    faces = []
    for i in range(n_z - 1):
        for j in range(n_theta - 1):
            p1 = [X[i, j], Y[i, j], Z[i, j]]
            p2 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
            p3 = [X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]]
            p4 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]

            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    faces = np.array(faces)
    points, inverse = np.unique(faces.reshape(-1, 3), axis=0, return_inverse=True)

    n_faces = faces.shape[0]
    cells = np.c_[np.full(n_faces, 3), inverse.reshape(-1, 3)].flatten()

    pv_mesh = pv.PolyData(points, cells)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True)
    plotter.camera_position = "iso"
    plotter.hide_axes()
    plotter.remove_bounds_axes()
    plotter.show()


# Conversion from a quad mesh to a tri (for STL compatibility), and model export
print("Exporting mesh and .STL file generation...")

faces = []
for i in range(n_z - 1):
    for j in range(n_theta - 1):
        p1 = [X[i, j], Y[i, j], Z[i, j]]
        p2 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
        p3 = [X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]]
        p4 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]

        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

faces = np.array(faces)

dimple_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    dimple_mesh.vectors[i] = f

# Dialog for file saving
Tk().withdraw()
file_path = filedialog.asksaveasfilename(
    defaultextension=".stl",
    filetypes=[("STL files", "*.stl")],
    title="Save STL model",
)

if file_path:
    dimple_mesh.save(file_path)
    print(f"STL model saved: {file_path}")
else:
    print("Saving failed.")
