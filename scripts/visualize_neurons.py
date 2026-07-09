import os
import glob
import numpy as np
import pyvista as pv
from plyfile import PlyData
import imageio
from pathlib import Path
from tqdm import tqdm

def get_cell_ply_files(path):
    """Finds all 2.ply files in the cell directories."""
    path = Path(path)
    return path.glob("**/[0-9].ply")
    # return glob.glob(os.path.join(path, "neuron_*", "surfaces", "2.ply"))

def load_ply_mesh(path):
    """Loads a PLY file and returns vertices and faces."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    # Faces are usually stored as a list of 3 indices for triangles
    faces = plydata['face']
    face_indices = np.vstack([face['vertex_indices'] for face in faces]).T

    return x, y, z, face_indices

def calculate_mass_center(mesh):
    """
    Calculates the mass center (centroid) of a surface mesh.
    For a surface, this is the area-weighted average of the triangle centroids.
    """
    points = mesh.points
    # PyVista faces are stored as [n_pts, i, j, k], we extract [i, j, k]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]

    # Centroids of each triangle
    centroids = (v0 + v1 + v2) / 3.0

    # Areas of each triangle: 0.5 * |(v1-v0) x (v2-v0)|
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    total_area = np.sum(areas)
    if total_area == 0:
        return np.mean(points, axis=0) # Fallback to simple average

    mass_center = np.sum(centroids * areas[:, np.newaxis], axis=0) / total_area
    return mass_center

def visualize_interactive_pyvista(ply_path):
    """Displays an interactive 3D surface using PyVista."""
    mesh = pv.read(ply_path)

    # Center the mesh for better viewing
    mass_center = calculate_mass_center(mesh)
    mesh.translate(-mass_center, inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightblue", show_edges=True, edge_color="black", edge_opacity=0.5)
    plotter.set_background("black")
    plotter.view_isometric()

    return plotter.show()

def create_rotation_gif(ply_path, output_gif=None, frames_per_degree=1, resolution=(600, 600), verbose=False):
    """Renders a rotating 3D animation of a .ply file and saves it as a GIF."""
    # PyVista can read PLY files directly
    mesh = pv.read(ply_path)
    if output_gif is None:
        output_gif = Path(ply_path).parent / "surface.gif"

    # Calculate the true mass center (centroid) of the surface
    mass_center = calculate_mass_center(mesh)
    mesh.translate(-mass_center, inplace=True)

    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.add_mesh(mesh, color="lightblue", smooth_shading=True)
    plotter.set_background("black")

    # Initial camera view
    plotter.view_isometric()
    plotter.camera.focal_point = (0, 0, 0)

    frames = int(2 * 360 * frames_per_degree)
    rotation_per_frame = 360.0 / frames

    images = []
    if verbose:
        rnge = tqdm(range(frames), desc="Frames", colour="green")
    else:
        rnge = range(frames)
    for i in rnge:
        # Rotate the mesh around each axis
        mesh.rotate_z(2 * rotation_per_frame, inplace=True)
        mesh.rotate_y(rotation_per_frame, inplace=True)
        mesh.rotate_x(rotation_per_frame, inplace=True)
        plotter.render()
        img = plotter.screenshot()
        images.append(img)

    plotter.close()

    # Save as GIF with looping enabled
    imageio.mimsave(output_gif, images, fps=10, loop=0)
    if verbose:
        print(f"GIF saved to {output_gif}")
    return output_gif
