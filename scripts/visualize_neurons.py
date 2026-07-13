import os
import glob
import numpy as np
import pyvista as pv
from plyfile import PlyData
import imageio.v3 as imageio
from pathlib import Path
from tqdm import tqdm
import math
from PIL import Image

def get_cell_ply_files(path):
    """Finds all 2.ply files in the cell directories."""
    path = Path(path)
    return path.glob("**/[0-9].ply")

def load_ply_mesh(path):
    """Loads a PLY file and returns vertices and faces."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    faces = plydata['face']
    face_indices = np.vstack([face['vertex_indices'] for face in faces]).T
    return x, y, z, face_indices

def calculate_mass_center(mesh):
    """Calculates the mass center (centroid) of a surface mesh."""
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    total_area = np.sum(areas)
    if total_area == 0:
        return np.mean(points, axis=0)
    mass_center = np.sum(centroids * areas[:, np.newaxis], axis=0) / total_area
    return mass_center

def visualize_interactive_pyvista(ply_path):
    """Displays an interactive 3D surface using PyVista."""
    plotter = pv.Plotter()
    update_interactive_plotter(plotter, ply_path)
    return plotter.show()

def update_interactive_plotter(plotter, ply_path):
    """Updates an existing plotter with a new PLY surface."""
    mesh = pv.read(ply_path)
    mass_center = calculate_mass_center(mesh)
    mesh.translate(-mass_center, inplace=True)
    plotter.clear()
    plotter.add_mesh(mesh, color="lightblue", show_edges=True, edge_color="black", edge_opacity=0.5)
    plotter.set_background("black")
    plotter.view_isometric()
    plotter.render()

COLORS = [
    "antiquewhite", "aqua", "aquamarine", "beige", "bisque", "blanchedalmond", "blue",
    "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral",
    "cornflowerblue", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray",
    "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
    "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
    "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgrey",
    "dodgerblue", "firebrick", "forestgreen", "fuchsia", "gold", "goldenrod", "green",
    "greenyellow", "hotpink", "indianred", "indigo", "khaki", "lawngreen", "lemonchiffon",
    "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray",
    "lightgreen", "lightgrey", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
    "lightsteelblue", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
    "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue",
    "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mistyrose",
    "moccasin", "navajowhite", "navy", "olive", "olivedrab", "orange", "orangered",
    "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip",
    "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple", "red", "rosybrown",
    "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna",
    "silver", "skyblue", "slateblue", "snow", "springgreen", "steelblue", "tan", "teal",
    "thistle", "tomato", "turquoise", "violet", "wheat", "white", "yellow", "yellowgreen"
]

def create_rotation_gif(ply_path_idx, output_gif=None, frames_per_degree=1, resolution=(600, 600), verbose=False):
    """Renders a rotating 3D animation of a .ply file and saves it as a GIF."""
    ply_path, color_idx = ply_path_idx
    mesh = pv.read(ply_path)
    if output_gif is None:
        output_gif = Path(ply_path).parent / "surface.gif"
    mass_center = calculate_mass_center(mesh)
    mesh.translate(-mass_center, inplace=True)
    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.add_mesh(mesh, color=COLORS[color_idx % len(COLORS)])
    plotter.set_background("#111116")
    plotter.view_isometric()
    plotter.camera.focal_point = (0, 0, 0)
    frames = int(360 * frames_per_degree)
    original_mesh = mesh.copy()
    images = []
    rnge = tqdm(range(frames), desc="Frames", colour="green") if verbose else range(frames)
    for i in rnge:
        z_angle = (i / frames) * 360.0
        y_angle = 15.0 * np.sin(2 * np.pi * i / frames)
        mesh.copy_from(original_mesh)
        mesh.rotate_z(z_angle, inplace=True)
        mesh.rotate_y(y_angle, inplace=True)
        plotter.render()
        images.append(plotter.screenshot())
    plotter.close()
    imageio.mimsave(output_gif, images, fps=10, loop=0)
    if verbose:
        print(f"GIF saved to {output_gif}")
    return output_gif

def create_combined_gif(gif_paths, output_path, N=None, start_idx=0, cell_type=None, resize_factor=1.0, frame_step=1, duration=0.1):
    """
    Combines N GIFs into a single grid GIF with optional downsampling.

    Args:
        gif_paths (list): List of paths to source GIF files.
        output_path (str/Path): Path to save the resulting grid GIF.
        N (int, optional): Number of cells to include. Defaults to len(gif_paths).
        start_idx (int): Starting index in gif_paths.
        cell_type (str, optional): Filter gifs by this string (e.g., "astrocytes").
        resize_factor (float): Factor to scale dimensions (e.g., 0.5 for half size).
        frame_step (int): Step size for frames (e.g., 2 takes every second frame).
        duration (float): Duration of each frame in seconds.
    """
    # Filter by cell type if provided
    if cell_type:
        gif_paths = [p for p in gif_paths if cell_type in str(p)]

    if N is None:
        N = len(gif_paths)

    selected_paths = gif_paths[start_idx : start_idx + N]
    if len(selected_paths) < N:
        print(f"Warning: Only found {len(selected_paths)} gifs, using all available.")
        N = len(selected_paths)

    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    print("Loading and preprocessing GIFs...")
    all_gifs_frames = []
    for p in selected_paths:
        frames = imageio.imread(p)[::frame_step][:, ::int(1/resize_factor), ::int(1/resize_factor)]
        all_gifs_frames.append(frames)

    num_frames = min(len(f) for f in all_gifs_frames)
    orig_h, orig_w = imageio.imread(selected_paths[0]).shape[1:3]
    h = int(orig_h * resize_factor)
    w = int(orig_w * resize_factor)

    print(f"Combining {N} gifs into a {rows}x{cols} grid...")
    grid_frames = []

    for i in range(num_frames):
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, frames in enumerate(all_gifs_frames):
            r = idx // cols
            c = idx % cols
            frame_data = frames[i]
            print(f'{frame_data.shape = }')
            if frame_data.shape[-1] == 4:
                frame_data = frame_data[..., :3]
            frame_data = frame_data
            canvas[r*h : (r+1)*h, c*w : (c+1)*w] = frame_data
        grid_frames.append(canvas)

    imageio.imwrite(output_path, grid_frames, duration=duration, loop=0)
    return output_path
