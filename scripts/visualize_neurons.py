import numpy as np
import pyvista as pv
from plyfile import PlyData
import imageio.v2 as imageio
import imageio.v3 as imageio_v3
from pathlib import Path
from tqdm import tqdm
import math
import yaml
import h5py
from PIL import Image

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

def create_rotation_mp4_from_h5(h5_path_idx, output_mp4=None, length_sec=20, resolution=(1024, 1024), verbose=False, fps=16):
    """Renders a rotating 3D animation of a mesh loaded from an HDF5 file and saves it as an MP4.
    Expects the HDF5 to contain point coordinates in dataset 'data0' and cell connectivity in dataset 'data1'.
    The mesh is assumed to be a volume mesh (tetrahedral) from which the surface is extracted.
    """
    h5_path, color_idx = h5_path_idx
    # Load mesh from HDF5
    with h5py.File(h5_path, 'r') as f:
        points = f['data0'][:]
        cells = f['data1'][:]
        labels = f['data3'][:]  # Cell labels
        # Create PyVista UnstructuredGrid (tetrahedral)
    n_cells = cells.shape[0]
    cells_pv = np.hstack([np.full((n_cells, 1), 4, dtype=int), cells]).ravel()
    mesh = pv.UnstructuredGrid(cells_pv, np.full(n_cells, pv.CellType.TETRA), points)
    mesh.cell_data['label'] = labels
    # Extract surface to get a surface mesh for visualization
    mesh = mesh.threshold([2, 99], scalars='label')
    # mesh = mesh.extract_surface()
    surf = mesh.extract_surface(algorithm='dataset_surface')
        
    if output_mp4 is None:
        output_mp4 = Path(h5_path).parent / "mesh.mp4"
        
    mass_center = calculate_mass_center(surf)
    surf.translate(-mass_center, inplace=True)
    
    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.add_mesh(surf, color=COLORS[color_idx % len(COLORS)])
    plotter.set_background("#111106ff")
    plotter.view_isometric()
    plotter.camera.focal_point = (0, 0, 0)
    
    frames = int(length_sec * fps)
    original_mesh = surf.copy()
    images = []

    if verbose:
        rnge = tqdm(range(frames), desc="Frames", colour="green") if verbose else range(frames)
    else:
        rnge = range(frames)
    for i in rnge:
        z_angle = (i / frames) * 360.0
        y_angle = 15.0 * np.sin(2 * np.pi * i / frames)
        surf.copy_from(original_mesh)
        surf.rotate_z(z_angle, inplace=True)
        surf.rotate_y(y_angle, inplace=True)
        plotter.render()
        images.append(plotter.screenshot())

    plotter.close()
    
    # Save as MP4 (requires imageio-ffmpeg installed)
    images = np.stack(images, axis=0)  # Convert list of images to a 4D numpy array
    imageio.mimwrite(
        output_mp4, 
        images, 
        fps=fps,
        codec='libx264', 
        pixelformat='yuv420p'
    )
    
    if verbose:
        print(f"MP4 saved to {output_mp4}")
        
    return output_mp4

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import imageio
import imageio.v3 as imageio_v3
import numpy as np
from PIL import Image
from tqdm import tqdm

def _process_single_file(p, r, c, cell_w, cell_h, frame_step, target_num_frames):
    """
    Worker function to read, step, and resize frames for a single video/GIF.
    Executed concurrently in separate threads.
    """
    try:
        frames = imageio_v3.imread(p)[::frame_step]
    except Exception as e:
        print(f"Error loading {p}: {e}")
        return r, c, None

    n_frames = min(len(frames), target_num_frames)
    
    # Pre-allocate array for this cell across all frames
    cell_frames = np.zeros((n_frames, cell_h, cell_w, 3), dtype=np.uint8)

    for i in range(n_frames):
        frame_data = frames[i]
        pil_img = Image.fromarray(frame_data)#.convert('RGB')
        
        pil_img_resized = pil_img.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        cell_frames[i] = np.array(pil_img_resized)

    return r, c, cell_frames


def create_combined_mp4(
    gif_paths, 
    output_path, 
    N, 
    start_idx=0, 
    frame_step=1, 
    fps=32, 
    verbal=False, 
    final_resolution=1024,
    max_workers=None,
):
    """
    Combines N MP4s/GIFs into a single grid MP4 using multi-threaded parallel processing.
    """
    selected_paths = gif_paths[start_idx : start_idx + N]
    if len(selected_paths) < N:
        print(f"Warning: Only found {len(selected_paths)} files, using all available.")
        N = len(selected_paths)

    if N == 0:
        print("No files to combine.")
        return None

    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    # Force cell dimensions to even numbers up front so canvas totals are always even
    cell_w = ((final_resolution // cols) // 2) * 2
    cell_h = ((final_resolution // rows) // 2) * 2

    canvas_w = cols * cell_w
    canvas_h = rows * cell_h

    # Inspect first video/GIF for frame count
    first_frames = imageio_v3.imread(selected_paths[0])
    num_frames = len(first_frames[::frame_step])

    if verbal:
        print(f"Combining {N} files into a {rows}x{cols} grid with {num_frames} frames "
              f"(cell size: {cell_w}x{cell_h}, final size: {canvas_w}x{canvas_h})...")

    # Pre-allocate full 4D canvas: (num_frames, canvas_h, canvas_w, 3)
    grid_frames = np.zeros((num_frames, canvas_h, canvas_w, 3), dtype=np.uint8)

    # Safely extract top-left RGB pixel
    if first_frames.ndim == 4:
        bg_color = first_frames[0, 0, 0, :3]
    elif first_frames.ndim == 3:
        bg_color = first_frames[0, 0, :3]
    else:
        bg_color = np.array([0, 0, 0], dtype=np.uint8)

    grid_frames[:] = bg_color

    # Determine thread count (defaults to CPU count + 4)
    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) + 4)

    # Prepare parallel tasks
    tasks = []
    for idx, p in enumerate(selected_paths):
        r = idx // cols
        c = idx % cols
        tasks.append((p, r, c))

    # Process files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single_file, p, r, c, cell_w, cell_h, frame_step, num_frames)
            for p, r, c in tasks
        ]

        iterator = as_completed(futures)
        if not verbal:
            iterator = tqdm(iterator, total=len(futures), desc="Processing Files (Parallel)", colour="blue")

        for future in iterator:
            r, c, cell_frames = future.result()
            if cell_frames is not None:
                n_f = cell_frames.shape[0]
                
                # Compute placement coordinates
                y_start = r * cell_h
                y_end = min(y_start + cell_h, canvas_h)
                x_start = c * cell_w
                x_end = min(x_start + cell_w, canvas_w)

                actual_h = y_end - y_start
                actual_w = x_end - x_start

                # Safely slice both canvas and cell_frames to guarantee matching dimensions
                grid_frames[:n_f, y_start:y_end, x_start:x_end] = cell_frames[:, :actual_h, :actual_w]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbal:
        print("Writing output MP4...")

    imageio.mimwrite(
        output_path, 
        grid_frames, 
        fps=fps, 
        codec='libx264', 
        pixelformat='yuv420p'
    )
    
    print(f"Saved combined MP4 to {output_path}...")
    return output_path

def create_volume_gif(
    result_folder,
    output_gif=None,
    primary_opacity=1.0,
    other_opacity=0.2,
    frames_per_degree=1,
    resolution=(800, 800),
    verbose=False,
    cell_type="neuron",
    cell_type_other=None,
    primary_color=None,
    other_color=None,
    post_str="",

):
    """
    Creates a rotating 3D animation GIF from a volumetric mesh where neurons
    are rendered at full opacity and other cells at low opacity.

    Args:
        result_folder (str/Path): Path to the result folder containing meshes/ and processed/
        output_gif (str/Path): Output GIF path (default: <result_folder>/volume_fraction.gif)
        primary_opacity (float): Opacity for neuron cells (0-1)
        other_opacity (float): Opacity for non-neuron cells (0-1)
        frames_per_degree (float): Number of frames per degree of rotation
        resolution (tuple): Window size for rendering
        verbose (bool): Print progress information
        cell_type (str): Cell type to highlight (default: "neuron").
        primary_color (tuple): Color for primary cells (R, G, B)
        other_color (tuple): Color for other cells (R, G, B)
        post_str (str): Additional string to append to the output file name
    Returns:
        Path: Path to the created GIF
    """
    result_folder = Path(result_folder)
    meshes_dir = result_folder / 'meshes'

    # Load cell type mapping
    cell_type_mapping_path = meshes_dir / 'cell_type_mapping.yml'
    if not cell_type_mapping_path.exists():
        raise FileNotFoundError(f"cell_type_mapping.yml not found at {cell_type_mapping_path}")

    with open(cell_type_mapping_path, 'r') as f:
        cell_type_mapping = yaml.safe_load(f)

    # Load mesh from HDF5 (via XDMF reference)
    mesh_h5 = meshes_dir / 'mesh.h5'
    if not mesh_h5.exists():
        raise FileNotFoundError(f"mesh.h5 not found at {mesh_h5}")

    with h5py.File(mesh_h5, 'r') as f:
        points = f['data0'][:]
        cells = f['data1'][:]
        labels = f['data3'][:]  # Cell labels
        # Compute center of the entire point set
        center = points.mean(axis=0)

    # Create PyVista UnstructuredGrid
    n_cells = cells.shape[0]
    cells_pv = np.hstack([np.full((n_cells, 1), 4, dtype=int), cells]).ravel()
    mesh = pv.UnstructuredGrid(cells_pv, np.full(n_cells, pv.CellType.TETRA), points)
    mesh.cell_data['label'] = labels

    # Identify neuron cell IDs and other cell IDs (excluding extracellular)
    if cell_type == "all":
        neuron_ids = {int(k) for k, v in cell_type_mapping.items() if v != 'extracellular'}
        other_ids = set()
    else:
        if cell_type_other is not None:
            neuron_ids = {int(k) for k, v in cell_type_mapping.items() if v == cell_type}
            other_ids = {int(k) for k, v in cell_type_mapping.items() if v == cell_type_other}
        else:
            neuron_ids = {int(k) for k, v in cell_type_mapping.items() if v == cell_type}
            other_ids = {int(k) for k, v in cell_type_mapping.items() if v != cell_type and v != 'extracellular'}
    cell_ids_to_show = neuron_ids | other_ids

    # Assign a unique color from COLORS to each cell id
    color_map = {}
    for idx, cell_id in enumerate(sorted(cell_ids_to_show)):
        color_map[cell_id] = COLORS[idx % len(COLORS)]

    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.set_background("#111106ff") # #111106ff

    # Add each cell as a separate mesh
    for cell_id in tqdm(cell_ids_to_show, desc="Cells", colour="green"):
        lab = cell_id
        sub = mesh.threshold([lab, lab], scalars='label')
        if sub.n_cells == 0:
            continue
        surf = sub.extract_surface(algorithm='dataset_surface')
        # Center the surface
        surf.translate(-center, inplace=True)
        color = color_map[cell_id]
        # Set opacity based on cell type
        if cell_id in neuron_ids:
            color = primary_color if primary_color is not None else color
            opacity = primary_opacity
        else:
            color = other_color if other_color is not None else color
            opacity = other_opacity
        plotter.add_mesh(surf, color=color, opacity=opacity, smooth_shading=True)

    # Set camera
    plotter.view_isometric()
    plotter.camera.focal_point = (0, 0, 0)

    if output_gif is None:
        output_gif = result_folder / "meshes" / f'volume_{cell_type}{post_str}.gif'

    frames = int(360 * frames_per_degree)
    images = []

    rng = tqdm(range(frames), desc="Frames", colour="green") if verbose else range(frames)
    for i in rng:
        z_angle = (i / frames) * 360.0
        y_angle = 15.0 * np.sin(2 * np.pi * i / frames)
        plotter.camera.azimuth = z_angle
        plotter.camera.elevation = y_angle
        plotter.render()
        images.append(plotter.screenshot())

    plotter.close()
    imageio.imwrite(output_gif, images, fps=16, loop=0)

    if verbose:
        print(f"GIF saved to {output_gif}")
    return output_gif

if __name__ == "__main__":
    path = Path("../emimesh/results/cells_cube_highres/branch/size10000_ncells550_0")

    resolution = (512, 512)
    frames_per_degree = 1.0

    # Solid
    for cell_type in ["astrocyte", "microglia", "neuron", "all"]:
        gif = create_volume_gif(
            result_folder=path,
            primary_opacity=1.0,
            other_opacity=0.0,
            resolution=resolution,
            verbose=True,
            frames_per_degree=frames_per_degree,
            cell_type=cell_type,
        )

    # neuron red, astrocyte blue
    # gif = create_volume_gif(
    #     result_folder=path,
    #     primary_opacity=1.0,
    #     other_opacity=0.7,
    #     resolution=resolution,
    #     verbose=True,
    #     frames_per_degree=frames_per_degree,
    #     cell_type="neuron",
    #     cell_type_other="astrocyte",
    #     post_str="_red_blue",
    #     primary_color="red",
    #     other_color="blue",
    # )

    # gif = create_volume_gif(
    #     result_folder=path,
    #     primary_opacity=1.0,
    #     other_opacity=1.0,
    #     resolution=resolution,
    #     verbose=True,
    #     frames_per_degree=frames_per_degree,
    #     cell_type="neuron",
    #     cell_type_other="astrocyte",
    #     post_str="_red_blue_solid",
    #     primary_color="red",
    #     other_color="blue",
    # )

    # neuron red, other
    # gif = create_volume_gif(
    #     result_folder=path,
    #     primary_opacity=1.0,
    #     other_opacity=0.2,
    #     resolution=resolution,
    #     verbose=True,
    #     frames_per_degree=frames_per_degree,
    #     cell_type="neuron",
    #     post_str="_red",
    #     primary_color="red",
    # )

    # extracellular
    # gif = create_volume_gif(
    #     result_folder=path,
    #     primary_opacity=0.8,
    #     other_opacity=0.0,
    #     resolution=resolution,
    #     verbose=True,
    #     frames_per_degree=frames_per_degree,
    #     cell_type="extracellular",
    #     # post_str="_red",
    #     primary_color="blue",
    # )

    # for cell_type in ["astrocyte", "microglia", "neuron", "all"]:
    #     gif = create_volume_gif(
    #         result_folder=path,
    #         primary_opacity=.9,
    #         other_opacity=0.1,
    #         resolution=(256, 256),
    #         verbose=True,
    #         frames_per_degree=2.0,
    #         cell_type=cell_type,
    #         post_str="_opacity0901"
    #     )