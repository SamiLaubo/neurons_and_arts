import os
import numpy as np
import meshio
from scipy.spatial import KDTree
import json
import re
from pathlib import Path
import yaml


def verify_xdmf_labeling(ply_files, xdmf_path, tag_key="marker", rel_tolerance=0.01):
    """
    Verifies XDMF mesh cell labels against PLY surface files using a relative distance tolerance.

    Parameters:
    -----------
    ply_files : iterable of Path/str
        Paths to the PLY surface files.
    xdmf_path : Path/str
        Path to the output .xdmf file.
    tag_key : str
        Cell data attribute key in XDMF storing the tags.
    rel_tolerance : float
        Allowed spatial query distance as a fraction of the mesh bounding box diagonal.
        0.01 = 1% of the total mesh extent (e.g. ~480 units for a 30,000 unit box).
    """
    if not os.path.exists(xdmf_path):
        raise FileNotFoundError(f"XDMF file not found: {xdmf_path}")

    print(f"Loading XDMF mesh: {xdmf_path}...")
    msh = meshio.read(xdmf_path)

    # 1. Compute dynamic tolerance based on overall bounding box diagonal
    min_pt = msh.points.min(axis=0)
    max_pt = msh.points.max(axis=0)
    bbox_diag = np.linalg.norm(max_pt - min_pt)
    abs_tolerance = rel_tolerance * bbox_diag

    print(f"Mesh Bounding Box Diagonal: {bbox_diag:.2f} units")
    print(f"Using Absolute Tolerance ({rel_tolerance*100:.1f}%): {abs_tolerance:.2f} units\n")

    if tag_key not in msh.cell_data:
        raise KeyError(f"Attribute key '{tag_key}' not found. Available keys: {list(msh.cell_data.keys())}")

    cell_tags = np.concatenate(msh.cell_data[tag_key]).squeeze()

    out_centroids = []
    for cell_block in msh.cells:
        if cell_block.type in ["triangle", "tetra"]:
            points = msh.points[cell_block.data]
            out_centroids.append(points.mean(axis=1))

    if not out_centroids:
        raise ValueError("No triangle or tetrahedra elements found in XDMF mesh.")

    out_centroids = np.vstack(out_centroids)
    out_kdtree = KDTree(out_centroids)

    def to_numeric_tag(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return str(val).strip()

    print("--- Starting XDMF Label Consistency Verification ---")
    total_checked = 0
    total_matched = 0

    for ply_path in ply_files:
        if not os.path.exists(ply_path):
            print(f"[WARNING] Surface file missing: {ply_path}")
            continue

        surf_mesh = meshio.read(ply_path)
        if "triangle" not in surf_mesh.cells_dict:
            continue

        raw_tag = ply_path.stem
        expected_numeric = to_numeric_tag(raw_tag)

        tri_indices = surf_mesh.cells_dict["triangle"]
        in_centroids = surf_mesh.points[tri_indices].mean(axis=1)

        distances, nearest_indices = out_kdtree.query(in_centroids)
        assigned_raw = cell_tags[nearest_indices]
        assigned_numeric = np.array([to_numeric_tag(t) for t in assigned_raw])

        # Validate distance within relative threshold AND tag match
        within_tol = distances <= abs_tolerance
        tag_matches = (assigned_numeric == expected_numeric)
        valid_matches = tag_matches & within_tol

        n_faces = len(in_centroids)
        n_matched = np.sum(valid_matches)

        total_checked += n_faces
        total_matched += n_matched

        accuracy = (n_matched / n_faces) * 100 if n_faces > 0 else 0
        status = "PASSED" if n_matched == n_faces else "MISMATCH DETECTED"

        print(f"\nFile: {ply_path.name}")
        print(f"  Expected Tag ID: {raw_tag}")
        print(f"  Faces Verified:  {n_matched}/{n_faces} ({accuracy:.2f}%)")
        print(f"  Avg Distance:    {distances.mean():.2f} units")
        print(f"  Status:          [{status}]")

        if n_matched < n_faces:
            mismatched_tags, counts = np.unique(assigned_raw[~valid_matches], return_counts=True)
            summary = ", ".join([f"Tag {t}: {c} faces" for t, c in zip(mismatched_tags, counts)])
            print(f"  Found assigned tags: {summary}")

    print("\n-----------------------------------------------")
    overall_acc = (total_matched / total_checked * 100) if total_checked > 0 else 0
    print(f"Overall Label Accuracy: {overall_acc:.2f}% ({total_matched}/{total_checked} faces)")
    print("-----------------------------------------------\n")

    return total_matched == total_checked


import meshio
import numpy as np

def print_cell_counts_per_tag(xdmf_path, tag_key=None):
    """
    Prints a table showing the number of cells per tag in an XDMF mesh.
    """
    msh = meshio.read(xdmf_path)

    if not msh.cell_data:
        print("No cell data/tags found in the XDMF file.")
        return

    # Auto-detect tag key if not specified
    if tag_key is None:
        tag_key = list(msh.cell_data.keys())[0]

    cell_tags_list = msh.cell_data[tag_key]

    print(f"==================================================")
    print(f" Diagnostics for: {xdmf_path}")
    print(f" Tag Attribute:   '{tag_key}'")
    print(f"==================================================\n")

    # Dictionary to aggregate total counts across all cell blocks
    overall_tag_counts = {}

    # Print breakdown per cell block (e.g., triangle vs tetra)
    for block_idx, (cell_block, tags) in enumerate(zip(msh.cells, cell_tags_list)):
        tags_flat = np.asarray(tags).ravel()
        unique_tags, counts = np.unique(tags_flat, return_counts=True)

        print(f"--- Block {block_idx + 1}: Element Type = '{cell_block.type}' ---")
        print(f"{'Tag ID':<10} | {'Cell Count (n)':<15}")
        print("-" * 28)

        for tag, count in zip(unique_tags, counts):
            tag_val = int(tag) if np.issubdtype(type(tag), np.integer) else tag
            print(f"{tag_val:<10} | {count:<15}")

            # Accumulate overall totals
            overall_tag_counts[tag_val] = overall_tag_counts.get(tag_val, 0) + count
        print()

    # Print overall summary table
    print("==================================================")
    print(" OVERALL SUMMARY (All Element Types Combined)")
    print("==================================================")
    print(f"{'Tag ID':<10} | {'Total Cells (n)':<15}")
    print("-" * 28)
    for tag in sorted(overall_tag_counts.keys()):
        print(f"{tag:<10} | {overall_tag_counts[tag]:<15}")
    print("==================================================\n")

# Run diagnostic
# print_cell_counts_per_tag("output.xdmf")


def verify_xdmf_labeling_by_centroid(ply_files, xdmf_path, tag_key="marker", rel_tolerance=0.01):
    """
    Verifies XDMF mesh cell labels against PLY surface files by comparing the centroid of each PLY surface
    to the centroid of each XDMF cell, using the closest cell's tag.

    Parameters:
    -----------
    ply_files : iterable of Path/str
        Paths to the PLY surface files.
    xdmf_path : Path/str
        Path to the output .xdmf file.
    tag_key : str
        Cell data attribute key in XDMF storing the tags.
    rel_tolerance : float
        Allowed spatial query distance as a fraction of the mesh bounding box diagonal.
        0.01 = 1% of the total mesh extent.
    """
    import os
    import numpy as np
    import meshio
    from scipy.spatial import KDTree

    if not os.path.exists(xdmf_path):
        raise FileNotFoundError(f"XDMF file not found: {xdmf_path}")

    print(f"Loading XDMF mesh: {xdmf_path}...")
    msh = meshio.read(xdmf_path)

    # 1. Compute dynamic tolerance based on overall bounding box diagonal
    min_pt = msh.points.min(axis=0)
    max_pt = msh.points.max(axis=0)
    bbox_diag = np.linalg.norm(max_pt - min_pt)
    abs_tolerance = rel_tolerance * bbox_diag

    print(f"Mesh Bounding Box Diagonal: {bbox_diag:.2f} units")
    print(f"Using Absolute Tolerance ({rel_tolerance*100:.1f}%): {abs_tolerance:.2f} units\n")

    if tag_key not in msh.cell_data:
        raise KeyError(f"Attribute key '{tag_key}' not found. Available keys: {list(msh.cell_data.keys())}")

    # Collect all cells and their tags
    cell_tags = []
    cell_centroids = []
    for cell_block, tags in zip(msh.cells, msh.cell_data[tag_key]):
        # For each cell in the block
        for i, cell in enumerate(cell_block.data):
            points = msh.points[cell]  # shape (num_points_per_cell, 3)
            centroid = points.mean(axis=0)
            cell_centroids.append(centroid)
            cell_tags.append(tags[i])

    cell_centroids = np.array(cell_centroids)
    cell_tags = np.array(cell_tags)

    if len(cell_centroids) == 0:
        raise ValueError("No cells found in XDMF mesh.")

    # Build KDTree on cell centroids
    cell_kdtree = KDTree(cell_centroids)

    def to_numeric_tag(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return str(val).strip()

    print("--- Starting XDMF Label Consistency Verification (by Centroid) ---")
    total_checked = 0
    total_matched = 0

    for ply_path in ply_files:
        if not os.path.exists(ply_path):
            print(f"[WARNING] Surface file missing: {ply_path}")
            continue

        surf_mesh = meshio.read(ply_path)
        if "triangle" not in surf_mesh.cells_dict:
            continue

        raw_tag = ply_path.stem
        expected_numeric = to_numeric_tag(raw_tag)

        # Compute the centroid of the entire surface (average of all vertices)
        surface_centroid = surf_mesh.points.mean(axis=0)

        # Query the KDTree for the closest cell centroid
        distance, nearest_index = cell_kdtree.query(surface_centroid)
        assigned_tag = cell_tags[nearest_index]
        assigned_numeric = to_numeric_tag(assigned_tag)

        # Validate distance within relative threshold AND tag match
        within_tol = distance <= abs_tolerance
        tag_matches = (assigned_numeric == expected_numeric)
        valid_match = within_tol and tag_matches

        total_checked += 1
        if valid_match:
            total_matched += 1

        status = "PASSED" if valid_match else "MISMATCH DETECTED"
        print(f"\nFile: {ply_path.name}")
        print(f"  Expected Tag ID: {raw_tag}")
        print(f"  Surface Centroid: {surface_centroid}")
        print(f"  Assigned Cell Centroid: {cell_centroids[nearest_index]}")
        print(f"  Assigned Tag ID: {assigned_tag}")
        print(f"  Distance: {distance:.2f} units")
        print(f"  Status:          [{status}]")

    print("\n-----------------------------------------------")
    overall_acc = (total_matched / total_checked * 100) if total_checked > 0 else 0
    print(f"Overall Label Accuracy: {overall_acc:.2f}% ({total_matched}/{total_checked} surfaces)")
    print("-----------------------------------------------\n")

    return total_matched == total_checked


import json
import re


def get_csg_leaf_order(node, leaves=None):
    """Recursively collect leaf filenames in depth-first (left-to-right) order."""
    if leaves is None:
        leaves = []

    if isinstance(node, str):
        leaves.append(node)
    elif isinstance(node, dict):
        if "left" in node:
            get_csg_leaf_order(node["left"], leaves)
        if "right" in node:
            get_csg_leaf_order(node["right"], leaves)

    return leaves


def build_mapping_from_csg(json_path, offset=1):
    """Generates a mapping dict: { ftetwild_tag: original_marker }

    Set offset=1 if fTetWild output tags start at 1 (default for .msh files).
    Set offset=0 if fTetWild output tags start at 0.
    """
    with open(json_path, "r") as f:
        csg_tree = json.load(f)

    raw_leaves = get_csg_leaf_order(csg_tree)

    # Maintain first-appearance order of unique surface files
    unique_leaves = []
    for leaf in raw_leaves:
        if leaf not in unique_leaves:
            unique_leaves.append(leaf)

    mapping = {}
    for i, leaf in enumerate(unique_leaves):
        # Match digits before .ply (e.g. '5.ply' -> 5)
        match = re.search(r"(\d+)\.ply$", leaf)

        ftet_tag = i + offset

        if match:
            original_marker = int(match.group(1))
            mapping[ftet_tag] = original_marker
        else:
            # Handle non-numbered surfaces like 'roi.ply'
            mapping[ftet_tag] = "roi"

    return mapping


def create_mapping_from_yaml_to_marker_celltype(yaml_path):
    """
    Reads a YAML file mapping fTetWild tags to cell types and uses the CSG tree
    to create a mapping from original markers to cell types.

    Parameters:
    -----------
    yaml_path : str or Path
        Path to the YAML file (e.g., '.../processed/cell_type_mapping.yml')
        containing a mapping from fTetWild tags to cell types.

    Returns:
    --------
    dict
        Mapping from original marker (str or int) to cell type (str).
    """
    # Load the YAML file: fTetWild tag -> cell type
    with open(yaml_path, 'r') as f:
        tag_to_celltype = yaml.safe_load(f)

    # Determine the base directory: assume the YAML is in <base>/processed/
    yaml_path = Path(yaml_path)
    base_dir = yaml_path.parent.parent  # Go up two levels: processed -> result base
    csg_path = base_dir / "surfaces" / "csgtree.json"

    if not csg_path.exists():
        raise FileNotFoundError(f"CSG tree not found at: {csg_path}")

    # Get mapping from fTetWild tag to original marker (leaf path)
    tag_to_marker = build_mapping_from_csg(csg_path, offset=1)

    print(f'{tag_to_marker = }')

    # Build marker -> cell type mapping
    marker_to_celltype = {}
    for tag, celltype in tag_to_celltype.items():
        print(f'{tag = }, {celltype = }')

        if tag in tag_to_marker:
            print("yes")
            marker = tag_to_marker[tag]
            print(f'{marker = }')
            # Extract marker from the path (e.g., '5.ply' -> 5, 'roi.ply' -> 'roi')
            # marker = int(Path(marker_path).stem)
            marker_to_celltype[marker] = celltype

    return marker_to_celltype


from pathlib import Path
if __name__ == "__main__":
    print("Hello")
    folder = Path("emimesh/results/fix_labeling/size30000_ncells5_0_org")
    # folder = Path("emimesh/results/fix_labeling/correct_surfaces")
    # folder = Path("emimesh/results/fix_labeling/test_1")
    # folder = Path("emimesh/results/fix_labeling/test_2")
    # folder = Path("emimesh/results/fix_labeling/test_3")
    # ply_files = list((folder / "surfaces").glob("*.ply"))
    # output_msh = folder / "meshes" / "mesh.xdmf"
    # print("\n=== Original Method (per triangle) ===")
    # verify_xdmf_labeling(ply_files, output_msh, tag_key="marker")
    # print("\n=== New Method (by surface centroid) ===")
    # verify_xdmf_labeling_by_centroid(ply_files, output_msh, tag_key="marker")


    # Example usage
    # tag_map = build_mapping_from_csg(folder / "surfaces/csgtree.json", offset=1)
    # print("Direct CSG Mapping Table:", tag_map)
    # Output will look like: {1: 'roi', 2: 5, 3: 4, 4: 3, 5: 2}


    created_mapping = create_mapping_from_yaml_to_marker_celltype(folder / "processed/cell_type_mapping.yml")
    print("Final Mapping Table (marker -> cell type):", created_mapping)