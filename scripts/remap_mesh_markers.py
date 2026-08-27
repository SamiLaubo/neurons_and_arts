#!/usr/bin/env python
"""
Remap XDMF mesh cell markers to match original surface segmentation labels.

Given a set of PLY surface files (each named with its original label, e.g. '3.ply')
and an XDMF mesh with arbitrary cell markers, this script produces a new XDMF
file where cells that lie near a surface are reassigned that surface's label.

Approach:
1. For each surface PLY:
   - Load the surface mesh.
   - Sample points on the surface (vertices or face centroids).
   - Build a KDTree of these surface points.
2. For each cell in the XDMF mesh (triangle or tetra):
   - Compute its centroid.
   - Query the KDTree of each surface to find the closest surface point.
   - If the distance is within a tolerance (relative to mesh bbox diagonal),
     consider the cell a candidate for that surface.
3. Assign each cell to the surface with the smallest distance (if within tolerance).
   - If multiple surfaces are equally close, pick the one with the smallest label
     (or you could vote; here we use nearest).
4. Write a new XDMF file with the updated cell tags.

Usage:
    python remap_mesh_markers.py \
        --surfaces emimesh/results/fix_labeling/size30000_ncells5_0_org/surfaces \
        --mesh emimesh/results/fix_labeling/size30000_ncells5_0_org/meshes/mesh.xdmf \
        --output remapped.xdmf \
        --tag_key marker \
        --tol 0.01
"""

import os
import argparse
import numpy as np
import meshio
from scipy.spatial import KDTree
from pathlib import Path


def to_numeric_tag(val):
    """Convert tag to numeric if possible, otherwise keep as string."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return str(val).strip()


def remap_mesh_markers(
    surface_dir,
    xdmf_path,
    output_path=None,
    tag_key="marker",
    rel_tolerance=0.01,
):
    """
    Remap mesh cell markers based on proximity to labeled surfaces.

    Parameters
    ----------
    surface_dir : str or Path
        Directory containing PLY surface files named "<label>.ply".
    xdmf_path : str or Path
        Path to input XDMF mesh.
    output_path : str or Path, optional
        Where to save the remapped XDMF. If None, overwrites input.
    tag_key : str, default "marker"
        Cell data key storing the tags.
    rel_tolerance : float, default 0.01
        Allowed distance as fraction of mesh bounding-box diagonal.

    Returns
    -------
    str
        Path to the written XDMF file.
    """
    surface_dir = Path(surface_dir)
    xdmf_path = Path(xdmf_path)
    if output_path is None:
        output_path = xdmf_path
    else:
        output_path = Path(output_path)

    if not xdmf_path.exists():
        raise FileNotFoundError(f"XDMF file not found: {xdmf_path}")

    # ------------------------------------------------------------------ #
    # 1. Load mesh
    # ------------------------------------------------------------------ #
    print(f"Loading XDMF mesh: {xdmf_path}")
    msh = meshio.read(xdmf_path)

    if tag_key not in msh.cell_data:
        raise KeyError(
            f"Attribute key '{tag_key}' not found. Available: {list(msh.cell_data.keys())}"
        )

    # ------------------------------------------------------------------ #
    # 2. Compute cell centroids and collect current tags
    # ------------------------------------------------------------------ #
    cell_centroids = []
    cell_tags = []  # original tags (we will overwrite)
    cell_offset = 0  # to map back to cell_data per block

    for block_idx, (cell_block, tags) in enumerate(zip(msh.cells, msh.cell_data[tag_key])):
        pts = msh.points[cell_block.data]  # (n_cells_in_block, pts_per_cell, 3)
        centroids = pts.mean(axis=1)       # (n_cells_in_block, 3)
        cell_centroids.append(centroids)
        cell_tags.append(np.asarray(tags))
        cell_offset += len(centroids)

    cell_centroids = np.vstack(cell_centroids) if cell_centroids else np.empty((0, 3))
    cell_tags = np.concatenate(cell_tags) if cell_tags else np.empty((0,))

    if len(cell_centroids) == 0:
        raise ValueError("No cells found in XDMF mesh.")

    # ------------------------------------------------------------------ #
    # 3. Load surfaces and build KDTree of surface points
    # ------------------------------------------------------------------ #
    ply_files = sorted(surface_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {surface_dir}")

    print(f"Found {len(ply_files)} surface files.")

    # We will store for each surface: (label, KDTree_of_surface_points)
    surface_kdtrees = []
    surface_labels = []

    for ply_path in ply_files:
        label_str = ply_path.stem
        label_num = to_numeric_tag(label_str)

        surf = meshio.read(ply_path)
        # Use vertices as sample points; could also use face centroids for better accuracy
        surf_pts = surf.points
        if len(surf_pts) == 0:
            print(f"[WARNING] Surface {ply_path.name} has no points; skipping.")
            continue

        kdtree = KDTree(surf_pts)
        surface_kdtrees.append(kdtree)
        surface_labels.append(label_num)
        print(f"  Loaded {ply_path.name} -> label {label_num} ({len(surf_pts)} points)")

    if not surface_kdtrees:
        raise RuntimeError("No valid surfaces loaded.")

    # ------------------------------------------------------------------ #
    # 4. For each cell, find nearest surface within tolerance
    # ------------------------------------------------------------------ #
    # Compute mesh bbox diagonal for absolute tolerance
    mesh_pts = msh.points
    bbox_diag = np.linalg.norm(mesh_pts.max(axis=0) - mesh_pts.min(axis=0))
    abs_tol = rel_tolerance * bbox_diag
    print(f"\nMesh bbox diagonal: {bbox_diag:.2f}")
    print(f"Using absolute tolerance ({rel_tolerance*100:.1f}%): {abs_tol:.2f}")

    # Initialize new tags as a copy of original (will overwrite where matched)
    new_tags = cell_tags.copy()

    # For efficiency, we loop over surfaces and assign cells that are closest to that surface
    # and within tolerance. To avoid assigning a cell to multiple surfaces, we keep track
    # of the best (smallest) distance seen so far.
    best_dist = np.full(len(cell_centroids), np.inf)
    assigned_label = np.full(len(cell_centroids), -1, dtype=object)  # -1 = unassigned

    for surf_idx, (kdtree, label) in enumerate(zip(surface_kdtrees, surface_labels)):
        # Query: distance from each cell centroid to nearest point on this surface
        dists, _ = kdtree.query(cell_centroids, workers=-1)
        # Where this surface is closer than any previously seen and within tolerance
        mask = (dists < best_dist) & (dists <= abs_tol)
        if np.any(mask):
            best_dist[mask] = dists[mask]
            assigned_label[mask] = label
            print(f"  Surface {label}: assigned {mask.sum()} cells (new closest)")

    # ------------------------------------------------------------------ #
    # 5. Apply new tags
    # ------------------------------------------------------------------ #
    n_assigned = np.sum(assigned_label != -1)
    n_total = len(cell_centroids)
    print(f"\nAssigned {n_assigned}/{n_total} cells ({100*n_assigned/n_total:.1f}%)")
    if n_assigned < n_total:
        print(f"  {n_total - n_assigned} cells remain with original markers "
              f"(outside tolerance of any surface).")

    # Where we assigned a label, write it back into new_tags
    new_tags[assigned_label != -1] = assigned_label[assigned_label != -1]

    # ------------------------------------------------------------------ #
    # 6. Pack new tags back into ms h.cell_data structure
    # ------------------------------------------------------------------ #
    # We need to split new_tags according to original block sizes.
    new_cell_data = []
    idx = 0
    for cell_block, tags in zip(msh.cells, msh.cell_data[tag_key]):
        n = len(tags)
        new_cell_data.append(new_tags[idx:idx+n].tolist())
        idx += n

    # Create a new mesh with updated cell data
    new_msh = meshio.Mesh(
        points=msh.points,
        cells=msh.cells,
        cell_data={tag_key: new_cell_data},
        field_data=msh.field_data,
    )

    # ------------------------------------------------------------------ #
    # 7. Write output
    # ------------------------------------------------------------------ #
    print(f"\nWriting remapped mesh to: {output_path}")
    meshio.write(output_path, new_msh)
    print("Done.")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Remap XDMF mesh markers to match original surface labels."
    )
    parser.add_argument(
        "--surfaces",
        type=str,
        required=True,
        help="Directory containing PLY surface files named '<label>.ply'.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to input XDMF mesh.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save remapped XDMF (default: overwrite input).",
    )
    parser.add_argument(
        "--tag_key",
        type=str,
        default="marker",
        help="Cell data key storing the tags (default: marker).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Relative tolerance as fraction of mesh bbox diagonal (default: 0.01).",
    )
    args = parser.parse_args()

    remap_mesh_markers(
        surface_dir=args.surfaces,
        xdmf_path=args.mesh,
        output_path=args.output,
        tag_key=args.tag_key,
        rel_tolerance=args.tol,
    )


if __name__ == "__main__":
    main()