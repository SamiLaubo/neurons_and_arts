#!/usr/bin/env python
"""
Remap XDMF mesh cell markers by connected component to match original surface labels.

Given:
- A set of PLY surface files named "<label>.ply" (e.g., 3.ply, 5.ply)
- An XDMF mesh where each cell currently has an arbitrary marker (may differ per cell)
Goal:
- Determine connected components of cells (based on face/edge adjacency)
- For each component, find the surface label whose surface points are closest (within tolerance)
- Assign that label to every cell in the component
- Write a new XDMF file with updated marker field.

Usage:
    python remap_mesh_by_component.py \
        --surfaces <surface_dir> \
        --mesh <input.xdmf> \
        --output <output.xdmf> \
        --tag_key marker \
        --tol 0.01
"""

import os
import argparse
import numpy as np
import meshio
from scipy.spatial import KDTree
from collections import defaultdict, deque
from pathlib import Path


def to_numeric_tag(val):
    """Convert tag to numeric if possible, otherwise keep as string."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return str(val).strip()


def build_face_adjacency(msh):
    """
    Build adjacency list of cells based on shared faces (for tetra) or edges (for triangles).
    Returns:
        adj: list of lists, adj[i] = list of neighbor cell indices
        n (global)
    """
    # We'll map each face (represented as sorted tuple of vertex indices) to list of cell indices
    face_to_cells = defaultdict(list)
    cell_start = 0  # global index offset for current block
    n_cells_total = sum(len(block.data) for block in msh.cells if block.type in ["tetra", "triangle"])
    # For simplicity, we only consider tetra and triangle cells as in verification script.
    # If other cell types exist, they are ignored for adjacency (treated as isolated).
    # Build mapping
    for block in msh.cells:
        if block.type not in ["tetra", "triangle"]:
            cell_start += len(block.data)
            continue
        pts = msh.points  # (N, 3)
        if block.type == "triangle":
            # each cell: 3 vertices -> faces are edges (2 vertices)
            for idx, vert in enumerate(block.data):
                a, b, c = vert
                # edges: (a,b), (b,c), (c,a)
                for edge in [(a, b), (b, c), (c, a)]:
                    face = tuple(sorted(edge))
                    face_to_cells[face].append(cell_start + idx)
        elif block.type == "tetra":
            # each cell: 4 vertices -> faces are triangles (3 vertices)
            for idx, vert in enumerate(block.data):
                a, b, c, d = vert
                faces = [
                    (a, b, c),
                    (a, b, d),
                    (a, c, d),
                    (b, c, d),
                ]
                for face in faces:
                    face_to_cells[tuple(sorted(face))].append(cell_start + idx)
        cell_start += len(block.data)

    # Build adjacency from face map: cells sharing a face are neighbors
    adj = [[] for _ in range(n_cells_total)]
    for cell_list in face_to_cells.values():
        if len(cell_list) < 2:
            continue
        for i in range(len(cell_list)):
            for j in range(i + 1, len(cell_list)):
                u, v = cell_list[i], cell_list[j]
                adj[u].append(v)
                adj[v].append(u)
    return adj, n_cells_total


def get_cell_centroids(msh):
    """Return array of centroids for tetra and triangle cells (in order of global index)."""
    centroids = []
    for block in msh.cells:
        if block.type not in ["tetra", "triangle"]:
            continue
        pts = msh.points[block.data]  # (n_cells_in_block, verts_per_cell, 3)
        centroids.append(pts.mean(axis=1))
    if not centroids:
        return np.empty((0, 3))
    return np.vstack(centroids)


def compute_component_labels(
    msh,
    surface_kdtrees,
    surface_labels,
    abs_tol,
):
    """
    For each connected component of cells (adjacency via faces/edges),
    determine the label to assign:
        - Compute component centroid (mean of cell centroids).
        - Find nearest surface within abs_tol.
        - If multiple surfaces equally close, pick the one with smallest label.
    Returns:
        new_labels_per_cell: array same length as number of tetra+triangle cells,
                             with assigned label (or -1 if no surface within tol)
    """
    adj, n_cells = build_face_adjacency(msh)
    visited = [False] * n_cells
    new_labels = np.full(n_cells, -1, dtype=object)  # -1 = unassigned/no match

    cell_centroids = get_cell_centroids(msh)

    for start in range(n_cells):
        if visited[start]:
            continue
        # BFS to collect component
        q = deque([start])
        comp = []
        visited[start] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        # Component centroid: average of cell centroids
        comp_centroids = cell_centroids[comp]  # (len(comp), 3)
        comp_centroid = comp_centroids.mean(axis=0) if len(comp) > 0 else np.zeros(3)
        # Find nearest surface
        best_dist = np.inf
        best_label = None
        for kdtree, label in zip(surface_kdtrees, surface_labels):
            dist, _ = kdtree.query(comp_centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        if best_dist <= abs_tol:
            for idx in comp:
                new_labels[idx] = best_label
        # else leave as -1 (no change)
    return new_labels


def remap_mesh_by_component(
    surface_dir,
    xdmf_path,
    output_path=None,
    tag_key="marker",
    rel_tolerance=0.01,
):
    """
    Remap mesh cell markers by connected component based on proximity to surfaces.

    Parameters
    ----------
    surface_dir : str or Path
        Directory containing PLY surface files named "<label>.ply".
    xdmf_path : str or Path
        Path to input XDMF mesh.
    output_path : str or Path, optional
        Where to save remapped XDMF; if None, overwrites input.
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
    # 2. Load surfaces and build KDTree of surface points (vertices)
    # ------------------------------------------------------------------ #
    ply_files = sorted(surface_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {surface_dir}")

    print(f"Found {len(ply_files)} surface files.")
    surface_kdtrees = []
    surface_labels = []
    for ply_path in ply_files:
        label_str = ply_path.stem
        label_num = to_numeric_tag(label_str)
        surf = meshio.read(ply_path)
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
    # 3. Compute mesh bbox diagonal for absolute tolerance
    # ------------------------------------------------------------------ #
    mesh_pts = msh.points
    bbox_diag = np.linalg.norm(mesh_pts.max(axis=0) - mesh_pts.min(axis=0))
    abs_tol = rel_tolerance * bbox_diag
    print(f"\nMesh bbox diagonal: {bbox_diag:.2f}")
    print(f"Using absolute tolerance ({rel_tolerance*100:.1f}%): {abs_tol:.2f}")

    # ------------------------------------------------------------------ #
    # 4. Compute component-based new labels
    # ------------------------------------------------------------------ #
    new_labels = compute_component_labels(msh, surface_kdtrees, surface_labels, abs_tol)

    # Count how many cells got a label
    n_labeled = np.sum(new_labels != -1)
    n_total = len(new_labels)
    print(f"\nAssigned labels to {n_labeled}/{n_total} cells ({100*n_labeled/n_total:.1f}%)")
    if n_labeled < n_total:
        print(f"  {n_total - n_labeled} cells remain with original markers "
              f"(no surface within tolerance).")

    # ------------------------------------------------------------------ #
    # 5. Pack new tags back into ms h.cell_data structure (only tetras+triangles)
    # ------------------------------------------------------------------ #
    # We need to split new_labels according to original block sizes for tetras+triangles.
    new_cell_data = []
    idx = 0
    for block, tags in zip(msh.cells, msh.cell_data[tag_key]):
        if block.type not in ["tetra", "triangle"]:
            # Keep original tags for other cell types unchanged
            new_cell_data.append(list(tags))
            idx += len(tags)
            continue
        n = len(tags)
        new_cell_data.append(new_labels[idx:idx+n].tolist())
        idx += n

    # Create a new mesh with updated cell data
    new_msh = meshio.Mesh(
        points=msh.points,
        cells=msh.cells,
        cell_data={tag_key: new_cell_data},
        field_data=msh.field_data,
    )

    # ------------------------------------------------------------------ #
    # 6. Write output
    # ------------------------------------------------------------------ #
    print(f"\nWriting remapped mesh to: {output_path}")
    meshio.write(output_path, new_msh)
    print("Done.")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Remap path)


if __name__ == "__main__":
    main()