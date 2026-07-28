#!/usr/bin/env python3
"""
Script to plot volume fraction of extracellular and intracellular space
vs number of cells from simulation results.

Reads meshstatistic.yml files from
results/cells_cube_highres_volfrac/soma/size30000_ncells<NCELLS>_0/meshes/
and plots ECS and ICS volume fraction as a function of NCELLS.
"""

import os
import re
import yaml
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Base directory where the results are stored
    base_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        'emimesh',
        'results',
        'cells_cube_highres_volfrac',
        'soma'
    )

    # List to store data
    ncells_list = []
    ecs_frac_list = []
    ics_frac_list = []

    print(f"Searching in {base_dir}")
    # Iterate over subdirectories in base_dir
    for dirname in sorted(os.listdir(base_dir)):
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        # Extract number of cells from directory name
        # Pattern: size30000_ncells<number>_0
        match = re.search(r'ncells(\d+)_', dirname)
        if not match:
            print(f"Skipping directory {dirname}: does not match pattern")
            continue
        ncells = int(match.group(1))

        # Path to meshstatistic.yml
        stats_file = os.path.join(dirpath, 'meshes', 'meshstatistic.yml')
        if not os.path.exists(stats_file):
            print(f"Warning: {stats_file} not found, skipping")
            continue

        # Load the YAML file
        with open(stats_file, 'r') as f:
            stats = yaml.safe_load(f)

        # Extract extracellular volume fraction (ecs_share)
        ecs_share = stats.get('ecs_share')
        if ecs_share is None:
            print(f"Warning: ecs_share not found in {stats_file}, skipping")
            continue

        # Intracellular volume fraction is 1 - ecs_share
        ics_share = 1.0 - ecs_share

        # Store data
        ncells_list.append(ncells)
        ecs_frac_list.append(ecs_share)
        ics_frac_list.append(ics_share)
        print(f"Processed {dirname}: ncells={ncells}, ecs_share={ecs_share:.4f}, ics_share={ics_share:.4f}")

    # Check if we have any data
    if len(ncells_list) == 0:
        print("Error: No data found. Exiting.")
        return

    # Sort by number of cells (just in case)
    sorted_indices = np.argsort(ncells_list)
    ncells_list = [ncells_list[i] for i in sorted_indices]
    ecs_frac_list = [ecs_frac_list[i] for i in sorted_indices]
    ics_frac_list = [ics_frac_list[i] for i in sorted_indices]

    print(f"\nFound {len(ncells_list)} data points.")
    print("ncells:", ncells_list)
    print("ecs_frac:", ecs_frac_list)
    print("ics_frac:", ics_frac_list)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ncells_list, ecs_frac_list, 'o-', label='Extracellular (ECS)')
    plt.plot(ncells_list, ics_frac_list, 's-', label='Intracellular (ICS)')
    plt.xlabel('Number of Cells')
    plt.ylabel('Volume Fraction')
    plt.title('Volume Fraction of Extracellular and Intracellular Space vs Number of Cells')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, 'volume_fraction_vs_ncells.png')
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

    # Optionally show the plot (uncomment if running interactively)
    # plt.show()

if __name__ == '__main__':
    main()