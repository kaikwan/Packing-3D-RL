import numpy as np
import cvxpy as cp
from scipy.spatial import cKDTree
from itertools import combinations
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import time

from .object import Item

Vec3 = Tuple[float, float, float]

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _item_world_voxels(item: Item) -> np.ndarray:
    # Extract world voxel positions from item.curr_geometry.cube and item.position
    cube = item.curr_geometry.cube
    voxels = []
    for z in range(cube.shape[0]):
        for x in range(cube.shape[1]):
            for y in range(cube.shape[2]):
                if cube[z][x][y] == 1:
                    # Convert to meters (assume 1 unit = 1cm)
                    wx = (x + item.position.x) * 0.01
                    wy = (y + item.position.y) * 0.01
                    wz = (z + item.position.z) * 0.01
                    voxels.append([wx, wy, wz])
    return np.array(voxels)

def _item_com(item: Item) -> np.ndarray:
    voxels = _item_world_voxels(item)
    return voxels.mean(axis=0)

def _item_kdtree(item: Item):
    return cKDTree(_item_world_voxels(item))

def _generate_contacts(items: List[Item], grid: float = 0.01, mu: float = 0.5):
    contacts = []
    # Body-to-body contacts
    for i, j in combinations(range(len(items)), 2):
        tree_i = _item_kdtree(items[i])
        tree_j = _item_kdtree(items[j])
        contact_dist = 2 * grid
        pairs = tree_i.query_ball_tree(tree_j, r=contact_dist)
        contacts_found = []
        voxels_i = _item_world_voxels(items[i])
        voxels_j = _item_world_voxels(items[j])
        for vi, vjs in enumerate(pairs):
            if vjs:
                for vj in vjs:
                    pi = voxels_i[vi]
                    pj = voxels_j[vj]
                    mid = (pi + pj) * 0.5
                    n = _normalize(pj - pi)
                    contacts_found.append((mid, n))
        if contacts_found:
            positions = np.array([c[0] for c in contacts_found])
            normals = np.array([c[1] for c in contacts_found])
            for pos, normal in zip(positions, normals):
                contacts.append(dict(A=i, B=j, c=pos, n=normal, mu=mu))
    # Ground contacts (z = 0)
    for idx, item in enumerate(items):
        voxels = _item_world_voxels(item)
        below = voxels[:, 2] < 1e-6
        if np.any(below):
            pts = voxels[below]
            for p in pts:
                contacts.append(dict(A=-1, B=idx, c=p, n=np.array([0, 0, 1]), mu=mu))
    return contacts

def _cluster_contacts(contacts, grid):
    buckets: Dict[Tuple[int, int, int], List[dict]] = {}
    for ct in contacts:
        key = tuple((ct['c'] / grid).round().astype(int))
        buckets.setdefault(key, []).append(ct)
    merged = []
    for group in buckets.values():
        A, B = group[0]['A'], group[0]['B']
        c = np.mean([g['c'] for g in group], axis=0)
        n = _normalize(np.mean([g['n'] for g in group], axis=0))
        mu = np.mean([g['mu'] for g in group])
        merged.append(dict(A=A, B=B, c=c, n=n, mu=mu))
    return merged

def _pyramid_dirs(n):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(ang), np.sin(ang)]

def check_stability(items: List[Item], grid: float = 0.01, pyramid_facets: int = 1, g: Vec3 = (0, 0, -9.81), mu: float = 0.5, mass: float = 1.0, verbose: bool = True, plot: bool = True, container_size: Tuple[int, int, int] = None) -> bool:
    contacts = _generate_contacts(items, grid=grid, mu=mu)
    contacts = _cluster_contacts(contacts, grid)
    K = len(contacts)
    if K == 0:
        return False
    F = cp.Variable((K, 3))
    constraints = []
    circle = _pyramid_dirs(pyramid_facets)
    # Friction pyramid constraints
    for k, ct in enumerate(contacts):
        n = ct['n']
        mu_val = ct['mu']
        f_n = F[k] @ n
        f_t = F[k] - f_n * n
        constraints.append(f_n >= 0)
        t1 = _normalize(np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.cross(n, [0, 1, 0]))
        t2 = np.cross(n, t1)
        R = np.vstack([t1, t2])
        for d in circle:
            dir3 = R.T @ d
            constraints.append(f_t @ dir3 <= mu_val * f_n)
            constraints.append(-f_t @ dir3 <= mu_val * f_n)
    g_vec = np.asarray(g)
    for idx, item in enumerate(items):
        gravity_force = mass * g_vec
        sum_f = np.array([gravity_force[0], gravity_force[1], gravity_force[2]])
        sum_tau = np.array([0.0, 0.0, 0.0])
        has_contacts = False
        com = _item_com(item)
        for k, ct in enumerate(contacts):
            if ct['B'] == idx:
                fk = F[k]
                r = ct['c'] - com
                has_contacts = True
            elif ct['A'] == idx:
                fk = -F[k]
                r = ct['c'] - com
                has_contacts = True
            else:
                continue
            sum_f = sum_f + fk
            sum_tau = sum_tau + np.array([
                r[1]*fk[2] - r[2]*fk[1],
                r[2]*fk[0] - r[0]*fk[2],
                r[0]*fk[1] - r[1]*fk[0]
            ])
        if not has_contacts:
            return False
        constraints += [sum_f[i] == 0 for i in range(3)]
        constraints += [sum_tau[i] == 0 for i in range(3)]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    stable = prob.status == cp.OPTIMAL
    forces = F.value if stable else None
    if plot:
        plot_scene(items, contacts, forces, stable, container_size=container_size)
    return stable

def plot_scene(items: List[Item], contacts, forces_dict, stable, force_scale: float = 0.01, figsize: Tuple[float, float] = (12, 8), container_size: Tuple[int, int, int] = None):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    # Determine container size (in meters)
    container_z, container_x, container_y = container_size
    colors = [
        'lightcoral', 'lightsalmon', 'gold', 'olive', 'mediumaquamarine', 'deepskyblue', 'blueviolet', 'pink',
        'brown', 'darkorange', 'yellow', 'lawngreen', 'turquoise', 'dodgerblue', 'darkorchid', 'hotpink',
        'deeppink', 'peru', 'orange', 'darkolivegreen', 'cyan', 'purple', 'crimson'
    ]
    color_rgba_table = np.asarray([plt.matplotlib.colors.to_rgba(c) for c in colors])
    # Build a single voxel and color array for the whole container
    voxels = np.zeros((container_z, container_x, container_y), dtype=bool)
    facecolors = np.zeros((container_z, container_x, container_y, 4), dtype=float)
    for i, item in enumerate(items):
        cube = item.curr_geometry.cube
        active = cube > 0
        if not np.any(active):
            continue
        z_idx, x_idx, y_idx = np.where(active)
        wx = (x_idx + item.position.x)
        wy = (y_idx + item.position.y)
        wz = (z_idx + item.position.z)
        # Clamp indices to container size
        wx = np.clip(wx, 0, container_x - 1)
        wy = np.clip(wy, 0, container_y - 1)
        wz = np.clip(wz, 0, container_z - 1)
        voxels[wz, wx, wy] = True
        facecolors[wz, wx, wy] = color_rgba_table[i % len(color_rgba_table)]
    # Debug: print number of set voxels and their coordinates
    num_voxels = np.sum(voxels)
    # Transpose to (x, y, z) for ax.voxels
    ax.voxels(
        np.transpose(voxels, (1, 2, 0)),
        facecolors=np.transpose(facecolors, (1, 2, 0, 3)),
        edgecolor=None, alpha=0.3
    )
    # Plot center of mass for each item
    for item in items:
        com = _item_com(item) * 100  # Convert from meters to voxel units
        ax.scatter(com[0], com[1], com[2], c='red', marker='x', s=200, alpha=0.8)
    # Plot ground plane
    xx, yy = np.meshgrid(
        np.arange(0, container_x),
        np.arange(0, container_y),
        indexing='ij'
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    # Plot contacts
    if contacts:
        contact_points = np.array([ct['c'] for ct in contacts]) * 100  # Convert to voxel units
        ax.scatter(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2],
                   c='orange', marker='o', s=200, alpha=0.8,
                   label='Contacts', edgecolors='black')
        # Plot contact normals
        for ct in contacts:
            c = ct['c'] * 100  # Convert to voxel units
            n = ct['n'] * 3   # Make the normal arrow shorter for visibility (3 voxels)
            ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2],
                      color='blue', alpha=0.7, arrow_length_ratio=0.3)
    # Plot force vectors
    if stable and forces_dict is not None and contacts:
        for k, force in enumerate(forces_dict):
            if k < len(contacts):
                ct = contacts[k]
                c = ct['c'] * 100  # Convert to voxel units
                f = force * (force_scale * 100)  # Scale force to voxel units
                ax.quiver(c[0], c[1], c[2], f[0], f[1], f[2],
                          color='red', alpha=0.8, arrow_length_ratio=0.1,
                          linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Scene Visualization')
    # Set aspect ratio and limits
    ax.set_xlim(0, container_x)
    ax.set_ylim(0, container_y)
    ax.set_zlim(0, container_z)
    ax.set_box_aspect([container_x, container_y, container_z])
    ax.view_init(50, 45)
    plt.tight_layout()
    plt.savefig('scene.png')
    return fig, ax 