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
        below = voxels[:, 2] < 0.05
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

def check_stability(items: List[Item], grid: float = 0.015, pyramid_facets: int = 4, g: Vec3 = (0, 0, -9.81), mu: float = 0.5, mass: float = 1.0, verbose: bool = False, plot: bool = True, container_size: Tuple[int, int, int] = None, cached_contacts=None) -> bool:
    # If cached_contacts is provided, use them for the placed items
    if cached_contacts is not None:
        # Assume last item is the candidate
        candidate = items[-1]
        # Compute only candidate's contacts
        candidate_contacts = _generate_contacts([candidate], grid=grid, mu=mu)
        candidate_contacts = _cluster_contacts(candidate_contacts, grid)
        # Adjust indices for candidate contacts (A/B indices)
        for ct in candidate_contacts:
            if ct['A'] >= 0:
                ct['A'] += len(items) - 1  # candidate index
            if ct['B'] >= 0:
                ct['B'] += len(items) - 1
        contacts = list(cached_contacts) + candidate_contacts
    else:
        contacts = _generate_contacts(items, grid=grid, mu=mu)
        contacts = _cluster_contacts(contacts, grid)
    K = len(contacts)
    if K == 0:
        return False
    # Precompute contact arrays
    N = np.array([ct['n'] for ct in contacts])     # (K, 3)
    C = np.array([ct['c'] for ct in contacts])     # (K, 3)
    mu_vals = np.array([ct['mu'] for ct in contacts])  # (K,)
    A_idx = np.array([ct['A'] for ct in contacts])     # (K,)
    B_idx = np.array([ct['B'] for ct in contacts])     # (K,)

    F = cp.Variable((K, 3))  # Contact forces, shape (K, 3)

    constraints = []
    # Normal force constraints
    f_n = cp.sum(cp.multiply(F, N), axis=1)  # (K,)
    constraints.append(f_n >= 0)

    # Friction pyramid constraints
    up_vecs = []
    for n in N:
        if abs(n[2]) < 0.9:
            up_vecs.append([0, 0, 1])
        else:
            up_vecs.append([0, 1, 0])
    up_vecs = np.array(up_vecs)  # (K, 3)
    
    t1s = np.cross(N, up_vecs)
    t1s = np.apply_along_axis(_normalize, 1, t1s)
    t2s = np.cross(N, t1s)
    R = np.stack([t1s, t2s], axis=-1)  # (K, 3, 2)
    dirs = _pyramid_dirs(pyramid_facets)  # (pyramid_facets, 2)
    dir3 = np.einsum('kij,pj->kpi', R, dirs)  # (K, pyramid_facets, 3)
    f_t = F - cp.multiply(f_n[:, None], N)  # (K, 3)
    # Friction cone dot products
    # We'll use cp.matmul for batch dot products
    # f_dot[k, p] = f_t[k] @ dir3[k, p]
    f_dot = cp.hstack([cp.sum(cp.multiply(f_t, dir3[:, p, :]), axis=1, keepdims=True) for p in range(pyramid_facets)])  # (K, pyramid_facets)
    mu_f_n = cp.multiply(mu_vals[:, None], f_n[:, None])  # (K, 1)
    constraints.append(f_dot <= mu_f_n)
    constraints.append(-f_dot <= mu_f_n)

    # Equilibrium constraints (per item)
    for i, item in enumerate(items):
        com = _item_com(item)
        g_force = mass * np.array(g)
        # Find contact indices where A or B == i
        b_mask = (B_idx == i)
        a_mask = (A_idx == i)
        sign = np.where(b_mask, 1, np.where(a_mask, -1, 0))  # +1 for B, -1 for A
        idxs = np.where(sign != 0)[0]
        if len(idxs) == 0:
            continue  # TODO (kaikwan): dive deeper, No contacts for this item, skip
        signs = sign[idxs][:, None]  # (n_i, 1)
        rel_pos = C[idxs] - com  # (n_i, 3)
        F_i = cp.multiply(F[idxs], signs)    # (n_i, 3)
        # Force equilibrium
        constraints.append(cp.sum(F_i, axis=0) + g_force == 0)
        # Torque equilibrium: sum over r × f
        r = rel_pos
        f = F_i
        r_cross_f = cp.hstack([
            cp.multiply(r[:, 1], f[:, 2]) - cp.multiply(r[:, 2], f[:, 1]),
            cp.multiply(r[:, 2], f[:, 0]) - cp.multiply(r[:, 0], f[:, 2]),
            cp.multiply(r[:, 0], f[:, 1]) - cp.multiply(r[:, 1], f[:, 0])
        ])
        constraints.append(cp.sum(r_cross_f, axis=0) == 0)
    print("Solving stability problem with {} contacts...".format(K))
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.HIGHS, verbose=verbose) # , max_iter=1000)
    stable = prob.status == cp.OPTIMAL
    forces = F.value if stable else None
    if plot:
        plot_scene(items, contacts, forces, stable, container_size=container_size)

    # Optionally, return more info for caching
    class ResultObj:
        def __init__(self, stable, contacts):
            self.stable = stable
            self.contacts = contacts
        def __bool__(self):
            return self.stable
    return ResultObj(stable, contacts)

def plot_scene(items: List[Item], contacts, forces_dict, stable, force_scale: float = 0.01, figsize: Tuple[float, float] = (12, 8), container_size: Tuple[int, int, int] = None):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Determine container size (in meters)
    if container_size is None:
        all_voxels = np.vstack([_item_world_voxels(item) for item in items]) if items else np.zeros((1, 3))
        x_max = all_voxels[:, 0].max() if items else 0.2
        y_max = all_voxels[:, 1].max() if items else 0.2
        z_max = all_voxels[:, 2].max() if items else 0.2
        container_size = (int(np.ceil(z_max * 100)), int(np.ceil(x_max * 100)), int(np.ceil(y_max * 100)))
    container_z, container_x, container_y = container_size

    # Colors
    colors = [
        'lightcoral', 'lightsalmon', 'gold', 'olive', 'mediumaquamarine', 'deepskyblue', 'blueviolet', 'pink',
        'brown', 'darkorange', 'yellow', 'lawngreen', 'turquoise', 'dodgerblue', 'darkorchid', 'hotpink',
        'deeppink', 'peru', 'orange', 'darkolivegreen', 'cyan', 'purple', 'crimson'
    ]
    color_rgba_table = np.asarray([plt.matplotlib.colors.to_rgba(c) for c in colors])

    # Build voxel and color arrays
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
        wx = np.clip(wx, 0, container_x - 1)
        wy = np.clip(wy, 0, container_y - 1)
        wz = np.clip(wz, 0, container_z - 1)
        voxels[wz, wx, wy] = True
        facecolors[wz, wx, wy] = color_rgba_table[i % len(color_rgba_table)]

    # Plot voxels
    ax.voxels(
        np.transpose(voxels, (1, 2, 0)),
        facecolors=np.transpose(facecolors, (1, 2, 0, 3)),
        edgecolor=None, alpha=0.3
    )

    # Plot all centers of mass in one call
    if items:
        coms = np.array([_item_com(item) * 100 for item in items])
        ax.scatter(coms[:, 0], coms[:, 1], coms[:, 2], c='red', marker='x', s=200, alpha=0.8)

    # Plot ground plane
    xx, yy = np.meshgrid(
        np.arange(0, container_x),
        np.arange(0, container_y),
        indexing='ij'
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    # Plot all contacts in one call
    if contacts:
        contact_points = np.array([ct['c'] for ct in contacts]) * 100
        ax.scatter(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2],
                   c='orange', marker='o', s=60, alpha=0.8,
                   label='Contacts', edgecolors='black')

        # Vectorized contact normals (blue arrows)
        normals = np.array([ct['n'] for ct in contacts]) * 3
        ax.quiver(
            contact_points[:, 0], contact_points[:, 1], contact_points[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            color='blue', alpha=0.7, arrow_length_ratio=0.3
        )

    # Vectorized force vectors (red arrows)
    if stable and forces_dict is not None and contacts:
        forces = np.array(forces_dict) * (force_scale * 100)
        ax.quiver(
            contact_points[:, 0], contact_points[:, 1], contact_points[:, 2],
            forces[:, 0], forces[:, 1], forces[:, 2],
            color='red', alpha=0.8, arrow_length_ratio=0.1, linewidth=2
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Scene Visualization')
    ax.set_xlim(0, container_x)
    ax.set_ylim(0, container_y)
    ax.set_zlim(0, container_z)
    ax.set_box_aspect([container_x, container_y, container_z])
    ax.view_init(50, 45)
    plt.tight_layout()
    plt.savefig('scene.png')
    return fig, ax