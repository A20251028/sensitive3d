import numpy as np

from sensitive3d.repair.geometry import find_free_rect, new_boundary_loops, triangulate_loop
from sensitive3d.repair.inpaint import inpaint


def _grid(n=6):
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    v = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1).astype(float)
    f = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
            f += [[a, b, c], [a, c, d]]
    return v, np.array(f)


def test_hole_loop_and_fill():
    v, f = _grid()
    c = v[f].mean(axis=1)
    hole = (c[:, 0] > 2) & (c[:, 0] < 4) & (c[:, 1] > 2) & (c[:, 1] < 4)
    kept = f[~hole]
    loops = new_boundary_loops(f, kept, v)
    assert len(loops) == 1 and len(loops[0]) == 8
    tri = triangulate_loop(v[loops[0]], np.array([0, 0, 1.0]))
    assert len(tri) == 6
    # the outer border of the grid is not a "new" hole
    assert all(np.all((v[l][:, :2] >= 2) & (v[l][:, :2] <= 4)) for l in loops)


def test_find_free_rect():
    occ = np.ones((20, 30), bool)
    occ[5:12, 10:25] = False
    assert find_free_rect(occ, 10, 5) == (10, 5)
    assert find_free_rect(occ, 16, 5) is None


def test_inpaint_fills_with_surrounding_texture():
    rng = np.random.default_rng(1)
    img = np.empty((120, 160, 3), np.uint8)
    img[:] = (150, 140, 120)
    img += rng.integers(0, 20, img.shape[:2] + (1,)).astype(np.uint8)
    img[40:80, 50:110] = (20, 70, 160)
    mask = np.zeros(img.shape[:2], bool)
    mask[36:84, 46:114] = True
    out = inpaint(img, mask)
    filled = out[mask].astype(float)
    ring = img[~mask].astype(float)
    assert np.abs(filled.mean(axis=0) - ring.mean(axis=0)).max() < 12
    assert filled[:, 2].max() < 160  # no blue left
