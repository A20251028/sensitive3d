import numpy as np

from sensitive3d.core.camera import OrthoCamera
from sensitive3d.core.mesh import MeshPart, Texture
from sensitive3d.core.raster import RasterBuffers, rasterize_uv
from sensitive3d.core.render import RenderItem, render
from sensitive3d.core.texture import texel_map, texel_size


def test_raster_coverage_and_barycentrics():
    xy = np.array([[2.0, 2.0], [62.0, 2.0], [2.0, 62.0]])
    buf = RasterBuffers(64, 64)
    buf.draw(xy, np.zeros(3), np.array([[0, 1, 2]]))
    assert abs(buf.valid.sum() - 0.5 * 60 * 60) < 80
    b = buf.bary[buf.valid]
    assert np.allclose(b.sum(axis=1), 1.0)
    assert (b > -1e-6).all()


def test_depth_test_keeps_nearest():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    buf = RasterBuffers(10, 10)
    buf.draw(xy, np.full(4, 5.0), np.array([[0, 1, 2], [1, 3, 2]]), part_id=0)
    buf.draw(xy, np.full(4, 1.0), np.array([[0, 1, 2], [1, 3, 2]]), part_id=1)
    assert (buf.part[buf.valid] == 1).all()


def _quad(size=1.0):
    v = np.array([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return MeshPart(v, np.array([[0, 1, 2], [0, 2, 3]]), uvs=uv, texture=0)


def test_render_texture_orientation():
    img = np.zeros((32, 32, 3), np.uint8)
    img[:16] = (255, 0, 0)  # top half (v > 0.5) red
    img[16:] = (0, 0, 255)
    cam = OrthoCamera.looking([0.5, 0.5, 1.0], [0, 0, -1], 1.0, 1.0, 0.02, up_hint=[0, 1, 0])
    res = render([RenderItem(_quad(), img)], cam)
    assert res.color[5, 25, 0] > 200 and res.color[45, 25, 2] > 200
    assert np.allclose(res.position[25, 25], [0.51, 0.49, 0.0], atol=0.02)


def test_texel_map_lifts_texels_to_surface():
    tex = Texture(np.zeros((64, 64, 3), np.uint8))
    part = _quad(2.0)
    tm = texel_map(part, tex)
    assert len(tm) == 64 * 64
    assert tm.positions[:, 0].min() >= 0 and tm.positions[:, 0].max() <= 2.0
    assert abs(texel_size(part, tex) - 2.0 / 64) < 1e-3


def test_rasterize_uv_owner():
    buf = rasterize_uv(np.array([[0, 0], [8, 0], [0, 8.0]]), np.array([[0, 1, 2]]), 8, 8)
    assert buf.face[0, 0] == 0 and buf.face[7, 7] == -1
