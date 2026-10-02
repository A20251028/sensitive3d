import numpy as np

from conftest import needs_bridge
from sensitive3d.core.mesh import MeshFile, MeshPart, Texture
from sensitive3d.io.gltf import read_gltf, write_gltf
from sensitive3d.io.obj import read_obj, write_obj


def _mesh():
    rng = np.random.default_rng(0)
    v = rng.random((4, 3)) * 10
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    img = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    part = MeshPart(v, np.array([[0, 1, 2], [0, 2, 3]]), uvs=uv, texture=0, name="m0")
    return MeshFile("x", "obj", [part], [Texture(img, name="t.png", encoding="png")])


def test_obj_roundtrip(tmp_path):
    m = _mesh()
    write_obj(m, tmp_path / "a.obj")
    r = read_obj(tmp_path / "a.obj")
    assert len(r.parts) == 1 and len(r.parts[0].faces) == 2
    assert np.allclose(np.sort(r.parts[0].vertices, axis=0), np.sort(m.parts[0].vertices, axis=0), atol=1e-5)
    assert np.array_equal(r.textures[0].image, m.textures[0].image)


def test_gltf_roundtrip_keeps_z_up(tmp_path):
    m = _mesh()
    write_gltf(m, tmp_path / "a.glb")
    r = read_gltf(tmp_path / "a.glb")
    lo, hi = r.bounds()
    lo0, hi0 = m.bounds()
    assert np.allclose(lo, lo0, atol=1e-5) and np.allclose(hi, hi0, atol=1e-5)


@needs_bridge
def test_osgb_build_export_patch(tmp_path):
    from PIL import Image

    from sensitive3d.io.osgb import build_osgb, read_osgb, scan_osgb, write_osgb

    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    t = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    v.tofile(tmp_path / "v.f32")
    uv.tofile(tmp_path / "uv.f32")
    t.tofile(tmp_path / "t.u32")
    img = np.zeros((32, 64, 3), np.uint8)
    img[:16, :, 0] = 255
    Image.fromarray(img).save(tmp_path / "tex.jpg", quality=90)
    desc = "BEGIN_PAGEDLOD 0.5 0.5 0 1 1\nCHILD 0 100 BEGIN_GEODE GEOMETRY v.f32 uv.f32 - t.u32 tex.jpg END_GEODE\nFILE_CHILD fine.osgb 100 1e30\nEND_PAGEDLOD\n"
    build_osgb(desc, tmp_path, tmp_path / "a.osgb")

    info = scan_osgb([tmp_path / "a.osgb"])[0]
    assert info["ok"] and info["children"] == ["fine.osgb"] and info["geometries"][0]["has_finer"]

    m = read_osgb(tmp_path / "a.osgb")
    assert len(m.parts) == 1 and m.parts[0].has_finer
    assert m.textures[0].image[2, 2, 0] > 200  # top row is v == 1

    # geometry-only patch: the untouched JPEG must be embedded byte for byte
    m.parts[0].vertices[2, 2] = 0.5
    m.parts[0].dirty = True
    write_osgb(m, tmp_path / "a.osgb", tmp_path / "b.osgb")
    assert (tmp_path / "tex.jpg").read_bytes() in (tmp_path / "b.osgb").read_bytes()
    m2 = read_osgb(tmp_path / "b.osgb")
    assert abs(m2.parts[0].vertices[2, 2] - 0.5) < 1e-6
    assert m2.meta["children"] == ["fine.osgb"]

    # texture patch
    m2.textures[0].image[:] = (0, 200, 0)
    m2.textures[0].dirty = True
    write_osgb(m2, tmp_path / "b.osgb", tmp_path / "c.osgb")
    m3 = read_osgb(tmp_path / "c.osgb")
    assert abs(int(m3.textures[0].image[5, 5, 1]) - 200) < 6
