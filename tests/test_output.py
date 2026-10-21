import pytest

from pangu_weather.output import copy_products


def test_interrupted_copy_preserves_published_product(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    case = tmp_path / "case"
    shared.mkdir()
    case.mkdir()
    for name in ("surface.nc", "upper.nc"):
        (shared / name).write_bytes(b"new product")
        (case / name).write_bytes(b"previous product")

    def interrupted_copy(source, destination):
        destination.write_bytes(b"incomplete")
        raise OSError("injected copy failure")

    monkeypatch.setattr("pangu_weather.output.shutil.copyfile", interrupted_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        copy_products(case, shared)
    for name in ("surface.nc", "upper.nc"):
        assert (case / name).read_bytes() == b"previous product"
        assert (shared / name).read_bytes() == b"new product"
    assert not list(case.glob("*.partial"))
