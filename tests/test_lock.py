import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _torch_wheels():
    """Locked torch wheel file names, keyed by version string."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {
        package["version"]: [wheel["url"].rsplit("/", 1)[-1] for wheel in package["wheels"]]
        for package in lock["package"]
        if package["name"] == "torch"
    }


def _has(wheels, platform_tag):
    return any(name.endswith(f"{platform_tag}.whl") for name in wheels)


def test_lock_pins_both_linux_torch_builds():
    """Linux resolves to the CPU or the CUDA build by extra; without both, a GPU host gets CPU."""
    wheels = _torch_wheels()
    assert set(wheels) == {"2.15.0+cpu", "2.15.0+cu132"}
    assert _has(wheels["2.15.0+cpu"], "manylinux_2_28_x86_64")
    assert _has(wheels["2.15.0+cu132"], "manylinux_2_28_x86_64")
    assert _has(wheels["2.15.0+cu132"], "win_amd64")


def test_linux_torch_extras_are_mutually_exclusive():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(project["project"]["optional-dependencies"]) == {"cpu", "cu132"}
    assert [{"extra": "cpu"}, {"extra": "cu132"}] in project["tool"]["uv"]["conflicts"]
