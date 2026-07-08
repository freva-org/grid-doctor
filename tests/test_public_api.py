from __future__ import annotations

import inspect

import pytest

import grid_doctor as gd


class TestPublicNamespace:
    def test_public_api_names_are_listed_in_dir(self) -> None:
        names = set(dir(gd))
        missing = [name for name in gd.__all__ if name not in names]
        assert missing == []

    @pytest.mark.parametrize("name", gd.__all__)
    def test_public_api_names_resolve(self, name: str) -> None:
        obj = getattr(gd, name)
        assert obj is not None

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            getattr(gd, "definitely_not_a_real_grid_doctor_symbol")


class TestPublicCallables:
    @pytest.mark.parametrize(
        "name",
        [
            "apply_weight_file",
            "cached_open_dataset",
            "cached_weights",
            "chunk_for_target_store_size",
            "coarsen_healpix",
            "compute_healpix_weights",
            "create_healpix_pyramid",
            "get_latlon_resolution",
            "get_s3_options",
            "latlon_to_healpix_pyramid",
            "regrid_to_healpix",
            "regrid_unstructured_to_healpix",
            "resolution_to_healpix_level",
            "save_pyramid",
            "setup_logging",
        ],
    )
    def test_public_callables_have_signatures(self, name: str) -> None:
        obj = getattr(gd, name)
        signature = inspect.signature(obj)
        assert signature is not None

    def test_repeated_lookup_returns_same_object(self) -> None:
        first = gd.regrid_to_healpix
        second = gd.regrid_to_healpix
        assert first is second


class TestStubFileCompleteness:
    """The ``__init__.pyi`` stub must mirror the lazy loader.

    Regenerate it with::

        stubgen -m grid_doctor -o src

    (stubgen reads the ``TYPE_CHECKING`` block, so keep that block in
    sync with ``_SUBMODULES`` / ``_ATTRS`` as well.)
    """

    @staticmethod
    def _stub_exports() -> set[str]:
        import ast
        from pathlib import Path

        import grid_doctor

        stub = Path(grid_doctor.__file__).with_suffix(".pyi")
        names: set[str] = set()
        for node in ast.parse(stub.read_text()).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                names.add(node.target.id)
        return names

    def test_stub_covers_public_api(self) -> None:
        import grid_doctor

        missing = set(grid_doctor.__all__) - self._stub_exports() - {"__all__"}
        assert not missing, (
            f"__init__.pyi is missing {sorted(missing)}; "
            "regenerate with `stubgen -m grid_doctor -o src`."
        )

    def test_stub_has_no_stale_exports(self) -> None:
        import grid_doctor

        stale = self._stub_exports() - set(grid_doctor.__all__)
        assert not stale, f"__init__.pyi exports non-public names: {sorted(stale)}"
