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
            "save_pyramid_to_s3",
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
