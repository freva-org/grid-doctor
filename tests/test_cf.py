"""Tests for `grid_doctor.cf`."""

from __future__ import annotations

import pytest

from grid_doctor.cf import (
    _healpix_cf_attrs,
    CFKey,
)

# ===================================================================
#
# ===================================================================


class TestCFAttributes:
    @pytest.mark.parametrize(
        "scheme,level,expected",
        [
            (
                "ring",
                0,
                {
                    "grid_mapping_name": "healpix",
                    "indexing_scheme": "ring",
                    "refinement_level": "0",
                },
            ),
            (
                "nested",
                1,
                {
                    "grid_mapping_name": "healpix",
                    "indexing_scheme": "nested",
                    "refinement_level": "1",
                },
            ),
            (
                "nuniq",
                2,
                {
                    "grid_mapping_name": "healpix",
                    "indexing_scheme": "nuniq",
                },
            ),
            (
                "zuniq",
                None,
                {
                    "grid_mapping_name": "healpix",
                    "indexing_scheme": "zuniq",
                },
            ),
        ],
        ids=["ring", "nested", "nuniq", "zuniq"],
    )
    def test_healpix_cf_attrs(self, scheme, level, expected) -> None:
        r = _healpix_cf_attrs(scheme, level)
        assert r == expected

    @pytest.mark.parametrize(
        "level,match",
        [
            (0.0, r"Cannot set invalid refinement level: '0.0' \(must be integer\)\."),
            ("0", r"Cannot set invalid refinement level: '0' \(must be integer\)\."),
            (-1, r"Cannot set negative refinement level: '-1'"),
        ],
        ids=["float", "str", "-negative"],
    )
    def test_healpix_cf_attrs_error_level(self, level, match) -> None:
        with pytest.raises(ValueError, match=match):
            _healpix_cf_attrs("ring", level)

    @pytest.mark.parametrize(
        "scheme,match",
        [
            ("RING", r"Cannot attach unsupported scheme.*"),
        ],
        ids=[
            "upper",
        ],
    )
    def test_healpix_cf_attrs_error_scheme(self, scheme, match) -> None:
        with pytest.raises(ValueError, match=match):
            _healpix_cf_attrs(scheme, 0)

    @pytest.mark.parametrize(
        "scheme",
        [
            "ring",
            "nested",
        ],
    )
    def test_healpix_cf_attr_mandatory_level(self, scheme):
        with pytest.raises(
            ValueError,
            match=f"Indexing scheme {scheme} requires a valid healpix refinement level.",
        ):
            _healpix_cf_attrs(scheme, None)
