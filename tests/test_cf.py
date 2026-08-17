"""Tests for `grid_doctor.cf`."""

from __future__ import annotations

import pytest

from grid_doctor.cf import (
    _healpix_cf_attrs,
    CFConventions,
    CFHealpixGridAttrs,
)


class TestCFConventions:
    def test_version(self):
        cf = CFConventions()
        assert cf.Conventions == 'CF-1.13'
        assert cf.to_dict() == { 'Conventions': 'CF-1.13' }


# ===================================================================
# Test CFHealpixGridAttrs
# ===================================================================


class TestCFHealpixGridAttrs:
    @pytest.mark.parametrize(
        "scheme, level", [("nested", 1), ("ring", 2), ("zuniq", None), ("nuniq", None)]
    )
    def test_construct(self, scheme, level):
        attrs = CFHealpixGridAttrs(indexing_scheme=scheme, refinement_level=level)
        assert attrs.indexing_scheme == scheme
        assert attrs.refinement_level == level
        assert attrs.grid_mapping_name == "healpix"

    @pytest.mark.parametrize(
        "scheme,level,msg",
        [
            (
                "nested",
                "1",
                "Invalid `refinement_level`: '1'; Must be integer."
            ),
            ("ring", -1, "Cannot set negative `refinement_level`: '-1'."),
            (
                "zuniq",
                1,
                "`refinement_level` cannot be specified when `indexing_scheme` is 'zuniq'.",
            ),
            (
                "nuniq",
                0.2,
                "`refinement_level` cannot be specified when `indexing_scheme` is 'nuniq'.",
            ),
        ],
    )
    def test_invalid_construct(self, scheme, level, msg):
        with pytest.raises(ValueError, match=msg):
            CFHealpixGridAttrs(indexing_scheme=scheme, refinement_level=level)

    @pytest.mark.parametrize("scheme", ["nested", "ring"])
    def test_mandatory_level_construct(self, scheme):
        with pytest.raises(
            ValueError,
            match=f"Indexing scheme {scheme!r} requires a `refinement_level` to be set."
        ):
            CFHealpixGridAttrs(indexing_scheme=scheme)

    @pytest.mark.parametrize("scheme", ["zuniq", "nuniq"])
    def test_optional_level_construct(self, scheme):
        attrs = CFHealpixGridAttrs(indexing_scheme=scheme)
        assert attrs.grid_mapping_name == "healpix"
        assert attrs.refinement_level == None
        assert attrs.indexing_scheme == scheme

    @pytest.mark.parametrize("scheme", ["NESTED", "bad"])
    def test_unknown_scheme(self, scheme):
        with pytest.raises(
            ValueError,
            match=r"`indexing_scheme` must be one of \['nested', 'nuniq', 'ring', 'zuniq'\], not "
            + f"{scheme!r}",
        ):
            CFHealpixGridAttrs(indexing_scheme=scheme)

    def test_earth_radius(self):
        attrs = CFHealpixGridAttrs(indexing_scheme='zuniq')
        assert attrs.earth_radius is not None
        assert attrs.earth_radius == 6371009


class TestDictIntegrationCFHealpixGridAttrs:
    @pytest.mark.parametrize(
        "scheme, level", [("nested", 1), ("ring", 2), ("zuniq", None), ("nuniq", None)]
    )
    def test_dict_construct(self, scheme, level):
        attrs = CFHealpixGridAttrs(indexing_scheme=scheme, refinement_level=level)
        expected = {"grid_mapping_name": "healpix", "indexing_scheme": scheme, "earth_radius": 6371009 } | (
            {"refinement_level": level} if level else {}
        )
        assert dict(attrs) == expected

    def test_dict_no_radius(self):
        attrs = CFHealpixGridAttrs(indexing_scheme='zuniq',earth_radius=None)
        assert 'earth_radius' not in dict(attrs)
