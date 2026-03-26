import pytest

from scripts.icdc import collections


class TestICDC:
    def test_collections(self):
        assert len(collections) == 7

    @pytest.mark.parametrize("collections", collections)
    @pytest.mark.skip(reason="Not implemented (yet?!)")
    def test_spec(self, collections):
        pass  # TODO
