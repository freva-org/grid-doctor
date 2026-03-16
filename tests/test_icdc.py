import pytest

from scripts.icdc import specs


class TestICDC:
    def test_specs(self):
        assert len(specs) == 10

    @pytest.mark.parametrize("spec", specs)
    @pytest.mark.skip(reason="Not implemented (yet?!)")
    def test_spec(self, spec):
        pass  # TODO
