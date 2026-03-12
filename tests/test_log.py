"""Tests for grid_doctor.log."""

from __future__ import annotations

import logging

import pytest

from grid_doctor.log import (
    decrease,
    get_level,
    increase,
    set_level,
    setup_logging,
)


class TestSetupLogging:
    def setup_method(self) -> None:
        """Reset to defaults before each test."""
        setup_logging(verbosity=0)

    def test_default_is_warning(self) -> None:
        setup_logging()
        assert get_level() == logging.WARNING

    def test_verbosity_1(self) -> None:
        setup_logging(verbosity=1)
        assert get_level() == logging.INFO

    def test_verbosity_2(self) -> None:
        setup_logging(verbosity=2)
        assert get_level() == logging.DEBUG

    def test_explicit_level_overrides(self) -> None:
        setup_logging(verbosity=0, level="DEBUG")
        assert get_level() == logging.DEBUG

    def test_explicit_level_int(self) -> None:
        setup_logging(level=logging.ERROR)
        assert get_level() == logging.ERROR

    def test_clamps_high_verbosity(self) -> None:
        setup_logging(verbosity=99)
        assert get_level() == logging.DEBUG

    def test_adds_handler(self) -> None:
        # Clear handlers first
        logging.root.handlers.clear()
        setup_logging()
        assert len(logging.root.handlers) >= 1


class TestSetLevel:
    def setup_method(self) -> None:
        setup_logging(verbosity=0)

    def test_by_int(self) -> None:
        set_level(logging.DEBUG)
        assert get_level() == logging.DEBUG

    def test_by_string(self) -> None:
        set_level("error")
        assert get_level() == logging.ERROR

    def test_case_insensitive(self) -> None:
        set_level("DeBuG")
        assert get_level() == logging.DEBUG

    def test_applies_to_existing_loggers(self) -> None:
        logger = logging.getLogger("test.set_level.existing")
        set_level(logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_applies_to_new_loggers(self) -> None:
        set_level(logging.ERROR)
        logger = logging.getLogger("test.set_level.new")
        assert logger.level == logging.ERROR

    def test_non_standard_level_snaps_closest(self) -> None:
        set_level(25)  # between INFO (20) and WARNING (30)
        assert get_level() in (logging.INFO, logging.WARNING)


class TestIncrease:
    def setup_method(self) -> None:
        setup_logging(verbosity=0)

    def test_one_step(self) -> None:
        new = increase()
        assert new == logging.INFO
        assert get_level() == logging.INFO

    def test_two_steps(self) -> None:
        increase(2)
        assert get_level() == logging.DEBUG

    def test_clamps_at_debug(self) -> None:
        increase(99)
        assert get_level() == logging.DEBUG

    def test_returns_level(self) -> None:
        level = increase()
        assert level == get_level()


class TestDecrease:
    def setup_method(self) -> None:
        setup_logging(verbosity=0)

    def test_one_step(self) -> None:
        new = decrease()
        assert new == logging.ERROR
        assert get_level() == logging.ERROR

    def test_clamps_at_critical(self) -> None:
        decrease(99)
        assert get_level() == logging.CRITICAL

    def test_returns_level(self) -> None:
        level = decrease()
        assert level == get_level()


class TestRoundTrip:
    def test_increase_then_decrease(self) -> None:
        setup_logging(verbosity=0)
        assert get_level() == logging.WARNING

        increase()
        assert get_level() == logging.INFO

        decrease()
        assert get_level() == logging.WARNING

    def test_full_ladder(self) -> None:
        """Walk all the way up and back down."""
        setup_logging(verbosity=0)

        increase(4)
        assert get_level() == logging.DEBUG

        decrease(4)
        assert get_level() == logging.CRITICAL
