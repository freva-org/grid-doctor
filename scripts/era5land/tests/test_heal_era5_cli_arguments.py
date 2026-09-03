"""Tests for shared heal_era5 CLI argument definitions."""

from argparse import ArgumentParser

import pytest


def load_arguments_module():
    """Import the package CLI arguments module."""

    from heal_era5.cli import arguments

    return arguments


def test_shared_dataset_frequency_and_interval_arguments():
    """Shared selectors should retain the standard defaults and destinations."""

    arguments = load_arguments_module()
    parser = ArgumentParser()
    arguments.add_dataset_argument(parser)
    arguments.add_variable_argument(parser)
    arguments.add_frequency_argument(parser)
    arguments.add_interval_argument(parser)

    args = parser.parse_args(["--var", "tas,pr", "--interval", "2000,2001"])

    assert args.dataset is None
    assert args.variables == "tas,pr"
    assert args.freq == "all"
    assert args.interval == "2000,2001"


def test_merge_dataset_and_frequency_defaults_can_be_unset():
    """Merge selectors must support direct-store mode without dataset metadata."""

    arguments = load_arguments_module()
    parser = ArgumentParser()
    arguments.add_dataset_argument(parser, default=None)
    arguments.add_frequency_argument(parser, default=None)

    args = parser.parse_args([])

    assert args.dataset is None
    assert args.freq is None


def test_dataset_argument_rejects_unknown_dataset():
    """Dataset choices should be enforced by argparse."""

    arguments = load_arguments_module()
    parser = ArgumentParser()
    arguments.add_dataset_argument(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "unknown"])


def test_publication_arguments_validate_format_and_required_output():
    """Publication options should preserve required paths and Zarr choices."""

    arguments = load_arguments_module()
    parser = ArgumentParser()
    arguments.add_publication_arguments(parser, output_required=True)

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(["--output-path", "/tmp/output", "--zarr-format", "3"])
    assert args.output_path == "/tmp/output"
    assert args.zarr_format == 3
    assert args.chunk_size == arguments.DEFAULT_CHUNK_SIZE


def test_cache_arguments_support_aliases_and_optional_highest_level():
    """Cache aliases should map to stable destinations without duplicate flags."""

    arguments = load_arguments_module()
    parser = ArgumentParser()
    arguments.add_cache_arguments(
        parser,
        weights_dir="/tmp/weights",
        highest_level_help="Only use the finest level.",
        include_highest_level=False,
    )

    args = parser.parse_args(["--no-inventory-cache", "--cache-input-datasets", "-fdt"])
    assert args.use_inventory_cache is False
    assert args.use_input_cache is True
    assert args.fail_on_duplicate_times is True
    assert args.weights_dir == "/tmp/weights"
    assert not hasattr(args, "highest_level_only")


def load_main_module():
    """Import the package CLI module."""

    from heal_era5 import main

    return main


def test_main_parser_orders_commands_and_accepts_remap_modes():
    """The complete parser should expose ordered commands without option conflicts."""

    main = load_main_module()
    parser = main.build_parser()
    args = parser.parse_args(["remap", "-hlo"])
    assert args.highest_level_only is True

    merge_args = parser.parse_args(["merge", "--source", "/tmp/source", "--output-path", "/tmp/output"])
    assert merge_args.dataset is None
    assert merge_args.freq is None
    assert merge_args.variables is None

    help_text = parser.format_help()
    expected = "{clean,fetch,merge,reflow-queue,remap,remap-reflow,update}"
    assert expected in help_text


def test_main_dispatches_normal_commands(monkeypatch):
    """Normal commands should be routed through the handler table."""

    main = load_main_module()
    called = []
    monkeypatch.setattr(main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        main,
        "run_fetch",
        lambda args: called.append(args.command) or 17,
    )

    assert main.main(["fetch", "--dataset", "era5land"]) == 17
    assert called == ["fetch"]


@pytest.mark.parametrize("command", ["fetch", "remap", "update", "clean", "merge"])
def test_bare_normal_commands_print_subcommand_help(capsys, command):
    """A bare command should be equivalent to requesting that command's help."""

    main = load_main_module()

    assert main.main([command]) == 0
    assert f" {command} [-h]" in capsys.readouterr().out


def test_main_dispatches_delegated_commands(monkeypatch):
    """Delegated Reflow commands should receive only their trailing arguments."""

    main = load_main_module()
    calls = []
    monkeypatch.setattr(
        main,
        "run_reflow",
        lambda args: calls.append(("workflow", args)) or 3,
    )
    monkeypatch.setattr(
        main,
        "run_reflow_queue",
        lambda args: calls.append(("queue", args)) or 4,
    )

    assert main.main(["remap-reflow", "status", "run-1"]) == 3
    assert main.main(["reflow-queue", "--plan", "intervals.txt"]) == 4
    assert main.main(["remap-reflow"]) == 3
    assert main.main(["reflow-queue"]) == 4
    assert calls == [
        ("workflow", ["status", "run-1"]),
        ("queue", ["--plan", "intervals.txt"]),
        ("workflow", ["-h"]),
        ("queue", ["-h"]),
    ]
