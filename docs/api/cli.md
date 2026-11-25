# Command line

The command-line interface exposes the main `grid_doctor` workflows for
use in scripts and batch processing environments.

## Entry point

::: grid_doctor.cli
    options:
      show_root_full_path: true
      members:
        - get_parser
        - setup_logging_from_args

## Logging

`-v` flags from [`get_parser`][grid_doctor.cli.get_parser] are applied by
`setup_logging_from_args`; outside the CLI, call `setup_logging` directly.

::: grid_doctor.setup_logging
