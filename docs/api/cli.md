# Command-line Interface

The command-line interface exposes the main `grid_doctor` workflows for
use in scripts and batch processing environments.

## Entry point

::: grid_doctor.cli
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - get_parser
        - setup_logging_from_args
