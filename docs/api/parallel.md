# Parallel Writes

Functions and data classes for writing a pyramid with many independent
workers, for example as an HPC array job.  The workflow is

1. [`init_pyramid_store`][grid_doctor.init_pyramid_store] creates empty
   level stores and returns a
   [`PyramidManifest`][grid_doctor.PyramidManifest] with one
   [`WriteTask`][grid_doctor.WriteTask] per unit of work,
2. every worker calls
   [`write_pyramid_task`][grid_doctor.write_pyramid_task] for its task,
3. [`check_pyramid_store`][grid_doctor.check_pyramid_store] reports
   whether all tasks finished.

The [Parallel writes recipe](../recipes/parallel-writes.md) explains the
design, the manifest format and a complete reflow workflow.

## Workflow steps

::: grid_doctor.init_pyramid_store
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.write_pyramid_task
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.check_pyramid_store
    options:
      show_root_heading: true
      show_root_full_path: false

## Manifest and tasks

These classes are frozen dataclasses that round-trip through JSON and
through reflow's manifest codec.

::: grid_doctor.PyramidManifest
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.WriteTask
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.io.LevelSpec
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.io.ArraySpec
    options:
      show_root_heading: true
      show_root_full_path: false

## Completeness report

::: grid_doctor.StoreReport
    options:
      show_root_heading: true
      show_root_full_path: false

## Store state helpers

These read the manifest and the completion markers below
``<path>/_grid_doctor/``.  The two predicates are cheap enough to serve
as cache checks, e.g. reflow's ``verify=`` hook.

::: grid_doctor.load_manifest
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.is_current
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.is_task_done
    options:
      show_root_heading: true
      show_root_full_path: false

## Exceptions

::: grid_doctor.io.ManifestError
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.io.StaleTaskError
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.io.StoreExistsError
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.io.IncompleteStoreError
    options:
      show_root_heading: true
      show_root_full_path: false
