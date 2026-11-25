# Parallel writes

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

::: grid_doctor.write_pyramid_task

::: grid_doctor.check_pyramid_store

## Manifest and tasks

These classes are frozen dataclasses that round-trip through JSON and
through reflow's manifest codec.

::: grid_doctor.PyramidManifest

::: grid_doctor.WriteTask

::: grid_doctor.io.LevelSpec

::: grid_doctor.io.ArraySpec

## Completeness report

::: grid_doctor.StoreReport

## Store state helpers

These read the manifest and the completion markers below
``<path>/_grid_doctor/``.  The two predicates are cheap enough to serve
as cache checks, e.g. reflow's ``verify=`` hook.

::: grid_doctor.load_manifest

::: grid_doctor.is_current

::: grid_doctor.is_task_done

## JSON layout

The `to_dict()` methods return these typed dictionaries, so the layout
of `manifest.json` and of a serialised task is spelled out in the types.

::: grid_doctor.io.ManifestDict

::: grid_doctor.io.LevelSpecDict

::: grid_doctor.io.ArraySpecDict

::: grid_doctor.io.TaskEntryDict

::: grid_doctor.io.WriteTaskDict

## Exceptions

::: grid_doctor.io.ManifestError

::: grid_doctor.io.StaleTaskError

::: grid_doctor.io.StoreExistsError

::: grid_doctor.io.IncompleteStoreError
