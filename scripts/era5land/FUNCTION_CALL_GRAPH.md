# ERA5-Land converter call graph

This document maps the commands dispatched by `converter.py` to the local
functions that do the work.  It deliberately groups standard-library, xarray,
zarr, and `grid_doctor` calls under their local wrapper, so the graph remains
useful as a code-navigation guide.

## How to read this document

The first five diagrams are **orientation views**.  They are not separate
executions and they intentionally omit repeated detail: each gives one useful
slice of the overall system.  The **Detailed command graphs** section starts
at [Fetch](#fetch) and expands every command path.

| Color | Module that owns the function |
| --- | --- |
| <span style="display:inline-block;width:4rem;background:#e5e7eb;border:1px solid #4b5563">&nbsp;</span> Gray `#e5e7eb` | `converter.py`: command parsing and orchestration. |
| <span style="display:inline-block;width:4rem;background:#fef3c7;border:1px solid #d97706">&nbsp;</span> Amber `#fef3c7` | `helpers/file_fetcher.py`: request configuration and source-file resolution. |
| <span style="display:inline-block;width:4rem;background:#fde68a;border:1px solid #ca8a04">&nbsp;</span> Gold `#fde68a` | `helpers/formatter.py`: frequency, level, and destination formatting. |
| <span style="display:inline-block;width:4rem;background:#fee2e2;border:1px solid #dc2626">&nbsp;</span> Red `#fee2e2` | `helpers/cleanup.py`: truncation and deletion. |
| <span style="display:inline-block;width:4rem;background:#dbeafe;border:1px solid #2563eb">&nbsp;</span> Blue `#dbeafe` | `helpers/mapper.py`: remapping, coarsening, and zoom-level writes. |
| <span style="display:inline-block;width:4rem;background:#ede9fe;border:1px solid #7c3aed">&nbsp;</span> Violet `#ede9fe` | `helpers/datasets.py`: opening, normalising, and merging datasets. |
| <span style="display:inline-block;width:4rem;background:#ffedd5;border:1px solid #ea580c">&nbsp;</span> Orange `#ffedd5` | `helpers/special.py`: special (`fx`) variables. |
| <span style="display:inline-block;width:4rem;background:#dcfce7;border:1px solid #16a34a">&nbsp;</span> Green `#dcfce7` | `helpers/zarr_publisher.py`: Zarr publication, metadata sync, and source-store merge. |
| <span style="display:inline-block;width:4rem;background:#fce7f3;border:1px solid #db2777">&nbsp;</span> Pink `#fce7f3` | `cli/reflow_workflow.py`: parallel reflow planning and workers. |
| <span style="display:inline-block;width:4rem;background:#cffafe;border:1px solid #0891b2">&nbsp;</span> Cyan `#cffafe` | External `grid_doctor` calls. |
| <span style="display:inline-block;width:4rem;background:#cbd5e1;border:1px solid #475569">&nbsp;</span> Slate `#cbd5e1` | `cli.arguments.py`, `helpers.metadata.py`, and `helpers.grib.py`: small supporting modules. |

## Command entry points

This is the top-level dispatcher.  It answers: “which command starts which
major workflow?”, without showing their internal helper calls.

```mermaid
flowchart LR
    CLI[converter.py main]:::converter --> FETCH[run_fetch]:::converter
    CLI --> REMAP[run_remap]:::converter
    CLI --> UPDATE[run_update]:::converter
    CLI --> CLEAN[run_clean]:::converter
    CLI --> MERGE[run_merge]:::converter
    CLI --> REFLOW[run_reflow]:::converter

    FETCH --> REQUESTS[selected_requests]:::converter
    FETCH --> RESOLVE[resolve_records]:::file_fetcher

    REMAP --> REQUESTS
    REMAP --> RESOLVE
    REMAP --> BATCH[build_file_batch_plan]:::converter
    REMAP --> MAP[map_records]:::converter
    REMAP --> TRUNCATE[truncate_existing_healpix_stores]:::cleanup
    REMAP --> RECHUNK[rechunk_existing_healpix_stores]:::mapper

    UPDATE --> REQUESTS
    UPDATE --> LAST[_existing_variable_last_date]:::converter
    UPDATE --> UPDATE_RECORDS[_resolve_update_records]:::converter
    UPDATE --> PERMANENT[_apply_permanent_update]:::converter
    UPDATE --> FORWARD[_apply_forward_update]:::converter

    CLEAN --> TRUNCATE
    CLEAN --> DELETE[delete_dataset_root / delete_frequency_directory]:::cleanup
    CLEAN --> REMOVE[remove_variables_from_frequency_stores]:::cleanup

    MERGE --> MERGE_STORES[merge_zarr_stores]:::zarr_publisher
    REFLOW --> WORKFLOW[cli.reflow_workflow.main]:::reflow_cli

    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef cleanup fill:#fee2e2,stroke:#dc2626,color:#000000
    classDef mapper fill:#dbeafe,stroke:#2563eb,color:#000000
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
    classDef reflow_cli fill:#fce7f3,stroke:#db2777,color:#000000
```

`run_reflow` is not another mapper implementation.  It delegates to the
workflow module, whose workers ultimately call the same `map_grib_to_healpix`
function used by normal remapping.

## Request and source resolution

This is the common input-preparation path.  `fetch`, `remap`, `update`, and
reflow planning all use it to turn selected variables and source files into
resolved records.

```mermaid
flowchart LR
    REQUESTS[selected_requests]:::converter --> LOAD_REQ[load_variable_requests]:::file_fetcher
    REQUESTS --> SELECT[selected_variables]:::file_fetcher
    LOAD_REQ --> CSV[split_csv_list]:::file_fetcher

    RESOLVE[resolve_records]:::file_fetcher --> SELECT
    RESOLVE --> CMOR[load_cmor_variable_entries]:::file_fetcher
    CMOR --> JSON[load_json]:::file_fetcher
    RESOLVE --> ENTRY[find_variable_entry]:::file_fetcher
    RESOLVE --> LEVEL[parse_level_type]:::file_fetcher
    RESOLVE --> PRIORITY[resolve_priority_files]:::file_fetcher
    PRIORITY --> PATTERN[source_pattern_template]:::file_fetcher
    PRIORITY --> INTERVAL[file_interval]:::file_fetcher
    RESOLVE --> FACTOR[parse_conversion_factor]:::file_fetcher
    RESOLVE --> ATTRS[extract_output_attrs]:::file_fetcher

    FETCH[run_fetch]:::converter --> RESOLVE
    REMAP[run_remap]:::converter --> RESOLVE
    UPDATE_RECORDS[_resolve_update_records]:::converter --> RESOLVE
    GATHER[gather_plan]:::reflow_cli --> RESOLVE

    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef reflow_cli fill:#fce7f3,stroke:#db2777,color:#000000
```

## Shared remapping spine

This is only the transformation portion after a command has records to map.
It is shared by ordinary remap, the remapping part of update, and reflow
workers; it does not show update-specific date selection or cleanup.

```mermaid
flowchart LR
    MAP[converter.map_records]:::converter --> HEALPIX[mapper.map_grib_to_healpix]:::mapper
    HEALPIX --> GROUP[group_records_by_frequency]:::formatter
    HEALPIX --> MERGE_DATA[merge_frequency_dataset]:::datasets
    MERGE_DATA --> OPEN[open_record_dataset]:::datasets
    OPEN --> NORMALISE_TIME[normalise_time_for_frequency]:::datasets
    OPEN --> VALIDATE[validate_one_value_per_day]:::datasets
    MERGE_DATA --> ALIGN[_align_datasets_on_shared_time]:::datasets

    HEALPIX --> GAUSSIAN[normalise_reduced_gaussian_dataset]:::datasets
    GAUSSIAN --> GEOMETRY[_reduced_gaussian_geometry]:::datasets
    HEALPIX --> GRID[grid_doctor regrid/coarsen]:::grid_doctor
    HEALPIX --> ZOOM[_write_zoom_level]:::mapper
    ZOOM --> DEST[destination_for_level]:::formatter
    ZOOM --> PUBLISH[update_zarr_store]:::zarr_publisher

    HEALPIX --> SPECIAL[_write_special_frequency]:::mapper
    SPECIAL --> WRITE_SPECIAL[write_special_variables]:::special
    WRITE_SPECIAL --> SPECIAL_DATA[build_special_variable_dataset]:::special
    WRITE_SPECIAL --> PUBLISH

    REMAP[run_remap]:::converter --> MAP
    PERMANENT[_apply_permanent_update]:::converter --> UPDATE_MAP[_map_update_records]:::converter
    FORWARD[_apply_forward_update]:::converter --> UPDATE_MAP
    UPDATE_MAP --> MAP
    WORKER[remap_variable_frequency]:::reflow_cli --> HEALPIX

    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef datasets fill:#ede9fe,stroke:#7c3aed,color:#000000
    classDef grid_doctor fill:#cffafe,stroke:#0891b2,color:#000000
    classDef mapper fill:#dbeafe,stroke:#2563eb,color:#000000,stroke-width:2px
    classDef special fill:#ffedd5,stroke:#ea580c,color:#000000
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
    classDef reflow_cli fill:#fce7f3,stroke:#db2777,color:#000000


```

`map_grib_to_healpix` is the central transformation boundary.  It combines
resolved GRIB records, normalises their geometry and time axis, regrids or
coarsens them, then publishes each zoom level.  Special (`fx`) variables take
the `write_special_variables` branch instead of the ordinary regridding path.

## Update, cleanup, and merge paths

This view shows the work that is *not* repeated in the shared remapping spine.
For `run_update`, `_existing_variable_last_date` determines the already
published boundary and `_select_permanent_records` chooses the appropriate
source records.  Both update branches then enter `_map_update_records`, which
continues through `map_records` in the **Shared remapping spine** above.
`run_clean` is independent destructive-store maintenance, while `run_merge`
combines worker Zarr stores; they share this diagram only because both are
short storage-management paths.

```mermaid
flowchart LR
    UPDATE[run_update]:::converter --> LAST[_existing_variable_last_date]:::converter
    LAST --> EXISTING[existing_destinations_for_frequency]:::formatter
    UPDATE --> PERM[_select_permanent_records]:::converter
    UPDATE --> APPLY_PERM[_apply_permanent_update]:::converter
    UPDATE --> APPLY_FORWARD[_apply_forward_update]:::converter
    APPLY_PERM --> UPDATE_MAP[_map_update_records]:::converter
    APPLY_FORWARD --> UPDATE_MAP
    APPLY_PERM --> SYNC[sync_named_variable_attrs]:::zarr_publisher

    CLEAN[run_clean]:::converter --> TRUNC[truncate_existing_healpix_stores]:::cleanup
    TRUNC --> TRUNC_FREQ[truncate_frequency_destinations]:::cleanup
    TRUNC_FREQ --> SHRINK[truncate_zarr_store_after]:::cleanup
    SHRINK --> SHRINK_ARRAYS[_shrink_time_arrays_in_place]:::cleanup
    CLEAN --> REMOVE[remove_variables_from_frequency_stores]:::cleanup
    REMOVE --> DROP[drop_variables_from_zarr_store]:::cleanup
    CLEAN --> DELETE_LEVEL[delete_frequency_level_stores]:::cleanup
    CLEAN --> DELETE_FREQ[delete_frequency_directory]:::cleanup

    MERGE[run_merge]:::converter --> MERGE_STORES[merge_zarr_stores]:::zarr_publisher
    MERGE_STORES --> PUBLISH[update_zarr_store]:::zarr_publisher

    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef cleanup fill:#fee2e2,stroke:#dc2626,color:#000000
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
```

## Reflow workflow

Reflow is the parallel orchestration alternative to direct remap: it first
creates a work plan, maps each work item in a worker, then merges worker output
and writes special variables during finalisation.

```mermaid
flowchart TD
    REFLOW[run_reflow]:::converter --> MAIN[reflow_workflow.main]:::reflow_cli
    MAIN --> PLAN[gather_plan]:::reflow_cli
    PLAN --> ITEMS[_batched_work_items]:::reflow_cli
    ITEMS --> FILE_BATCHES[batched_source_record_files]:::file_fetcher
    ITEMS --> TIME_BATCHES[batched_intervals]:::formatter
    ITEMS --> PRESSURE[_pressure_level_groups_for_record]:::reflow_cli
    MAIN --> WORK[remap_variable_frequency]:::reflow_cli
    WORK --> HEALPIX[map_grib_to_healpix]:::mapper
    MAIN --> FINAL[finalize_outputs]:::reflow_cli
    FINAL --> MERGE[merge_zarr_stores]:::zarr_publisher
    FINAL --> SPECIAL[map_grib_to_healpix for special variables]:::mapper

    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef mapper fill:#dbeafe,stroke:#2563eb,color:#000000,stroke-width:2px
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
    classDef reflow_cli fill:#fce7f3,stroke:#db2777,color:#000000,stroke-width:2px
```

## Detailed command graphs

The remaining diagrams expand each command individually.  They repeat shared
functions by design, so a reader can follow one command without jumping back
to another diagram.  The same color legend applies to every diagram.

## Fetch

```mermaid
flowchart LR
    fetch[fetch] --> run_fetch
    run_fetch --> parse_cli_args
    run_fetch --> parse_cli_freqs
    parse_cli_freqs --> normalise_frequencies
    parse_cli_freqs --> split_csv_list
    run_fetch --> selected_requests
    selected_requests --> load_json
    selected_requests --> load_variable_requests
    selected_requests --> selected_variables
    run_fetch --> split_special_variables
    run_fetch --> extend_frequencies_for_special_variables
    extend_frequencies_for_special_variables --> split_special_variables
    run_fetch --> resolve_records
    resolve_records --> selected_variables
    resolve_records --> load_variable_requests
    load_variable_requests --> split_csv_list
    resolve_records --> load_cmor_variable_entries
    load_cmor_variable_entries --> load_json
    resolve_records --> find_variable_entry
    resolve_records --> parse_level_type
    resolve_records --> resolve_priority_files
    resolve_priority_files --> source_pattern_template
    resolve_priority_files --> overlaps_interval
    overlaps_interval --> file_interval
    resolve_priority_files --> file_interval
    resolve_records --> parse_conversion_factor
    parse_conversion_factor --> _safe_eval_numeric_expression
    resolve_records --> extract_output_attrs
    run_fetch --> unresolved_records

    class fetch,run_fetch,selected_requests,extend_frequencies_for_special_variables,unresolved_records converter
    class parse_cli_args,parse_cli_freqs supporting
    class normalise_frequencies formatter
    class split_csv_list,load_json,load_variable_requests,selected_variables,resolve_records,load_cmor_variable_entries,find_variable_entry,parse_level_type,resolve_priority_files,source_pattern_template,overlaps_interval,file_interval,parse_conversion_factor,_safe_eval_numeric_expression,extract_output_attrs file_fetcher
    class split_special_variables special
    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef special fill:#ffedd5,stroke:#ea580c,color:#000000
    classDef supporting fill:#cbd5e1,stroke:#475569,color:#000000
```

## Remap

```mermaid
flowchart LR
    remap[remap] --> run_remap
    run_remap --> parse_cli_args
    run_remap --> parse_cli_freqs
    run_remap --> parse_interval
    run_remap --> parse_truncate_after
    run_remap --> parse_coarsen_levels
    run_remap --> selected_requests
    run_remap --> split_special_variables
    run_remap --> extend_frequencies_for_special_variables
    run_remap --> validate_remap_args
    run_remap --> batch_file_child_state
    run_remap --> dataset_output_root
    run_remap --> truncate_existing_healpix_stores
    truncate_existing_healpix_stores --> truncate_frequency_destinations
    truncate_existing_healpix_stores --> existing_level_destinations
    existing_level_destinations --> existing_destinations_for_frequency
    truncate_existing_healpix_stores --> truncate_zarr_store_after
    truncate_zarr_store_after --> _shrink_time_arrays_in_place
    _shrink_time_arrays_in_place --> _time_axis_info
    run_remap --> rechunk_existing_healpix_stores
    rechunk_existing_healpix_stores --> mapper_existing_level_destinations[mapper._existing_level_destinations]
    mapper_existing_level_destinations --> existing_destinations_for_frequency
    rechunk_existing_healpix_stores --> rechunk_zarr_store
    rechunk_zarr_store --> _rewrite_dataset_via_temp
    _rewrite_dataset_via_temp --> _write_dataset
    _write_dataset --> normalise_published_dataset
    _write_dataset --> _encoding_for_target_chunks
    run_remap --> run_single_interval
    run_single_interval --> resolve_records
    run_single_interval --> update_healpix_attrs_only
    update_healpix_attrs_only --> split_special_variables
    update_healpix_attrs_only --> global_attrs_for_records
    update_healpix_attrs_only --> global_attrs_for_dataset_frequency
    update_healpix_attrs_only --> attrs_for_record
    update_healpix_attrs_only --> special_variable_attrs_by_name
    special_variable_attrs_by_name --> build_special_variable_dataset
    build_special_variable_dataset --> _special_variable_attrs
    build_special_variable_dataset --> _areacella_data_array
    build_special_variable_dataset --> _attach_healpix_metadata
    update_healpix_attrs_only --> existing_destinations_for_frequency
    update_healpix_attrs_only --> sync_global_attrs
    update_healpix_attrs_only --> sync_named_variable_attrs
    run_single_interval --> build_file_batch_plan
    build_file_batch_plan --> batched_source_record_files
    batched_source_record_files --> files_interval
    files_interval --> file_interval
    run_single_interval --> run_batched_files
    run_batched_files --> _run_subprocess
    _run_subprocess --> build_batch_command
    _run_subprocess --> write_batch_state
    _run_subprocess --> clear_batch_state
    run_single_interval --> map_records
    map_records --> map_grib_to_healpix
    map_grib_to_healpix --> split_special_variables
    map_grib_to_healpix --> group_records_by_frequency
    map_grib_to_healpix --> _missing_frequency_variables
    map_grib_to_healpix --> merge_frequency_dataset
    merge_frequency_dataset --> open_record_dataset
    open_record_dataset --> open_dataset
    open_record_dataset --> validate_one_value_per_day
    open_record_dataset --> normalise_time_for_frequency
    merge_frequency_dataset --> select_time_interval
    merge_frequency_dataset --> get_vars
    merge_frequency_dataset --> clean_output_attrs
    merge_frequency_dataset --> _align_datasets_on_shared_time
    _align_datasets_on_shared_time --> _record_time_alignment_mismatch
    _record_time_alignment_mismatch --> _format_timestamp_dates
    map_grib_to_healpix --> normalise_reduced_gaussian_dataset
    normalise_reduced_gaussian_dataset --> _find_coord_name
    normalise_reduced_gaussian_dataset --> _reduced_gaussian_geometry
    _reduced_gaussian_geometry --> _load_cached_reduced_gaussian_geometry
    _reduced_gaussian_geometry --> _compute_reduced_gaussian_geometry
    _reduced_gaussian_geometry --> _store_reduced_gaussian_geometry
    map_grib_to_healpix --> gd_get_latlon_resolution[gd.get_latlon_resolution]
    map_grib_to_healpix --> gd_cached_weights[gd.cached_weights]
    map_grib_to_healpix --> gd_regrid_to_healpix[gd.regrid_to_healpix]
    map_grib_to_healpix --> gd_coarsen_healpix[gd.coarsen_healpix]
    map_grib_to_healpix --> _write_zoom_level
    _write_zoom_level --> destination_for_level
    _write_zoom_level --> _ensure_output_directory
    _write_zoom_level --> update_zarr_store
    map_grib_to_healpix --> _write_special_frequency
    _write_special_frequency --> _special_zoom_numbers_for_frequency
    _special_zoom_numbers_for_frequency --> _existing_zoom_numbers
    _special_zoom_numbers_for_frequency --> _fallback_special_zoom_numbers
    _write_special_frequency --> write_special_variables
    write_special_variables --> build_special_variable_dataset
    write_special_variables --> update_zarr_store
    map_grib_to_healpix --> _resolve_requested_coarsen_levels
    _resolve_requested_coarsen_levels --> _coarsen_existing_frequency
    _coarsen_existing_frequency --> gd_coarsen_healpix
    _coarsen_existing_frequency --> _write_zoom_level

    class remap,run_remap,selected_requests,extend_frequencies_for_special_variables,validate_remap_args,batch_file_child_state,run_single_interval,build_file_batch_plan,run_batched_files,_run_subprocess,build_batch_command,write_batch_state,clear_batch_state,map_records converter
    class parse_cli_args,parse_cli_freqs,parse_interval,parse_truncate_after,parse_coarsen_levels supporting
    class dataset_output_root,existing_level_destinations,existing_destinations_for_frequency,destination_for_level formatter
    class truncate_existing_healpix_stores,truncate_frequency_destinations,truncate_zarr_store_after,_shrink_time_arrays_in_place,_time_axis_info cleanup
    class rechunk_existing_healpix_stores,mapper_existing_level_destinations,update_healpix_attrs_only,_missing_frequency_variables,_write_zoom_level,_ensure_output_directory,_write_special_frequency,_special_zoom_numbers_for_frequency,_existing_zoom_numbers,_fallback_special_zoom_numbers,_resolve_requested_coarsen_levels,_coarsen_existing_frequency mapper
    class rechunk_zarr_store,_rewrite_dataset_via_temp,_write_dataset,_encoding_for_target_chunks,update_zarr_store,sync_global_attrs,sync_named_variable_attrs zarr_publisher
    class normalise_published_dataset,merge_frequency_dataset,open_record_dataset,validate_one_value_per_day,normalise_time_for_frequency,select_time_interval,clean_output_attrs,_align_datasets_on_shared_time,_record_time_alignment_mismatch,_format_timestamp_dates,normalise_reduced_gaussian_dataset,_find_coord_name,_reduced_gaussian_geometry,_load_cached_reduced_gaussian_geometry,_compute_reduced_gaussian_geometry,_store_reduced_gaussian_geometry datasets
    class resolve_records,batched_source_record_files,files_interval,file_interval file_fetcher
    class split_special_variables,special_variable_attrs_by_name,build_special_variable_dataset,_special_variable_attrs,_areacella_data_array,_attach_healpix_metadata,write_special_variables special
    class global_attrs_for_records,global_attrs_for_dataset_frequency,attrs_for_record,get_vars metadata
    class group_records_by_frequency formatter
    class gd_get_latlon_resolution,gd_cached_weights,gd_regrid_to_healpix,gd_coarsen_healpix external
    class open_dataset external
    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef cleanup fill:#fee2e2,stroke:#dc2626,color:#000000
    classDef mapper fill:#dbeafe,stroke:#2563eb,color:#000000,stroke-width:2px
    classDef datasets fill:#ede9fe,stroke:#7c3aed,color:#000000
    classDef special fill:#ffedd5,stroke:#ea580c,color:#000000
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef metadata fill:#cbd5e1,stroke:#475569,color:#000000
    classDef external fill:#cffafe,stroke:#0891b2,color:#000000
    classDef supporting fill:#cbd5e1,stroke:#475569,color:#000000
```

## Update

```mermaid
flowchart LR
    update[update] --> run_update
    run_update --> parse_cli_args
    run_update --> parse_cli_freqs
    run_update --> selected_requests
    run_update --> _existing_variable_last_date
    _existing_variable_last_date --> existing_destinations_for_frequency
    run_update --> _update_remap_args
    run_update --> add_months
    run_update --> _resolve_update_records
    _resolve_update_records --> resolve_records
    run_update --> _select_permanent_records
    _select_permanent_records --> file_interval
    _select_permanent_records --> _is_final_source_file
    _is_final_source_file --> file_interval
    _select_permanent_records --> add_months
    run_update --> _apply_permanent_update
    _apply_permanent_update --> _map_update_records
    _map_update_records --> batched_intervals
    _map_update_records --> batched_source_record_files
    _map_update_records --> overlaps_interval
    _map_update_records --> file_interval
    _map_update_records --> map_records
    map_records --> map_grib_to_healpix
    _apply_permanent_update --> existing_destinations_for_frequency
    _apply_permanent_update --> sync_named_variable_attrs
    run_update --> _apply_forward_update
    _apply_forward_update --> _map_update_records
    run_update --> _preview_update_row
    run_update --> _log_update_preview

    class update,run_update,selected_requests,_existing_variable_last_date,_update_remap_args,add_months,_resolve_update_records,_select_permanent_records,_is_final_source_file,_apply_permanent_update,_map_update_records,_apply_forward_update,_preview_update_row,_log_update_preview,map_records converter
    class parse_cli_args,parse_cli_freqs supporting
    class existing_destinations_for_frequency formatter
    class resolve_records,batched_source_record_files,overlaps_interval,file_interval file_fetcher
    class batched_intervals formatter
    class sync_named_variable_attrs zarr_publisher
    class map_grib_to_healpix mapper
    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef mapper fill:#dbeafe,stroke:#2563eb,color:#000000,stroke-width:2px
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
    classDef supporting fill:#cbd5e1,stroke:#475569,color:#000000
```

## Clean

```mermaid
flowchart LR
    clean[clean] --> run_clean
    run_clean --> parse_cli_args
    run_clean --> parse_level_selection
    parse_level_selection --> parse_coarsen_levels
    run_clean --> parse_cli_freqs
    run_clean --> parse_truncate_after
    run_clean --> truncate_existing_healpix_stores
    run_clean --> delete_dataset_root
    run_clean --> delete_frequency_directory
    delete_frequency_directory --> existing_level_destinations
    existing_level_destinations --> existing_destinations_for_frequency
    run_clean --> delete_frequency_level_stores
    delete_frequency_level_stores --> selected_level_destinations
    selected_level_destinations --> existing_level_destinations
    run_clean --> remove_variables_from_frequency_stores
    remove_variables_from_frequency_stores --> selected_level_destinations
    remove_variables_from_frequency_stores --> drop_variables_from_zarr_store

    class clean,run_clean,parse_level_selection converter
    class parse_cli_args,parse_cli_freqs,parse_truncate_after,parse_coarsen_levels supporting
    class truncate_existing_healpix_stores,delete_dataset_root,delete_frequency_directory,delete_frequency_level_stores,selected_level_destinations,remove_variables_from_frequency_stores,drop_variables_from_zarr_store cleanup
    class existing_level_destinations,existing_destinations_for_frequency formatter
    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef cleanup fill:#fee2e2,stroke:#dc2626,color:#000000
    classDef supporting fill:#cbd5e1,stroke:#475569,color:#000000
```

## Merge

```mermaid
flowchart LR
    merge[merge] --> run_merge
    run_merge --> parse_cli_args
    run_merge --> expand_source_dirs
    expand_source_dirs --> parse_cli_args
    run_merge --> parse_level_selection
    parse_level_selection --> parse_coarsen_levels
    run_merge --> parse_interval
    run_merge --> merge_dataset_root    
    run_merge --> merge_zarr_stores
    merge_zarr_stores --> _frequency_names
    merge_zarr_stores --> _variable_names
    merge_zarr_stores --> merge_dataset_root    
    merge_zarr_stores --> _worker_output_roots
    merge_zarr_stores --> _dataset_root_destinations
    merge_zarr_stores --> _is_selected_level_store
    merge_zarr_stores --> _merge_source_stores
    merge_zarr_stores --> _select_merge_interval
    merge_zarr_stores --> update_zarr_store

    class merge,run_merge,expand_source_dirs,parse_level_selection converter
    class parse_cli_args,parse_coarsen_levels,parse_interval supporting
    class merge_zarr_stores,_frequency_names,_variable_names,_worker_output_roots,_dataset_root_destinations,_is_selected_level_store,_merge_source_stores,_select_merge_interval,update_zarr_store zarr_publisher
    class merge_dataset_root formatter
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
    classDef supporting fill:#cbd5e1,stroke:#475569,color:#000000
```

## Reflow

```mermaid
flowchart LR
    reflow[reflow] --> run_reflow
    run_reflow --> reflow_main[cli.reflow_workflow.main]
    reflow_main --> gather_plan
    gather_plan --> parse_cli_args
    gather_plan --> parse_cli_freqs
    gather_plan --> parse_interval
    gather_plan --> selected_requests
    gather_plan --> split_special_variables
    gather_plan --> extend_frequencies_for_special_variables
    gather_plan --> resolve_records
    gather_plan --> _batched_work_items
    _batched_work_items --> _batching_level_type
    _batched_work_items --> _batch_settings_for_item
    _batch_settings_for_item --> _level_policy
    _batched_work_items --> batched_source_record_files
    _batched_work_items --> batched_intervals
    _batched_work_items --> _pressure_level_groups_for_record
    _pressure_level_groups_for_record --> _record_pressure_levels
    _record_pressure_levels --> cached_grib_inventory
    cached_grib_inventory --> _chunk_pressure_levels
    reflow_main --> remap_variable_frequency
    remap_variable_frequency --> _load_record_cache
    _load_record_cache --> _record_from_payload
    remap_variable_frequency --> _worker_output_root
    remap_variable_frequency --> map_grib_to_healpix
    reflow_main --> finalize_outputs
    finalize_outputs --> dataset_output_root
    finalize_outputs --> destination_for_level
    finalize_outputs --> merge_zarr_stores
    finalize_outputs --> split_special_variables
    finalize_outputs --> special_fx_map[map_grib_to_healpix for special fx variables]

    class reflow,run_reflow converter
    class reflow_main,gather_plan,_batched_work_items,_batching_level_type,_batch_settings_for_item,_level_policy,_pressure_level_groups_for_record,_record_pressure_levels,remap_variable_frequency,_load_record_cache,_record_from_payload,_worker_output_root,finalize_outputs reflow_cli
    class selected_requests,extend_frequencies_for_special_variables converter
    class parse_cli_args,parse_cli_freqs,parse_interval supporting
    class batched_source_record_files,resolve_records file_fetcher
    class batched_intervals,dataset_output_root,destination_for_level formatter
    class split_special_variables special
    class cached_grib_inventory,_chunk_pressure_levels supporting
    class map_grib_to_healpix,special_fx_map mapper
    class merge_zarr_stores zarr_publisher
    classDef converter fill:#e5e7eb,stroke:#4b5563,color:#000000,stroke-width:2px
    classDef reflow_cli fill:#fce7f3,stroke:#db2777,color:#000000,stroke-width:2px
    classDef file_fetcher fill:#fef3c7,stroke:#d97706,color:#000000
    classDef formatter fill:#fde68a,stroke:#ca8a04,color:#000000
    classDef special fill:#ffedd5,stroke:#ea580c,color:#000000
    classDef supporting fill:#cbd5e1,stroke:#475569,color:#000000
    classDef mapper fill:#dbeafe,stroke:#2563eb,color:#000000,stroke-width:2px
    classDef zarr_publisher fill:#dcfce7,stroke:#16a34a,color:#000000
```

## Redundancy assessment

| Functions | Assessment | Recommended direction |
| --- | --- | --- |
| `cleanup.existing_level_destinations` and `mapper._existing_level_destinations` | Substantively duplicate directory discovery.  Their meaningful difference is return type (`Path` versus `str`). | Keep one shared function, ideally returning `Path`; convert to `str` only at the external API boundary if necessary. |
| `converter.parse_level_selection` and `parse_coarsen_levels` | A pure naming alias. | Remove the alias or rename callers to `parse_coarsen_levels` if command-specific terminology is not valuable. |
| `converter._resolve_update_records` and `resolve_records` | A thin update-specific adapter around the general resolver. | Keep only if it makes update defaults explicit; otherwise inline its small argument adaptation. |
| `converter._update_remap_args` | A thin `argparse.Namespace` adapter for reusing remap logic during update. | Keep as an explicit compatibility boundary; replacing it with an untyped dictionary would be worse. |
| `zarr_publisher.sync_global_attrs` and `_sync_global_attrs` | Public wrapper around private implementation, not duplicated behavior. | Keep; it defines the module's supported API. |
| `_apply_permanent_update` and `_apply_forward_update` | They share `_map_update_records`, but permanent updates also record permanent-update metadata. | Keep separate.  The common mapping helper is already the right extraction. |
| Cleanup deletion helpers | Similar traversal but distinct scopes: variable, selected level, frequency, or dataset root. | Keep separate; consolidating them would blur destructive-operation scope. |
| `run_remap` and reflow workers | Both intentionally converge on `map_grib_to_healpix`. | Keep shared; reflow is orchestration/batching, not a competing mapping implementation. |

## Source map

The most useful starting points are:

- [`converter.py`](converter.py): command dispatch and orchestration.
- [`helpers/mapper.py`](helpers/mapper.py): the main GRIB-to-HEALPix transformation.
- [`helpers/datasets.py`](helpers/datasets.py): opening, normalising, and merging source datasets.
- [`helpers/zarr_publisher.py`](helpers/zarr_publisher.py): append/rewrite publication semantics.
- [`helpers/cleanup.py`](helpers/cleanup.py): truncation and deletion operations.
- [`cli/reflow_workflow.py`](cli/reflow_workflow.py): parallel plan, worker, and finalisation flow.
