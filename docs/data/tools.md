# Tools

The hub is built from a few small services. Each has its own documentation;
this page is the index.

<div class="grid cards" markdown>

-   :material-database-arrow-down:{ .lg .middle } __blobmap__

    ---

    Decides which zarr objects move to tape together, and reads that decision
    back. A store is millions of objects, which is too many to track
    individually and too few to treat as one unit: blobs are what sits in
    between and makes tiering tractable.

    [:octicons-book-16: Documentation](/tech/blobmap/){ .md-button }
    [:octicons-file-code-16: Manifest format](/blobmap/)

</div>

## Tiering, and what you will notice

Data on the hub is tiered. Everything stays **browsable** at all times, so
opening a dataset, listing its variables and reading its metadata works
whether or not the data itself is on disk.

What changes is the first read. Data nobody has touched for a long time is
moved to tape, and reading it again triggers a recall:

- the **first** read of an archived dataset takes minutes rather than seconds
- everything after it is at normal speed, until the data goes cold again
- nothing is ever deleted, and nothing needs restoring by hand

If you are working with a dataset intensively and would rather it stayed on
disk throughout, ask us to pin it. That is a deliberate decision with a name
and a reason attached, rather than something to work around.

[Contact support](mailto:waterpark@support.dkrz.de){ .md-button }

## Where things are

| | |
|---|---|
| [Data Browser](databrowser.md) | search the catalogue |
| [STAC Browser](stac-browser.md) | the same holdings, as STAC |
| [Storage Concepts](storage_concepts/index.md) | why the data is laid out this way |
| [blobmap](/tech/blobmap/) | how the tiering decision is made |
