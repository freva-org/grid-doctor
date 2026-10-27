from mkdocs.structure.files import File

PREFIX = ("shared/",)


def on_files(files, config):
    for _prefix in PREFIX:
        for f in list(files):
            if f.src_uri.startswith(_prefix):
                files.remove(f)
                files.append(
                    File.generated(
                        config,
                        f.src_uri.removeprefix(_prefix),  # <_prefix>/foo.md -> foo.md
                        abs_src_path=f.abs_src_path,
                    )
                )
    return files
