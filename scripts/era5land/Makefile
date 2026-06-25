ERA5_TABLES_REPO = https://gitlab.dkrz.de/bm1159/cosodax/era5-tables.git
ERA5_TABLES_SRC  = Tables/era5-cmor-tables
ERA5_TABLES_DST  = tables/era5-cmor-tables

.PHONY: download clean

download: clean
	@echo "Downloading tables from $(ERA5_TABLES_REPO)..."
	tmpdir=$$(mktemp -d); \
	trap 'rm -rf "$$tmpdir"' EXIT; \
	git clone --depth 1 $(ERA5_TABLES_REPO) $$tmpdir; \
	mkdir -p $$(dirname $(ERA5_TABLES_DST)); \
	rm -rf $(ERA5_TABLES_DST); \
	cp -a $$tmpdir/$(ERA5_TABLES_SRC) $(ERA5_TABLES_DST)
	@echo "Tables downloaded to $(ERA5_TABLES_DST)/"

clean:
	rm -rf $(ERA5_TABLES_DST)
