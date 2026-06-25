# --- Repository & Table Configuration ---
ERA5_TABLES_REPO = https://gitlab.dkrz.de/bm1159/cosodax/era5-tables.git
ERA5_TABLES_SRC  = Tables/era5-cmor-tables
ERA5_TABLES_DST  = tables/era5-cmor-tables

# --- Conda Environment Configuration ---
ENV_PREFIX        := .conda-env
ENV_PREFIX_ABS    := $(abspath $(ENV_PREFIX))
ENV_FILE          := requirements.yml
PLATFORM          := $(shell uname)
ARCH              := $(shell uname -m)
MINIFORGE_INSTALLER := Miniforge3-$(PLATFORM)-$(ARCH).sh
MINIFORGE_URL     := https://github.com/conda-forge/miniforge/releases/latest/download/$(MINIFORGE_INSTALLER)

# --- Phony Targets ---
.PHONY: all env era5-tables update-era5-tables update5-era5-tables clean clean-era5-tables

all: env era5-tables

# ==========================================
# 1. Environment Build & Orchestration
# ==========================================
env: 
	curl -L -o "$(MINIFORGE_INSTALLER)" "$(MINIFORGE_URL)"
	if [ -x "$(ENV_PREFIX)/bin/mamba" ]; then \
		echo "Using existing Miniforge in $(ENV_PREFIX)"; \
	else \
		bash "$(MINIFORGE_INSTALLER)" -b -f -p "$(ENV_PREFIX)"; \
	fi
	"$(ENV_PREFIX)/bin/mamba" env update -y -p "$(ENV_PREFIX_ABS)" -f "$(ENV_FILE)"
# 	"$(ENV_PREFIX)/bin/pip" install --no-deps -e .
	rm -f "$(MINIFORGE_INSTALLER)"


# ==========================================
# 2. CMOR Data Tables Sync
# ==========================================
era5-tables: update-era5-tables

update5-era5-tables: update-era5-tables

update-era5-tables:
	@echo "Updating tables from $(ERA5_TABLES_REPO)..."
	tmpdir=$$(mktemp -d); \
	trap 'rm -rf "$$tmpdir"' EXIT; \
	git clone --depth 1 $(ERA5_TABLES_REPO) $$tmpdir; \
	mkdir -p $$(dirname $(ERA5_TABLES_DST)); \
	rm -rf $(ERA5_TABLES_DST).tmp; \
	cp -a $$tmpdir/$(ERA5_TABLES_SRC) $(ERA5_TABLES_DST).tmp; \
	rm -rf $(ERA5_TABLES_DST); \
	mv $(ERA5_TABLES_DST).tmp $(ERA5_TABLES_DST)
	@echo "Tables updated in $(ERA5_TABLES_DST)/"

# ==========================================
# 3. Cleanup Rules
# ==========================================
clean: clean-era5-tables
	rm -rf $(ENV_PREFIX) link_unifier.egg-info

clean-era5-tables:
	rm -rf $(ERA5_TABLES_DST)
