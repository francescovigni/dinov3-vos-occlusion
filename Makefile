VENV := .venv/bin
export PYTHONPATH := src
CFG ?= configs/default.yaml

.PHONY: setup test lint data extract baseline train eval ablations figures

setup:
	uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt

test:
	$(VENV)/python -m pytest

lint:
	$(VENV)/ruff check src tests && $(VENV)/ruff format --check src tests

data:
	./scripts/download_davis.sh

extract:
	$(VENV)/python -m dvos.extract --config $(CFG) --split val
	$(VENV)/python -m dvos.extract --config $(CFG) --split train

baseline:
	for v in clean occ0 occ1; do $(VENV)/python -m dvos.evaluate --config $(CFG) --method baseline --variant $$v --out runs/baseline_$$v --save-masks; done

train:
	$(VENV)/python -m dvos.train --config $(CFG) --out runs/head

eval:
	for v in clean occ0 occ1; do $(VENV)/python -m dvos.evaluate --config $(CFG) --method model --checkpoint runs/head/model.pt --variant $$v --out runs/head_$$v --save-masks; done

# eval-time ablations on the trained head, plus a head trained on clean features only
ablations:
	for v in occ0 occ1; do $(VENV)/python -m dvos.evaluate --config $(CFG) --method model --checkpoint runs/head/model.pt --variant $$v --out runs/abl_ungated_$$v --vis-gate 0; done
	for v in occ0 occ1; do $(VENV)/python -m dvos.evaluate --config $(CFG) --method model --checkpoint runs/head/model.pt --variant $$v --out runs/abl_fifo_$$v --no-keep-first; done
	$(VENV)/python -m dvos.train --config $(CFG) --out runs/head_cleanonly --variants clean
	for v in clean occ0 occ1; do $(VENV)/python -m dvos.evaluate --config $(CFG) --method model --checkpoint runs/head_cleanonly/model.pt --variant $$v --out runs/abl_cleanonly_$$v; done

figures:
	$(VENV)/python -m dvos.figures --config $(CFG) --run runs/head_occ0 --out docs/figures/qualitative_occ0.png
	$(VENV)/python -m dvos.figures --config $(CFG) --run runs/baseline_occ0 --out docs/figures/qualitative_baseline_occ0.png
