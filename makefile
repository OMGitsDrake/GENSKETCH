PYTHON ?= py

.PHONY: help classification_main run_dir

help:
	@echo "classification_main		Avvia addestramento e test del classificatore"
	@echo "run_dir model=<model>		Organizza i checkpoint in una nuova directory di run"

classification_main:
	$(PYTHON) -m src.classification

run_dir:
	@test -n "$(model)" || { echo "Uso: make run_dir model=cnn"; exit 1; }
	@set -- models/$(model)_*.pt; \
	  test -e "$$1" || { echo "Nessun checkpoint per $(model)"; exit 1; }; \
	  timestamp=$$(date +%Y%m%d_%H%M%S); \
	  dir="run_$(model)_$$timestamp"; \
	  mkdir -p "models/$$dir" && mv "$$@" "models/$$dir"/