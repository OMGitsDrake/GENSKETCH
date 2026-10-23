PYTHON ?= py

.PHONY: help MLP_main

help:
	@echo "MLP_main		Avvia addestramento e test del MLP"

MLP_main:
	$(PYTHON) -m src.classification.MLP_main