PYTHON ?= py

.PHONY: help classification_main

help:
	@echo "classification_main		Avvia addestramento e test del classificatore"

classification_main:
	$(PYTHON) -m src.classification.classification_main