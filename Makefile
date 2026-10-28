PYTHON ?= python3

.PHONY: help install test predict eval train demo all quick clean

help:
	@echo "MV-MGB Standalone Repository Targets:"
	@echo "  make install     Install requirements and package in editable mode"
	@echo "  make test        Execute verification pytest suite"
	@echo "  make demo        Run quick CLI inference on demo FASTA (./run_predict.sh --demo)"
	@echo "  make predict     Run prediction pipeline (./run_predict.sh)"
	@echo "  make eval        Evaluate model on benchmark test set (./run_evaluate.sh)"
	@echo "  make train       Train model from scratch on Dataset 1 (./run_training.sh)"
	@echo "  make quick       Run fast end-to-end pipeline (< 30s) (./run_all.sh --quick)"
	@echo "  make all         Run complete end-to-end pipeline (./run_all.sh)"
	@echo "  make clean       Remove temporary Python build and cache artifacts"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -v tests/

demo:
	./run_predict.sh --demo

predict:
	./run_predict.sh

eval:
	./run_evaluate.sh

train:
	./run_training.sh

quick:
	./run_all.sh --quick

all:
	./run_all.sh

clean:
	rm -rf __pycache__ src/__pycache__ src/views/__pycache__ src/models/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf *.egg-info build dist .pytest_cache results/
