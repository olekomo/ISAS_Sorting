.PHONY: test smoke benchmark

test:
	pytest

smoke:
	python examples/smoke_env.py

benchmark:
	python examples/benchmark.py
