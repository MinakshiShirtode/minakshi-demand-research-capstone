.PHONY: env test notebooks report all
env:        ## create venv from the lockfile
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.lock
test:       ## data-blind synthetic pipeline check
	python3 -m pytest tests -q
notebooks:  ## build + execute the 4 notebooks on the real data
	python3 scripts/build_notebooks.py
report:     ## build the PDF whitepaper
	python3 scripts/build_report.py
all: test notebooks report
