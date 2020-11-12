SOURCES := $(shell find . -type f -and \( -name '*.md' -or -name '*.tex' -or -name '*.bib' \))

default: build

.PHONY: build
build: tutorial.pdf

.PHONY: watch
watch:
	@fswatch -o $(SOURCES) | xargs -n1 -I{} make

.PHONY: clean
clean:
	@latexmk -C tutorial

.PHONY: view
view:
	@open -a Skim tutorial.pdf

tutorial.pdf: $(SOURCES)
	@pandoc --template template.tex -o tutorial.pdf tutorial.md

