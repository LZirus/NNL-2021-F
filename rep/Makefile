FILE := main
OUT  := build

.PHONY: pdf
pdf:
	latexmk -shell-escape -interaction=nonstopmode -outdir=$(OUT) -pdf -halt-on-error $(FILE)
	cp $(OUT)/$(FILE).pdf $(FILE).pdf

.PHONY: watch
watch:
	latexmk -shell-escape -interaction=nonstopmode -outdir=$(OUT) -pdf -pvc -halt-on-error $(FILE)
	cp $(OUT)/$(FILE).pdf $(FILE).pdf

.PHONY: clean
clean:
	rm -rf $(filter-out $(OUT)/$(FILE).pdf, $(wildcard $(OUT)/*))

.PHONY: purge
purge:
	rm -rf $(OUT)
