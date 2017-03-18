RPYTHON = rpython

RPYTHON_OPTS = --opt=jit

PYPY = pypy2-v5.6.0-src

SOURCES = wasty.py

all: wasty

$(PYPY):
	curl -O -L curl -O -L https://bitbucket.org/pypy/pypy/downloads/$@
	unzip $(PYPY).zip

#%: %.py $(PYPY)

%: %.py
	$(RPYTHON) $(RPYTHON_OPTS) --output=$@ $<

.PHONY: clean stats stats-lisp

clean:
	rm -f *.pyc wasty
	rm -rf __pycache__

stats: $(SOURCES)
	@wc $^
	@printf "%5s %5s %5s %s\n" `grep -E "^[[:space:]]*#|^[[:space:]]*$$" $^ | wc` "[comments/blanks]"
