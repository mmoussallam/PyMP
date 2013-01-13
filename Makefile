# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test: in
	$(NOSETESTS) PyMP --with-coverage --cover-package=PyMP

test-doc:
	$(NOSETESTS) --with-doctest --doctest-tests --doctest-extension=rst doc/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) --with-coverage --cover-package=PyMP --cover-html --cover-html-dir=coverage

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

upload-pipy:
	python setup.py sdist bdist_egg register upload
