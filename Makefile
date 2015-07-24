.PHONY: clean

include config.make


all:
	python setup.py build_ext \
	    --include-dirs=$(INCLUDE_DIRS)

install:
	python setup.py install \
	    --prefix=$(PREFIX)

test:
	./unit_test/lvq.py

clean:
	rm -rf build *.o
