#!/usr/bin/env python

from distutils.core import setup, Extension;


liblvq = Extension('liblvq',
    sources            = ['liblvq.cxx'],
    extra_compile_args = ['-std=c++11']);

setup(
    name         = 'lvq',
    version      = '1.0',
    description  = 'liblvq Python binding',
    author       = 'Vaclav Krpec',
    author_email = 'vencik@razdva.cz',
    ext_modules  = [liblvq]);
