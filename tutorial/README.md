# Neuro-Symbolic standalone tutorial

This is an installation and public-API smoke test, using the same structure
as the standalone Neural tutorial. It tests the installed shared library,
validated interval enclosure, and symbolic expansion (C++), plus installed
`pyariadne` bindings for validated inference.

After installing:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
./build/tutorial_neurosymbolic
python tutorial_neurosymbolic.py
```
