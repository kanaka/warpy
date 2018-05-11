# warpy - WebAssembly in RPython

A WebAssembly interpreter written in RPython.

## Usage

Currently supports the WebAssembly MVP (minimum viable product)
version of the spec. You will need a build of rpython on the path to
be able to build the warpy executable.

```
wast2wasm foo.wast -o foo.wasm
make warpy-jit   # or make warpy-nojit
./warpy-jit foo.wasm myfunc 10
```

You can also run warpy using standard python (but it's much slower of
course):

```
python warpy.py foo.wasm myfunc 10
```

## License

MPL-2.0 (see LICENSE)
