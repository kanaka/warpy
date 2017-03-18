# warpy - WebAssembly in RPython

A WebAssembly interpreter written in RPython.

## Usage

Currently support WebAssembly version 0xc (version 12). This is the
version currently support by node.js (7.x). You will also need a build
of rpython on the path to be able to build the warpy executable.

You will need to compile your wast sources using an older version of
`wast2wasm` or `wasm-opt`

```
wast2wasm foo.wast -o foo.wasm
make warpy
./warpy foo.wasm myfunc 10
```

You can also run warpy using standard python (but it's much slower of
course):

```
python warpy.py foo.wasm myfunc 10
```

## License

MPL-2.0
