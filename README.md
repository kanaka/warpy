# warpy - WebAssembly in RPython

A WebAssembly interpreter written in RPython.

Warpy supports the WebAssembly MVP (minimum viable product) version of
the spec.


## Prerequisites

Build whe warpy executable. You will need a build of rpython on the
path:

```
make warpy-jit   # or make warpy-nojit
```

Alternatively, you can build use an rypthon docker container (built
from Dockerfile.rpython) and do the compilation from there:

```
docker pull kanaka/warpy-rpython
docker run -it kanaka/warpy-rpython -v `pwd`:/build -w /build make warpy-jit
```

You will need `wast2wasm` to compile wast source to wasm bytecode.
Check-out and build [wabt](https://github.com/WebAssembly/wabt)
(wabbit):

```
git clone --recursive https://github.com/WebAssembly/wabt
make -C wabt gcc-release
```

## Usage

Compile a wasm module:

```
wast2wasm test/addTwo.wast -o test/addTwo.wasm
```

Load and call a function in a wasm module:

```
./warpy-jit test/addTwo.wasm addTwo 11 12
```

You can also use standard python (but it's much slower of course):

```
python warpy.py test/addTwo.wasm addTwo 13 14
```

There is also a REPL mode that allow you to interactively invoke
functions within a module:

```
./warpy-jit --repl test/addTwo.wasm
webassembly> addTwo 2 3
```

## License

MPL-2.0 (see LICENSE)
