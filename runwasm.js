#!/usr/bin/node --expose-wasm

// Copyright Joel Martin
// License MIT

const fs = require('fs'),
      assert = require('assert')

assert('WebAssembly' in global,
        'WebAssembly global object not detected')

// Convert node Buffer to Uint8Array
function toUint8Array(buf) {
  var u = new Uint8Array(buf.length)
  for (var i = 0; i < buf.length; ++i) {
    u[i] = buf[i]
  }
  return u
}

// Based on:
// https://gist.github.com/kripken/59c67556dc03bb6d57052fedef1e61ab
//   and
// http://thecodebarbarian.com/getting-started-with-webassembly-in-node.js.html

// Loads a WebAssembly dynamic library, returns a promise.
// imports is an optional imports object
function loadWebAssembly(filename, imports) {
  // Fetch the file and compile it
  const buffer = toUint8Array(fs.readFileSync(filename))
  return WebAssembly.compile(buffer)
  .then(module => {
    let memory = new WebAssembly.Memory({ initial: 256 })
    // Core imports
    function DEBUG(...nums) {
        console.log("DEBUG:", ...nums)
    }
    function writeline(addr) {
        let length = new DataView(memory.buffer, addr).getUint32()
        //console.log("writeline addr:", addr, ", length:", length)
        let bytes = new Uint8Array(memory.buffer, addr+4, length)
        let str = new TextDecoder('utf8').decode(bytes)
        console.log(str)
    }

    // Returns addr on success and -1 on failure
    // Truncates to max_length
    function readline(addr, max_length) {
        let line = node_readline.readline('user> '),
            view = new DataView(memory.buffer, addr),
            buf8 = new Uint8Array(memory.buffer, addr)
        if (line === null) {
            view.setUint32(0, 0)
            return -1
        }

        let bytes = new TextEncoder('utf8').encode(line)
        if (bytes.length > max_length) {
            bytes = bytes.slice(0, max_length)
        }

        // Strings are prefixed by uint32 (4 byte) length
        view.setUint32(0, bytes.length)
        buf8.set(bytes, 4)
        //console.log("memory(0..30):", new Uint8Array(memory.buffer, addr, 30))
        return addr
    }

    // Create the imports for the module, including the
    // standard dynamic library imports
    let core = {
            DEBUG     : DEBUG,
            writeline : writeline,
            readline  : readline
        }
    imports = {core: core}
    imports.env = imports.env || {}
    imports.env.memoryBase = imports.env.memoryBase || 0
    imports.env.tableBase = imports.env.tableBase || 0
    if (!imports.env.memory) {
      imports.env.memory = memory 
    }
    if (!imports.env.table) {
      imports.env.table = new WebAssembly.Table({ initial: 0, element: 'anyfunc' })
    }
    // Create the instance.
    return new WebAssembly.Instance(module, imports)
  })
}

if (module.parent) {
    module.exports.loadWebAssembly = loadWebAssembly
} else {
  assert(process.argv.length >= 4,
          'Usage: ./runwasm.js prog.wasm func INT_ARG...')

  const wasm = process.argv[2],
        func = process.argv[3],
        // Convert args to either floats or ints
        args = process.argv.slice(4).map(
                x => x.match(/[.]/) ? parseFloat(x) : parseInt(x))

  loadWebAssembly(wasm)
  .then(instance => {
    var exports = instance.exports
    assert(exports, 'no exports found')
    assert(func in exports, func + ' not found in wasm module exports')
    console.log('calling exports.'+func+'('+args+')')
    let res = exports[func](...args)
    if (res) {
        console.log('0x' + res.toString(16))
    }
  })
  .catch(res => {
    console.log(res)
  })
}
