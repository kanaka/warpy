#!/usr/bin/env python

import sys, os, math
IS_RPYTHON = sys.argv[0].endswith('rpython')

if IS_RPYTHON:
    # JIT Stuff
    from rpython.jit.codewriter.policy import JitPolicy
    def jitpolicy(driver):
        return JitPolicy()
    from rpython.rlib.jit import JitDriver, purefunction, unroll_safe

    from rpython.rtyper.lltypesystem import lltype
    from rpython.rtyper.lltypesystem.lloperation import llop
    from rpython.rlib.listsort import TimSort
    from rpython.rlib.rstruct.ieee import float_unpack
    from rpython.rlib.rfloat import round_double

    class IntSort(TimSort):
        def lt(self, a, b):
            assert isinstance(a, int)
            assert isinstance(b, int)
            return a < b

    def do_sort(a):
        IntSort(a).sort()

    @purefunction
    def unpack_f32(i32):
        return float_unpack(i32, 4)
    @purefunction
    def unpack_f64(i64):
        return float_unpack(i64, 8)

    @purefunction
    def fround(val, digits):
        return round_double(val, digits)

else:
    sys.path.append(os.path.abspath('./pypy2-v5.6.0-src'))
    import traceback
    import struct

    def purefunction(f): return f
    def unroll_safe(f): return f

    def do_sort(a):
        a.sort()

    def unpack_f32(i32):
        return struct.unpack('f', struct.pack('I', i32))[0]
    def unpack_f64(i64):
        return struct.unpack('d', struct.pack('q', i64))[0]

    def fround(val, digits):
        return round(val, digits)

INFO  = True   # informational logging
DEBUG = False  # verbose logging
#DEBUG = True   # verbose logging
TRACE = False  # trace instruction codes
#TRACE = True   # trace instruction codes


######################################
# Basic low-level types/classes
######################################

class WAException(Exception):
    def __init__(self, message):
        self.message = message

class ValueType():
    pass

class Empty(ValueType):
    pass

class NumericValueType(ValueType):
    TYPE_NAME = "empty"

class I32(NumericValueType):
    TYPE_NAME = "i32"
    def __init__(self, val):
        assert isinstance(val, int)
        self.val = val

class I64(NumericValueType):
    TYPE_NAME = "i64"
    def __init__(self, val):
        if isinstance(val, int):
            self.val = val
        elif isinstance(val, long):
            self.val = val & 0xffffffffffffffff
        else:
            raise Exception("invalid I64 value")

class F32(NumericValueType):
    TYPE_NAME = "f32"
    def __init__(self, val):
        assert isinstance(val, float)
        #self.val = fround(val, 5) # passes func, fails loop at nesting(7.0, 4.0)
        self.val = fround(val, 6) # fails func at value-f32, fails loops at nesting(7.0, 100.0)

class F64(NumericValueType):
    TYPE_NAME = "f64"
    def __init__(self, val):
        assert isinstance(val, float)
        self.val = val

class AnyFunc(ValueType):
    TYPE_NAME = "anyfunc"

class Func(ValueType):
    TYPE_NAME = "func"

class EmptyBlockType(ValueType):
    TYPE_NAME = "emtpy_block_type"

class Type():
    def __init__(self, index, form, params, results):
        self.index = index
        self.form = form
        self.params = params
        self.results = results

class Code():
    pass

class Block(Code):
    def __init__(self, kind, type, start):
        self.kind = kind # block opcode (0x00 for init_expr)
        self.type = type # value_type
        self.locals = []
        self.start = start
        self.end = 0
        self.else_addr = 0
        self.br_addr = 0

    def update(self, end, br_addr):
        self.end = end
        self.br_addr = br_addr

class Function(Code):
    def __init__(self, type, index):
        self.type = type # value_type
        self.index = index
        self.locals = []
        self.start = 0
        self.end = 0
        self.else_addr = 0
        self.br_addr = 0

    def update(self, locals, start, end):
        self.locals = locals
        self.start = start
        self.end = end
        self.br_addr = end

class FunctionImport(Code):
    def __init__(self, type, module, field):
        self.type = type  # value_type
        self.module = module
        self.field = field


######################################
# WebAssembly spec data
######################################

MAGIC = 0x6d736100
VERSION = 0xc

STACK_SIZE      = 65536
LOCALSTACK_SIZE = 65536
BLOCKSTACK_SIZE = 8192

VALUE_TYPE = { 0x01 : I32,
               0x02 : I64,
               0x03 : F32,
               0x04 : F64,
               0x10 : AnyFunc,
               0x20 : Func,
               0x40 : EmptyBlockType }

# Block signatures for blocks, loops, ifs
BLOCK_TYPE = { 0x00 : Type(-1, Empty, [], []),
               0x01 : Type(-1, Empty, [], [I32]),
               0x02 : Type(-1, Empty, [], [I64]),
               0x03 : Type(-1, Empty, [], [F32]),
               0x04 : Type(-1, Empty, [], [F64]) }

BLOCK_NAMES = { 0x00 : "fn",
                0x01 : "block",
                0x02 : "loop",
                0x03 : "if",
                0x04 : "else" }


EXTERNAL_KIND_NAMES = { 0x0 : "Function",
                        0x1 : "Table",
                        0x2 : "Memory",
                        0x3 : "Global" }

#                 ID :  section name
SECTION_NAMES = { 0  : 'Custom',
                  1  : 'Type',
                  2  : 'Import',
                  3  : 'Function',
                  4  : 'Table',
                  5  : 'Memory',
                  6  : 'Global',
                  7  : 'Export',
                  8  : 'Start',
                  9  : 'Element',
                  10 : 'Code',
                  11 : 'Data' }

#      opcode  name              immediate(s)
OPERATOR_INFO = {
        # Control flow operators
        0x00: ['unreachable',    ''],
        0x01: ['block',          'inline_signature_type'],
        0x02: ['loop',           'inline_signature_type'],
        0x03: ['if',             'inline_signature_type'],
        0x04: ['else',           ''],
        0x05: ['select',         ''],
        0x06: ['br',             'varuint32'],
        0x07: ['br_if',          'varuint32'],
        0x08: ['br_table',       'br_table'],
        0x09: ['return',         ''],
        0x0a: ['nop',            ''],
        0x0b: ['drop',           ''],
        #0x0c
        #0x0d
        #0x0e
        0x0f: ['end',            ''],

        # Basic operators
        0x10: ['i32.const',      'varint32'],
        0x11: ['i64.const',      'varint64'],
        0x12: ['f64.const',      'uint64'],
        0x13: ['f32.const',      'uint32'],
        0x14: ['get_local',      'varuint32'],
        0x15: ['set_local',      'varuint32'],
        0x16: ['call',           'varuint32'],
        0x17: ['call_indirect',  'varuint32'],
        0x19: ['tee_local',      'varuint32'],

        0xbb: ['get_global',     'varuint32'],
        0xbc: ['set_global',     'varuint32'],

        # Memory-related operators
        0x20: ['i32.load8_s',    'memory_immediate'],
        0x21: ['i32.load8_u',    'memory_immediate'],
        0x22: ['i32.load16_s',   'memory_immediate'],
        0x23: ['i32.load16_u',   'memory_immediate'],
        0x24: ['i64.load8_s',    'memory_immediate'],
        0x25: ['i64.load8_u',    'memory_immediate'],
        0x26: ['i64.load16_s',   'memory_immediate'],
        0x27: ['i64.load16_u',   'memory_immediate'],
        0x28: ['i64.load32_s',   'memory_immediate'],
        0x29: ['i64.load32_u',   'memory_immediate'],
        0x2a: ['i32.load',       'memory_immediate'],
        0x2b: ['i64.load',       'memory_immediate'],
        0x2c: ['f32.load',       'memory_immediate'],
        0x2d: ['f64.load',       'memory_immediate'],
        0x2e: ['i32.store8',     'memory_immediate'],
        0x2f: ['i32.store16',    'memory_immediate'],
        0x30: ['i64.store8',     'memory_immediate'],
        0x31: ['i64.store16',    'memory_immediate'],
        0x32: ['i64.store32',    'memory_immediate'],
        0x33: ['i32.store',      'memory_immediate'],
        0x34: ['i64.store',      'memory_immediate'],
        0x35: ['f32.store',      'memory_immediate'],
        0x36: ['f64.store',      'memory_immediate'],
        #0x37
        #0x38
        0x39: ['grow_memory',    ''],
        #0x3a
        0x3b: ['current_memory', ''],

        # Simple operators
        0x40: ['i32.add',        ''],
        0x41: ['i32.sub',        ''],
        0x42: ['i32.mul',        ''],
        0x43: ['i32.div_s',      ''],
        0x44: ['i32.div_u',      ''],
        0x45: ['i32.rem_s',      ''],
        0x46: ['i32.rem_u',      ''],
        0x47: ['i32.and',        ''],
        0x48: ['i32.or',         ''],
        0x49: ['i32.xor',        ''],
        0x4a: ['i32.shl',        ''],
        0x4b: ['i32.shr_u',      ''],
        0x4c: ['i32.shr_s',      ''],
        0x4d: ['i32.eq',         ''],
        0x4e: ['i32.ne',         ''],
        0x4f: ['i32.lt_s',       ''],
        0x50: ['i32.le_s',       ''],
        0x51: ['i32.lt_u',       ''],
        0x52: ['i32.le_u',       ''],
        0x53: ['i32.gt_s',       ''],
        0x54: ['i32.ge_s',       ''],
        0x55: ['i32.gt_u',       ''],
        0x56: ['i32.ge_u',       ''],
        0x57: ['i32.clz',        ''],
        0x58: ['i32.ctz',        ''],
        0x59: ['i32.popcnt',     ''],
        0x5a: ['i32.eqz',        ''],
        0x5b: ['i64.add',        ''],
        0x5c: ['i64.sub',        ''],
        0x5d: ['i64.mul',        ''],
        0x5e: ['i64.div_s',      ''],
        0x5f: ['i64.div_u',      ''],
        0x60: ['i64.rem_s',      ''],
        0x61: ['i64.rem_u',      ''],
        0x62: ['i64.and',        ''],
        0x63: ['i64.or',         ''],
        0x64: ['i64.xor',        ''],
        0x65: ['i64.shl',        ''],
        0x66: ['i64.shr_u',      ''],
        0x67: ['i64.shr_s',      ''],
        0x68: ['i64.eq',         ''],
        0x69: ['i64.ne',         ''],
        0x6a: ['i64.lt_s',       ''],
        0x6b: ['i64.le_s',       ''],
        0x6c: ['i64.lt_u',       ''],
        0x6d: ['i64.le_u',       ''],
        0x6e: ['i64.gt_s',       ''],
        0x6f: ['i64.ge_s',       ''],
        0x70: ['i64.gt_u',       ''],
        0x71: ['i64.ge_u',       ''],
        0x72: ['i64.clz',        ''],
        0x73: ['i64.ctz',        ''],
        0x74: ['i64.popcnt',     ''],
        0x75: ['f32.add',        ''],
        0x76: ['f32.sub',        ''],
        0x77: ['f32.mul',        ''],
        0x78: ['f32.div',        ''],
        0x79: ['f32.min',        ''],
        0x7a: ['f32.max',        ''],
        0x7b: ['f32.abs',        ''],
        0x7c: ['f32.neg',        ''],
        0x7d: ['f32.copysign',   ''],
        0x7e: ['f32.ceil',       ''],
        0x7f: ['f32.floor',      ''],
        0x80: ['f32.trunc',      ''],
        0x81: ['f32.nearest',    ''],
        0x82: ['f32.sqrt',       ''],
        0x83: ['f32.eq',         ''],
        0x84: ['f32.ne',         ''],
        0x85: ['f32.lt',         ''],
        0x86: ['f32.le',         ''],
        0x87: ['f32.gt',         ''],
        0x88: ['f32.ge',         ''],
        0x89: ['f64.add',        ''],
        0x8a: ['f64.sub',        ''],
        0x8b: ['f64.mul',        ''],
        0x8c: ['f64.div',        ''],
        0x8d: ['f64.min',        ''],
        0x8e: ['f64.max',        ''],
        0x8f: ['f64.abs',        ''],
        0x90: ['f64.neg',        ''],
        0x91: ['f64.copysign',   ''],
        0x92: ['f64.ceil',       ''],
        0x93: ['f64.floor',      ''],
        0x94: ['f64.trunc',      ''],
        0x95: ['f64.nearest',    ''],
        0x96: ['f64.sqrt',       ''],
        0x97: ['f64.eq',         ''],
        0x98: ['f64.ne',         ''],
        0x99: ['f64.lt',         ''],
        0x9a: ['f64.le',         ''],
        0x9b: ['f64.gt',         ''],
        0x9c: ['f64.ge',         ''],

        # Conversion operators
        0x9d: ['i32.trunc_s/f32',     ''],
        0x9e: ['i32.trunc_s/f64',     ''],
        0x9f: ['i32.trunc_u/f32',     ''],
        0xa0: ['i32.trunc_u/f64',     ''],
        0xa1: ['i32.wrap/i64',        ''],
        0xa2: ['i64.trunc_s/f32',     ''],
        0xa3: ['i64.trunc_s/f64',     ''],
        0xa4: ['i64.trunc_u/f32',     ''],
        0xa5: ['i64.trunc_u/f64',     ''],
        0xa6: ['i64.extend_s/i32',    ''],
        0xa7: ['i64.extend_u/i32',    ''],
        0xa8: ['f32.convert_s/i32',   ''],
        0xa9: ['f32.convert_u/i32',   ''],
        0xaa: ['f32.convert_s/i64',   ''],
        0xab: ['f32.convert_u/i64',   ''],
        0xac: ['f32.demote/f64',      ''],
        0xad: ['f32.reinterpret/i32', ''],
        0xae: ['f64.convert_s/i32',   ''],
        0xaf: ['f64.convert_u/i32',   ''],
        0xb0: ['f64.convert_s/i64',   ''],
        0xb1: ['f64.convert_u/i64',   ''],
        0xb2: ['f64.promote/f32',     ''],
        0xb3: ['f64.reinterpret/i64', ''],
        0xb4: ['i32.reinterpret/f32', ''],
        0xb5: ['i64.reinterpret/f64', ''],

        0xb6: ['i32.rotr',       ''],
        0xb7: ['i32.rotl',       ''],
        0xb8: ['i64.rotr',       ''],
        0xb9: ['i64.rotl',       ''],
        0xba: ['i64.eqz',        ''],
           }


######################################
# General Functions
######################################

def info(str, end='\n'):
    if INFO:
        os.write(2, str + end)
        #if end == '': sys.stderr.flush()

def debug(str, end='\n'):
    if DEBUG:
        os.write(2, str + end)
        #if end == '': sys.stderr.flush()

def trace(str, end='\n'):
    if TRACE:
        os.write(2, str + end)
        #if end == '': sys.stderr.flush()

@purefunction
def bytes2uint32(b):
    return ((b[3]<<24) + (b[2]<<16) + (b[1]<<8) + b[0])

@purefunction
def bytes2uint64(b):
    return ((b[7]<<56) + (b[6]<<48) + (b[5]<<40) + (b[4]<<32) +
            (b[3]<<24) + (b[2]<<16) + (b[1]<<8) + b[0])


# https://en.wikipedia.org/wiki/LEB128
@purefunction
def read_LEB(bytes, pos, maxbits=32, signed=False):
    result = 0
    shift = 0

    bcnt = 0
    startpos = pos
    while True:
        byte = bytes[pos]
        pos += 1
        result |= ((byte & 0x7f)<<shift)
        shift +=7
        if (byte & 0x80) == 0:
            break
        # Sanity check length against maxbits
        bcnt += 1
        if bcnt > math.ceil(maxbits/7.0):
            raise Exception("Unsigned LEB at byte %s overflow" %
                    startpos)
    if signed and (shift < maxbits) and (byte & 0x40):
        # Sign extend
        result |= - (1 << shift)
    return (pos, result)

@purefunction
def read_F32(bytes):
    bits = bytes2uint32(bytes)
    return fround(unpack_f32(bits), 5)

@purefunction
def read_F64(bytes):
    bits = bytes2uint64(bytes)
    return unpack_f64(bits)

def value_repr(val):
    if isinstance(val,I32):
        return "%s:%s" % (hex(int(val.val)), val.TYPE_NAME)
    elif isinstance(val,I64):
        return "%s:%s" % (hex(int(val.val)), val.TYPE_NAME)
    elif isinstance(val,F32):
        return "%f:%s" % (float(val.val), val.TYPE_NAME)
    elif isinstance(val,F64):
        return "%f:%s" % (float(val.val), val.TYPE_NAME)
    else:
        raise Exception("unknown value type %s" % val.TYPE_NAME)

def sig_repr(sig):
    if isinstance(sig, Block):
        return "%s<0->%d>" % (
                BLOCK_NAMES[sig.kind],
                len(sig.type.results))
    elif isinstance(sig, Function):
        return "fn%d<%d/%d->%d>" % (
                sig.index, len(sig.type.params),
                len(sig.locals), len(sig.type.results))

def stack_repr(sp, s):
    return "[" + " ".join([value_repr(s[i])
                           for i in range(sp+1)]) + "]"

def localstack_repr(lsp, ls):
    return "[" + " ".join([value_repr(ls[i])
                           for i in range(lsp+1)]) + "]"

def blockstack_repr(bsp, bs):
    return "[" + " ".join(["%s(sp:0x%x/ra:0x%x)" % (
        sig_repr(bs[i][0]),bs[i][1],bs[i][2])
                           for i in range(bsp+1)]) + "]"

def dump_stacks(sp, stack, lsp, localstack, bsp, blockstack):
    if INFO:
        trace("      * stack:       %s" % (
            stack_repr(sp, stack)))
        trace("      * localstack:  %s" % (
            localstack_repr(lsp, localstack)))
        trace("      * blockstack:  %s" % (
            blockstack_repr(bsp, blockstack)))

def byte_code_repr(bytes):
    res = []
    for val in bytes:
        if val < 16:
            res.append("%x" % val)
        else:
            res.append("%x" % val)
    return "[" + ",".join(res) + "]"

def skip_immediates(code, pos):
    opcode = code[pos]
    pos += 1
    imtype = OPERATOR_INFO[opcode][1]
    if   'varint32' == imtype:
        pos, _ = read_LEB(code, pos, 32)
    elif 'varuint32' == imtype:
        pos, _ = read_LEB(code, pos, 32)
    elif 'varint64' == imtype:
        pos, _ = read_LEB(code, pos, 64)
    elif 'varuint64' == imtype:
        pos, _ = read_LEB(code, pos, 64)
    elif 'uint32' == imtype:
        pos += 4
    elif 'uint64' == imtype:
        pos += 8
    elif 'inline_signature_type' == imtype:
        pos += 1  # 1 byte signature
    elif 'memory_immediate' == imtype:
        pos, _ = read_LEB(code, pos, 32)  # flags
        pos, _ = read_LEB(code, pos, 32)  # offset
    elif 'br_table' == imtype:
        pos, count = read_LEB(code, pos, 32)  # target count
        for i in range(count):
            pos, _ = read_LEB(code, pos, 32)  # target
        pos, _ = read_LEB(code, pos, 32)  # default target
    elif '' == imtype:
        pass # no immediates
    else:
        raise Exception("unknown immediate type %s" % imtype)
    return pos

def find_blocks(code, start, end, block_map):
    debug("find_blocks start: 0x%x, end: 0x%x" % (start, end))
    pos = start

    # map of blocks: {start : (type, end), ...}
    block_end_map = {}
    # stack of blocks with current at top: (opcode, pos) tuples
    opstack = []

    #
    # Build the map of blocks
    #
    opcode = 0
    while pos <= end:
        opcode = code[pos]
        #debug("0x%x: %s, opstack: %s" % (
        #    pos, OPERATOR_INFO[opcode][0],
        #    ["%d,%s,0x%x" % (o,s.index,p) for o,s,p in opstack]))
        if   0x01 <= opcode <= 0x03:  # block, loop, if
            block_sig = BLOCK_TYPE[code[pos+1]]
            block = Block(opcode, block_sig, pos)
            opstack.append(block)
            block_map[pos] = block
        elif 0x04 == opcode:  # mark else positions
            assert opstack[-1].kind == 0x03, "else not matched with if"
            opstack[-1].else_addr = pos+1
        elif 0x0f == opcode:  # end
            if pos == end: break
            block = opstack.pop()
            if block.kind == 0x02:  # loop: label at start
                block.update(pos, block.start)
            else:  # block/if: label after end
                block.update(pos, pos+1)
            block_end_map[pos] = True
        pos = skip_immediates(code, pos)

    assert opcode == 0xf, "function block did not end with 0xf"

    #debug("block_map: %s" % block_map)
    #debug("block_end_map: %s" % block_end_map)

    return block_map

@unroll_safe
def pop_sig(sp, stack, lsp, localstack, bsp, blockstack):
    block, orig_sp, ra = blockstack[bsp]
    bsp -= 1
    t = block.type
    local_cnt = len(block.type.params) + len(block.locals)

    # Restore localstack
    lsp -= local_cnt

    # Validate return value if there is one
    if len(t.results) > 1:
        raise Exception("multiple return values unimplemented")
    if len(t.results) > sp+1:
        raise Exception("stack underflow")

    if len(t.results) == 1:
        # Restore main value stack, saving top return value
        save = stack[sp]
        sp -= 1
        if not isinstance(save, t.results[0]):
            raise WAException("call signature mismatch: %s != %s" % (
                t.results[0].TYPE_NAME, save.TYPE_NAME))

        # Restore value stack to original size prior to call/block
        if orig_sp < sp:
            sp = orig_sp

        # Put back return value if we have one
        sp += 1
        stack[sp] = save
    else:
        # Restore value stack to original size prior to call/block
        if orig_sp < sp:
            sp = orig_sp

    return block, ra, sp, lsp, bsp

def do_branch(sp, stack, lsp, localstack, bsp, blockstack, depth):
    assert bsp+1 > depth
    target_block, _, _ = blockstack[bsp-depth]
    if isinstance(target_block, Block):
        block, _, sp, lsp, bsp = pop_sig(
                sp, stack, lsp, localstack, bsp, blockstack)
        for r in range(depth):
            block, _, sp, lsp, bsp = pop_sig(
                    sp, stack, lsp, localstack, bsp, blockstack)
        return block.br_addr, sp, lsp, bsp
    else:
        for r in range(depth):
            block, _, sp, lsp, bsp = pop_sig(
                    sp, stack, lsp, localstack, bsp, blockstack)
        return target_block.end, sp, lsp, bsp
        #raise Exception("br* from function unimplemented")

def do_call_import(sp, stack, memory, host_import_func, func):
    t = func.type

    # make sure args match signature
    args = []
    for idx in range(len(t.params)-1, -1, -1):
        PType = t.params[idx]
        arg = stack[sp]
        sp -= 1
        if PType.TYPE_NAME != arg.TYPE_NAME:
            raise WAException("call signature mismatch: %s != %s" % (
                PType.TYPE_NAME, arg.TYPE_NAME))
        args.append(arg)

    # Workaround rpython failure to identify type
    results = [I32(0)]
    results.pop()
    results.extend(host_import_func(memory,
            func.module, func.field, args))

    # make sure returns match signature
    for idx, RType in enumerate(t.results):
        if idx < len(results):
            res = results[idx]
            assert isinstance(res, NumericValueType)
            if RType.TYPE_NAME != res.TYPE_NAME:
                raise Exception("return signature mismatch")
            sp += 1
            stack[sp] = res
        else:
            raise Exception("return signature mismatch")
    return sp

def do_call(sp, stack, lsp, localstack, bsp, blockstack, func, pc):

    # Push block, stack size and return address onto blockstack
    t = func.type
    assert bsp < BLOCKSTACK_SIZE, "call stack exhausted"
    bsp += 1
    blockstack[bsp] = (func, sp, pc)

    # Update the pos/instruction counter to the function
    pc = func.start

    if TRACE:
        info("  Calling function 0x%x, start: 0x%x, end: 0x%x, %d locals, %d params, %d results" % (
            func.index, func.start, func.end,
            len(func.locals), len(t.params), len(t.results)))

    # push locals onto localstack (dropping extras)
    for lidx in range(len(func.locals)-1, -1, -1):
        LType = func.locals[lidx]
        if   LType.TYPE_NAME == "i32": val = I32(0)
        elif LType.TYPE_NAME == "i64": val = I64(0)
        elif LType.TYPE_NAME == "f32": val = F32(0.0)
        elif LType.TYPE_NAME == "f64": val = F64(0.0)
        else: raise Exception("invalid locals signature")
        lsp += 1
        localstack[lsp] = val

    for tidx in range(len(t.params)-1, -1, -1):
        PType = t.params[tidx]
        val = stack[sp]
        sp -= 1
        if PType.TYPE_NAME != val.TYPE_NAME:
            raise WAException("call signature mismatch: %s != %s" % (
                PType.TYPE_NAME, val.TYPE_NAME))
        lsp += 1
        localstack[lsp] = val

    return pc, sp, lsp, bsp


# Main loop/JIT

def get_location_str(opcode, pc, code, function, table, block_map):
    return "0x%x %s(0x%x)" % (
            pc, OPERATOR_INFO[opcode][0], opcode)

@purefunction
def get_block(block_map, pc):
    return block_map[pc]

@purefunction
def get_function(function, fidx):
    return function[fidx]

@purefunction
def get_table(table, tidx):
    return table[tidx]


if IS_RPYTHON:
    # greens/reds must be sorted: ints, refs, floats
    jitdriver = JitDriver(
            greens=['opcode', 'pc',
                    'code', 'function', 'table', 'block_map'],
            reds=['sp', 'lsp', 'bsp',
                  'memory', 'stack', 'localstack', 'blockstack'],
            get_printable_location=get_location_str)

# TODO: update for MVP
def interpret_v12(host_import_func,
        # Greens
        pc, code, function, table, block_map,
        # Reds
        memory, sp, stack, lsp, localstack, bsp, blockstack):

    while pc < len(code):
        opcode = code[pc]
        if IS_RPYTHON:
            jitdriver.jit_merge_point(
                    # Greens
                    opcode=opcode,
                    pc=pc,
                    code=code,
                    function=function,
                    table=table,
                    block_map=block_map,
                    # Reds
                    memory=memory,
                    sp=sp,   stack=stack,
                    lsp=lsp, localstack=localstack,
                    bsp=bsp, blockstack=blockstack)

#        if TRACE: dump_stacks(sp, stack, lsp, localstack, bsp, blockstack)
        cur_pc = pc
        pc += 1
#        trace("    [0x%x %s (0x%x)] - " % (
#                cur_pc, OPERATOR_INFO[opcode][0], opcode),
#                end='')
        if   0x00 == opcode:  # unreachable
#            trace("unreachable")
            raise WAException("unreachable")
        elif 0x01 == opcode:  # block
            pc, ignore = read_LEB(code, pc, 32) # ignore block_type
            block = get_block(block_map, cur_pc)
            bsp += 1
            blockstack[bsp] = (block, sp, 0)
#            trace("sig: %s at 0x%x" % (
#                sig_repr(block), cur_pc))
        elif 0x02 == opcode:  # loop
            pc, ignore = read_LEB(code, pc, 32) # ignore block_type
            block = get_block(block_map, cur_pc)
            bsp += 1
            blockstack[bsp] = (block, sp, 0)
#            trace("sig: %s at 0x%x" % (
#                sig_repr(block), cur_pc))
        elif 0x03 == opcode:  # if
            pc, ignore = read_LEB(code, pc, 32) # ignore block_type
            block = get_block(block_map, cur_pc)
            bsp += 1
            blockstack[bsp] = (block, sp, 0)
            cond = stack[sp]
            sp -= 1
            if not cond.val:  # if false
                # branch to else block or after end of if
                if block.else_addr == 0:
                    # no else block so pop if block
                    bsp -= 1
                    pc = block.br_addr
                else:
                    pc = block.else_addr
#            trace("cond: %s jump to 0x%x, sig: %s at 0x%x" % (
#                value_repr(cond), pc, sig_repr(block), cur_pc))
        elif 0x04 == opcode:  # else
            block, _, sp, lsp, bsp = pop_sig(
                    sp, stack, lsp, localstack, bsp, blockstack)
            pc = block.br_addr
#            trace("of %s jump to 0x%x" % (sig_repr(block), pc))
        elif 0x05 == opcode:  # select
            cond, a, b = stack[sp], stack[sp-1], stack[sp-2]
            sp -= 2
            if cond.val:
                stack[sp] = b
            else:
                stack[sp] = a
        elif 0x06 == opcode:  # br
            pc, relative_depth = read_LEB(code, pc, 32)
            pc, sp, lsp, bsp = do_branch(sp, stack, lsp, localstack,
                    bsp, blockstack, relative_depth)
#            trace("depth: 0x%x, to: 0x%x" % (relative_depth, pc))
        elif 0x07 == opcode:  # br_if
            pc, relative_depth = read_LEB(code, pc, 32)
            cond = stack[sp]
            sp -= 1
#            trace("cond: %s, depth: 0x%x" % (
#                value_repr(cond), relative_depth))
            if cond.val:
                pc, sp, lsp, bsp = do_branch(sp, stack, lsp, localstack,
                        bsp, blockstack, relative_depth)
        elif 0x08 == opcode:  # br_table
            pc, target_count = read_LEB(code, pc, 32)
            depths = []
            for c in range(target_count):
                pc, depth = read_LEB(code, pc, 32)
                depths.append(depth)
            pc, depth = read_LEB(code, pc, 32) # default
            expr = stack[sp]
            sp -= 1
            assert isinstance(expr, I32)
            didx = int(expr.val)
            if didx >= 0 and didx < len(depths):
                depth = depths[didx]
#            trace("depths: %s, index: %d, choosen depth: 0x%x" % (
#                depths, didx, depth))
            pc, sp, lsp, bsp = do_branch(sp, stack, lsp, localstack,
                    bsp, blockstack, depth)
        elif 0x09 == opcode:  # return
            # Pop blocks until reach Function signature
            while bsp >= 0:
                if isinstance(blockstack[bsp][0], Function): break
                # We don't use pop_sig because the end opcode
                # handler will do this for us and catch the return
                # value properly.
                block = blockstack[bsp]
                bsp -= 1
            assert bsp >= 0
            block = blockstack[bsp][0]
            assert isinstance(block, Function)
            # Set instruction pointer to end of function
            # The actual pop_sig and return is handled by handling
            # the end opcode
            pc = block.end
#            trace("to 0x%x" % block.br_addr)
        elif 0x0a == opcode:  # nop
#            trace("")
            pass
        elif 0x0b == opcode:  # drop
#            trace("%s" % value_repr(stack[sp]))
            sp -= 1
        elif 0x0f == opcode:  # end
            block, ra, sp, lsp, bsp = pop_sig(
                    sp, stack, lsp, localstack, bsp, blockstack)
#            trace("of %s" % sig_repr(block))
            if isinstance(block, Function):
                # Return to return address
                pc = ra
                if bsp == -1:
                    # Return to top-level, ignoring return_addr
                    return pc, sp
                else:
                    if DEBUG:
#                        trace("  Returning from function %d to %d" % (
#                            block.index, return_addr))
                        pass
            elif isinstance(block, Block) and block.kind == 0x00:
                # this is an init_expr
                return pc, sp
            else:
                pass # end of block/loop/if, keep going
        elif 0x10 == opcode:  # i32.const
            pc, val = read_LEB(code, pc, 32, signed=True)
            sp += 1
            stack[sp] = I32(int(val))
#            trace("%s" % value_repr(stack[sp]))
        elif 0x11 == opcode:  # i64.const
            pc, val = read_LEB(code, pc, 64, signed=True)
            sp += 1
            stack[sp] = I64(val)
#            trace("%s" % value_repr(stack[sp]))
        elif 0x12 == opcode:  # f64.const
            bytes = code[pc:pc+8]
            pc += 8
            sp += 1
            stack[sp] = F64(read_F64(bytes))
#            trace("%s" % value_repr(stack[sp]))
        elif 0x13 == opcode:  # f32.const
            bytes = code[pc:pc+4]
            pc += 4
            sp += 1
            stack[sp] = F32(read_F32(bytes))
#            trace("%s" % value_repr(stack[sp]))
        elif 0x14 == opcode:  # get_local
            pc, arg = read_LEB(code, pc, 32)
#            trace("0x%x" % arg)
            sp += 1
            stack[sp] = localstack[lsp-arg]
        elif 0x15 == opcode:  # set_local
            pc, arg = read_LEB(code, pc, 32)
            val = stack[sp]
            sp -= 1
            localstack[lsp-arg] = val
#            trace("0x%x to %s" % (arg, value_repr(val)))
        elif 0x19 == opcode:  # tee_local
            pc, arg = read_LEB(code, pc, 32)
            val = stack[sp] # like set_local but do not pop
            localstack[lsp-arg] = val
#            trace("0x%x to %s" % (arg, value_repr(val)))
        elif 0xbb == opcode:  # get_global
            raise Exception("get_global unimplemented")
        elif 0xbc == opcode:  # set_global
            raise Exception("set_global unimplemented")
        elif 0x16 == opcode:  # call
            pc, fidx = read_LEB(code, pc, 32)
            func = get_function(function, fidx)

            if isinstance(func, FunctionImport):
                t = func.type
#                trace("calling import %s.%s(%s)" % (
#                    func.module, func.field,
#                    ",".join([a.TYPE_NAME for a in t.params])))
                sp = do_call_import(sp, stack,
                        memory, host_import_func, func)
            elif isinstance(func, Function):
#                trace("calling function fidx: %d" % fidx)
                pc, sp, lsp, bsp = do_call(sp, stack, lsp, localstack,
                        bsp, blockstack, func, pc)
        elif 0x17 == opcode:  # call_indirect
            # TODO: what do we do with tidx?
            pc, tidx = read_LEB(code, pc, 32)
            table_index_val = stack[sp]
            sp -= 1
            assert isinstance(table_index_val, I32)
            table_index = int(table_index_val.val)
            tbl = get_table(table, 0x20) # TODO: fix 0x20 for MVP
            if table_index < 0 or table_index >= len(tbl):
                raise WAException("undefined element")
            fidx = tbl[table_index]
#            trace("table idx: 0x%x, tidx: 0x%x, calling function fidx: 0x%x" % (
#                table_index, tidx, fidx))
            pc, sp, lsp, bsp = do_call(sp, stack, lsp, localstack,
                    bsp, blockstack, get_function(function, fidx), pc)

        # Memory immediates
        elif 0x20 <= opcode <= 0x36:
            raise Exception("memory immediates unimplemented")

        # Other Memory
        elif 0x3b == opcode:  # current_memory
            raise Exception("current_memory unimplemented")
        elif 0x39 == opcode:  # grow_memory
            raise Exception("grow_memory unimplemented")

        # Simple operations

        # i32 unary operations
        elif opcode in [0x57, 0x58, 0x59, 0x5a]:
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I32)
            if   0x58 == opcode: # i32.ctz
                count = 0
                val = int(a.val)
                while (val % 2) == 0:
                    count += 1
                    val = val / 2
                res = I32(count)
            elif 0x5a == opcode: # i32.eqz
                res = I32(a.val == 0)
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res

        # i32 binary operations
        elif 0x40 <= opcode <= 0x5a or opcode in [0xb6, 0xb7]:
            b, a = stack[sp], stack[sp-1]
            sp -= 2
            assert isinstance(a, I32) and isinstance(b, I32)
            if   0x40 == opcode: # i32.add
                res = I32(int(a.val + b.val))
            elif 0x41 == opcode: # i32.sub
                res = I32(int(a.val - b.val))
            elif 0x42 == opcode: # i32.mul
                res = I32(int(a.val * b.val))
            elif 0x4d == opcode: # i32.eq
                res = I32(int(a.val == b.val))
            elif 0x4e == opcode: # i32.ne
                res = I32(int(a.val != b.val))
            elif 0x4f == opcode: # i32.lt_s
                res = I32(int(a.val < b.val))
            elif 0x50 == opcode: # i32.le_s
                res = I32(int(a.val <= b.val))
            elif 0x51 == opcode: # i32.lt_u
                res = I32(int(a.val < b.val))
            elif 0x52 == opcode: # i32.le_u
                res = I32(int(a.val <= b.val))
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])

#            trace("(%s, %s) = %s" % (
#                value_repr(a), value_repr(b), value_repr(res)))
            sp += 1
            stack[sp] = res

        # i64 unary operations
        elif opcode in [0x72, 0x73, 0x74, 0xba]:
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I64)
            if   0xba == opcode: # i64.eqz
                res = I32(int(a.val == 0))
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res

        # i64 binary operations
        elif 0x5b <= opcode <= 0x74 or opcode in [0xb8, 0xb9]:
            b, a = stack[sp], stack[sp-1]
            sp -= 2
            assert isinstance(a, I64) and isinstance(b, I64)
            if   0x5b == opcode: # i64.add
                res = I64(int(a.val + b.val))
            elif 0x5c == opcode: # i64.sub
                res = I64(int(a.val - b.val))
            elif 0x5d == opcode: # i64.mul
                res = I64(int(a.val * b.val))
            elif 0x5e == opcode: # i64.div_s
                res = I64(int(a.val / b.val))
            elif 0x68 == opcode: # i64.eq
                res = I32(int(a.val == b.val))
            elif 0x6d == opcode: # i64.le_s
                res = I32(int(a.val <= b.val))
            elif 0x6e == opcode: # i64.gt_s
                res = I32(int(a.val > b.val))
            elif 0x70 == opcode: # i64.gt_u
                res = I32(int(a.val > b.val))
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])

#            trace("(%s, %s) = %s" % (
#                value_repr(a), value_repr(b), value_repr(res)))
            sp += 1
            stack[sp] = res

        # f32 unary operations
        elif opcode in [0x7b, 0x7c]:
            a = stack[sp]
            sp -= 1
            assert isinstance(a, F32)
            if   0x7b == opcode: # f32.abs
                res = F32(abs(a.val))
            elif  0x7c == opcode: # f32.neg
                res = F32(-a.val)
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res

        # f32 binary operations
        elif 0x75 <= opcode <= 0x88:
            b, a = stack[sp], stack[sp-1]
            sp -= 2
            assert isinstance(a, F32) and isinstance(b, F32)
            if   0x75 == opcode: # f32.add
                res = F32(float(a.val + b.val))
            elif 0x76 == opcode: # f32.sub
                res = F32(float(a.val - b.val))
            elif 0x77 == opcode: # f32.mul
                res = F32(float(a.val * b.val))
            elif 0x78 == opcode: # f32.div
                res = F32(float(a.val / b.val))
            elif 0x83 == opcode: # f32.eq
                res = I32(int(a.val == b.val))
            elif 0x85 == opcode: # f32.lt
                res = I32(int(a.val < b.val))
            elif 0x87 == opcode: # f32.gt
                res = I32(int(a.val > b.val))
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])

#            trace("(%s, %s) = %s" % (
#                value_repr(a), value_repr(b), value_repr(res)))
            sp += 1
            stack[sp] = res

        # f64 unary operations
        elif opcode in [0x8f, 0x90]:
            a = stack[sp]
            sp -= 1
            assert isinstance(a, F64)
            if   0x8f == opcode: # f32.abs
                res = F64(abs(a.val))
            elif  0x90 == opcode: # f32.neg
                res = F64(-a.val)
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res

        # f64 binary operations
        elif 0x89 <= opcode <= 0x9c:
            b, a = stack[sp], stack[sp-1]
            sp -= 2
            assert isinstance(a, F64) and isinstance(b, F64)
            if   0x89 == opcode: # f64.add
                res = F64(float(a.val + b.val))
            elif 0x8a == opcode: # f64.sub
                res = F64(float(a.val - b.val))
            else:
                raise Exception("%s unimplemented"
                        % OPERATOR_INFO[opcode][0])

#            trace("(%s, %s) = %s" % (
#                value_repr(a), value_repr(b), value_repr(res)))
            sp += 1
            stack[sp] = res

        ## conversion operations
        #elif 0x9d <= opcode <= 0xb5:

        # conversion operations
        elif 0xa1 == opcode: # i32.wrap/i64
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I64)
            res = I32(int(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xa3 == opcode: # i64.trunc_s/f64
            a = stack[sp]
            sp -= 1
            assert isinstance(a, F64)
            res = I64(int(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xa6 == opcode: # i64.extend_s/i32
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I32)
            res = I64(int(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xa7 == opcode: # i64.extend_u/i32
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I32)
            res = I64(int(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xae == opcode: # f64.convert_s/i32
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I32)
            res = F64(float(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xaf == opcode: # f64.convert_u/i32
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I32)
            res = F64(float(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xb0 == opcode: # f64.convert_s/i64
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I64)
            res = F64(float(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xb1 == opcode: # f64.convert_u/i64
            a = stack[sp]
            sp -= 1
            assert isinstance(a, I64)
            res = F64(float(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res
        elif 0xb2 == opcode: # f64.promote/f32
            a = stack[sp]
            sp -= 1
            assert isinstance(a, F32)
            res = F64(float(a.val))
#            trace("(%s) = %s" % (
#                value_repr(a), value_repr(res)))
            sp += 1
            stack[sp] = res

        else:
            raise Exception("unrecognized opcode %d" % opcode)

    return pc, sp


######################################
# Higher level classes
######################################

class Reader():
    def __init__(self, bytes):
        self.bytes = bytes
        self.pos = 0

    def read_byte(self):
        b = self.bytes[self.pos]
        self.pos += 1
        return b

    def read_word(self):
        w = bytes2uint32(self.bytes[self.pos:self.pos+4])
        self.pos += 4
        return w

    def read_bytes(self, cnt):
        assert cnt >= 0
        assert self.pos >= 0
        bytes = self.bytes[self.pos:self.pos+cnt]
        self.pos += cnt
        return bytes

    def read_LEB(self, maxbits=32, signed=False):
        [self.pos, result] = read_LEB(self.bytes, self.pos,
                maxbits, signed)
        return result

    def eof(self):
        return self.pos >= len(self.bytes)

class Memory():
    def __init__(self, pages=1, bytes=[]):
        debug("pages: %d" % pages)
        self.bytes = bytes + ([0]*((pages*(2**16))-len(bytes)))
        #self.bytes = [0]*(pages*(2**16))

    def read_byte(self, pos):
        b = self.bytes[pos]
        return b

    def write_byte(self, pos, val):
        self.bytes[pos] = val

    def read_I32(self, pos):
        return bytes2uint32(self.bytes[pos:pos+4])

    def read_I64(self, pos):
        return bytes2uint32(self.bytes[pos:pos+8])

    def write_I32(self, pos, val):
        assert isinstance(pos, int)
        self.bytes[pos]   = val & 0xff
        self.bytes[pos+1] = (val & 0xff00)>>8
        self.bytes[pos+2] = (val & 0xff0000)>>16
        self.bytes[pos+3] = (val & 0xff000000)>>24

    def write_I64(self, pos, val):
        assert isinstance(pos, int)
        self.bytes[pos]   = val & 0xff
        self.bytes[pos+1] = (val & 0xff00)>>8
        self.bytes[pos+2] = (val & 0xff0000)>>16
        self.bytes[pos+3] = (val & 0xff000000)>>24
        self.bytes[pos+4] = (val & 0xff00000000)>>32
        self.bytes[pos+5] = (val & 0xff0000000000)>>40
        self.bytes[pos+6] = (val & 0xff000000000000)>>48
        self.bytes[pos+7] = (val & 0xff00000000000000)>>56

class Import():
    def __init__(self, module, field, kind, type=0,
            element_type=0, initial=0, maximum=0, global_type=0,
            mutability=0):
        self.module = module
        self.field = field
        self.kind = kind
        self.type = type # Function
        self.element_type = element_type # Table
        self.initial = initial # Table & Memory
        self.maximum = maximum # Table & Memory

        self.global_type = global_type # Global
        self.mutability = mutability # Global

class Export():
    def __init__(self, field, kind, index):
        self.field = field
        self.kind = kind
        self.index = index


class Module():
    def __init__(self, data, host_import_func, exports):
        assert isinstance(data, str)
        self.data = data
        self.rdr = Reader([ord(b) for b in data])
        self.host_import_func = host_import_func
        self.exports = exports

        # Sections
        self.type = []
        self.import_list = []
        self.function = []
        self.table = {}
        self.export_list = []
        self.export_map = {}
        self.memory = Memory(1)  # default to 1 page

        # block/loop/if blocks {start addr: Block, ...}
        self.block_map = {}

        # Execution state
        self.sp = -1
        self.stack = [NumericValueType()] * STACK_SIZE
        self.lsp = -1
        self.localstack = [NumericValueType()] * LOCALSTACK_SIZE
        self.bsp = -1
        block = Block(0x00, BLOCK_TYPE[0x01], 0)
        self.blockstack = [(block, 0, 0)] * BLOCKSTACK_SIZE

    def dump(self):
        #debug("raw module data: %s" % self.data)
        debug("module bytes: %s" % byte_code_repr(self.rdr.bytes))
        info("")

        info("Types:")
        for i, t in enumerate(self.type):
            info("  0x%x [form: %s, params: %s, results: %s]" % (
                i, t.form.TYPE_NAME,
                [p.TYPE_NAME for p in t.params],
                [r.TYPE_NAME for r in t.results]))

        info("Imports:")
        for i, imp in enumerate(self.import_list):
            if imp.kind == 0x0:  # Function
                info("  0x%x [type: %d, '%s.%s', kind: %s (%d)]" % (
                    i, imp.type, imp.module, imp.field,
                    EXTERNAL_KIND_NAMES[imp.kind], imp.kind))
            elif imp.kind in [0x1,0x2]:  # Table & Memory
                info("  0x%x ['%s.%s', kind: %s (%d), initial: %d, maximum: %d]" % (
                    i, imp.module, imp.field,
                    EXTERNAL_KIND_NAMES[imp.kind], imp.kind,
                    imp.initial, imp.maximum))
            elif imp.kind == 0x3:  # Global
                info("  0x%x ['%s.%s', kind: %s (%d), type: %d, mutability: %d]" % (
                    i, imp.module, imp.field,
                    EXTERNAL_KIND_NAMES[imp.kind], imp.kind,
                    imp.type, imp.mutability))

        info("Functions:")
        for i, f in enumerate(self.function):
            if isinstance(f, FunctionImport):
                info("  0x%x [type: 0x%x, import: '%s.%s']" % (
                    i, f.type.index, f.module, f.field))
            else:
                info("  0x%x [type: 0x%x, locals: %s, start: 0x%x, end: 0x%x]" % (
                    i, f.type.index, [p.TYPE_NAME for p in f.locals],
                    f.start, f.end))

        info("Tables:")
        for t, e in self.table.items():
            info("  0x%x -> %s" % (t,e))

        info("Exports:")
        for i, e in enumerate(self.export_list):
            info("  0x%x [kind: %s, field: %s, index: 0x%x]" % (
                i, EXTERNAL_KIND_NAMES[e.kind], e.field, e.index))
        info("")

        bl = self.block_map
        block_keys = bl.keys()
        do_sort(block_keys)
        info("block_map: %s" % (
            ["%s[0x%x->0x%x]" % (sig_repr(bl[k]), bl[k].start, bl[k].end)
             for k in block_keys]))
        info("")


    ## Wasm top-level readers

    def read_magic(self):
        magic = self.rdr.read_word()
        if magic != MAGIC:
            raise Exception("Wanted magic 0x%x, got 0x%x" % (
                MAGIC, magic))

    def read_version(self):
        self.version = self.rdr.read_word()
        if self.version != VERSION:
            raise Exception("Wanted version 0x%x, got 0x%x" % (
                VERSION, self.version))

    def read_section(self):
        id = self.rdr.read_LEB(7)
        name = SECTION_NAMES[id]
        length = self.rdr.read_LEB(32)
        if   "Type" == name:     self.parse_Type(length)
        elif "Import" == name:   self.parse_Import(length)
        elif "Function" == name: self.parse_Function(length)
        elif "Table" == name:    self.parse_Table(length)
        elif "Memory" == name:   self.parse_Memory(length)
        elif "Global" == name:   self.parse_Global(length)
        elif "Export" == name:   self.parse_Export(length)
        elif "Start" == name:    self.parse_Start(length)
        elif "Element" == name:  self.parse_Element(length)
        elif "Code" == name:     self.parse_Code(length)
        elif "Data" == name:     self.parse_Data(length)
        else:                    self.rdr.read_bytes(length)

    def read_sections(self):
        while not self.rdr.eof():
            self.read_section()

    ## Wasm section handlers

    def parse_Type(self, length):
        count = self.rdr.read_LEB(32)
        for c in range(count):
            form = VALUE_TYPE[self.rdr.read_LEB(7)]
            params = []
            results = []
            param_count = self.rdr.read_LEB(32)
            for pc in range(param_count):
                params.append(VALUE_TYPE[self.rdr.read_LEB(32)])
            result_count = self.rdr.read_LEB(32)
            for rc in range(result_count):
                results.append(VALUE_TYPE[self.rdr.read_LEB(32)])
            tidx = len(self.type)
            self.type.append(Type(tidx, form, params, results))


    def parse_Import(self, length):
        count = self.rdr.read_LEB(32)
        for c in range(count):
            module_len = self.rdr.read_LEB(32)
            module_bytes = self.rdr.read_bytes(module_len)
            module = "".join([chr(f) for f in module_bytes])

            field_len = self.rdr.read_LEB(32)
            field_bytes = self.rdr.read_bytes(field_len)
            field = "".join([chr(f) for f in field_bytes])

            kind = self.rdr.read_byte()

            if kind == 0x0:  # Function
                sig_index = self.rdr.read_LEB(32)
                type = self.type[sig_index]
                imp = Import(module, field, kind, type=sig_index)
                self.import_list.append(imp)
                func = FunctionImport(type, module, field)
                self.function.append(func)
            elif kind in [0x1,0x2]:  # Table & Memory
                if kind == 0x1:
                    etype = self.rdr.read_LEB(7) # TODO: ignore?
                flags = self.rdr.read_LEB(32)
                initial = self.rdr.read_LEB(32)
                if flags & 0x1:
                    maximum = self.rdr.read_LEB(32)
                else:
                    maximum = 0
                self.import_list.append(Import(module, field, kind,
                    initial=initial, maximum=maximum))
            elif kind == 0x3:  # Global
                type = self.rdr.read_byte()
                mutability = self.rdr.read_LEB(1)
                self.import_list.append(Import(module, field, kind,
                    global_type=type, mutability=mutability))

    def parse_Function(self, length):
        count = self.rdr.read_LEB(32)
        for c in range(count):
            type = self.type[self.rdr.read_LEB(32)]
            idx = len(self.function)
            self.function.append(Function(type, idx))

    def parse_Table(self, length):
        count = self.rdr.read_LEB(32)
        type = self.rdr.read_LEB(7)
        assert type == 0x20  # TODO: fix for MVP

        initial = 1
        for c in range(count):
            flags = self.rdr.read_LEB(32) # TODO: fix for MVP
            initial = self.rdr.read_LEB(32) # TODO: fix for MVP
            if flags & 0x1:
                maximum = self.rdr.read_LEB(32)
            else:
                maximum = initial

        self.table[type] = [0] * initial

    def parse_Memory(self, length):
        count = self.rdr.read_LEB(32)
        assert count <= 1  # MVP
        flags = self.rdr.read_LEB(32)  # TODO: fix for MVP
        initial = self.rdr.read_LEB(32)
        if flags & 0x1:
            maximum = self.rdr.read_LEB(32)
        else:
            maximum = 0
        self.memory = Memory(initial)

    def parse_Global(self, length):
        return self.rdr.read_bytes(length)

    def parse_Export(self, length):
        count = self.rdr.read_LEB(32)
        for c in range(count):
            field_len = self.rdr.read_LEB(32)
            field_bytes = self.rdr.read_bytes(field_len)
            field = "".join([chr(f) for f in field_bytes])
            kind = self.rdr.read_byte()
            index = self.rdr.read_LEB(32)
            exp = Export(field, kind, index)
            self.export_list.append(exp)
            self.export_map[field] = exp

    def parse_Start(self, length):
        return self.rdr.read_bytes(length)

    def parse_Element(self, length):
        start = self.rdr.pos
        count = self.rdr.read_LEB(32)

        for c in range(count):
            index = self.rdr.read_LEB(32)
            assert index == 0  # Only 1 default table in MVP

            # Run the init_expr
            block = Block(0x00, BLOCK_TYPE[0x01], self.rdr.pos)
            self.bsp += 1
            self.blockstack[self.bsp] = (block, self.sp, 0)
            # WARNING: running code here to get offset!
            self.interpret()  # run iter_expr
            offset_val = self.stack[self.sp]
            self.sp -= 1
            assert isinstance(offset_val, I32)
            offset = int(offset_val.val)

            num_elem = self.rdr.read_LEB(32)
            table = self.table[0x20]  # TODO: fix for MVP
            for n in range(num_elem):
                fidx = self.rdr.read_LEB(32)
                table[offset+n] = fidx

        assert self.rdr.pos == start+length

    def parse_Code_body(self, idx):
        body_size = self.rdr.read_LEB(32)
        payload_start = self.rdr.pos
        #debug("body_size %d" % body_size)
        local_count = self.rdr.read_LEB(32)
        #debug("local_count %d" % local_count)
        locals = []
        for l in range(local_count):
            count = self.rdr.read_LEB(32)
            type = self.rdr.read_LEB(7)
            for c in range(count):
                locals.append(VALUE_TYPE[type])
        # TODO: simplify this calculation and find_blocks
        start = self.rdr.pos
        self.rdr.read_bytes(body_size - (self.rdr.pos-payload_start)-1)
        end = self.rdr.pos
        self.rdr.read_bytes(1)
        func = self.function[idx]
        assert isinstance(func,Function)
        func.update(locals, start, end)
        self.block_map = find_blocks(
                self.rdr.bytes, start, end, self.block_map)

    def parse_Code(self, length):
        body_count = self.rdr.read_LEB(32)
        import_cnt = len(self.import_list)
        for idx in range(body_count):
            self.parse_Code_body(idx + import_cnt)

    def parse_Data(self, length):
        return self.rdr.read_bytes(length)


    def interpret(self):
        self.rdr.pos, self.sp = interpret_v12(self.host_import_func,
                # Greens
                self.rdr.pos, self.rdr.bytes, self.function,
                self.table, self.block_map,
                # Reds
                self.memory, self.sp, self.stack,
                self.lsp, self.localstack, self.bsp, self.blockstack)


    def run(self, name, args):
        # Reset stacks
        self.sp  = -1
        self.lsp = -1
        self.bsp = -1

        fidx = self.export_map[name].index

        # Args are strings so convert to expected numeric type
        # Also reverse order to get correct stack order
        tparams = self.function[fidx].type.params
        #for idx in range(len(tparams)-1, -1, -1):
        for idx, arg in enumerate(args):
            arg = args[idx]
            assert isinstance(arg, str)
            #tname = tparams[len(tparams)-idx-1].TYPE_NAME
            tname = tparams[idx].TYPE_NAME
#            print("run arg idx: %d, str: %s, tname: %s" % (idx, arg,
#                tname))
            if   tname == "i32": val = I32(int(arg))
            elif tname == "i64": val = I64(int(arg))
            elif tname == "f32": val = F32(float(arg))
            elif tname == "f64": val = F64(float(arg))
            else: raise Exception("invalid argument %d: %s" % (
                idx, arg))
            self.sp += 1
            self.stack[self.sp] = val

        info("Running function %s (0x%x)" % (name, fidx))
        dump_stacks(self.sp, self.stack, self.lsp, self.localstack,
                self.bsp, self.blockstack)
        self.rdr.pos, self.sp, self.lsp, self.bsp = do_call(
                self.sp, self.stack,
                self.lsp, self.localstack,
                self.bsp, self.blockstack,
                self.function[fidx], len(self.rdr.bytes))

        self.interpret()
        if self.sp >= 0:
            ret = self.stack[self.sp]
            self.sp -= 1
            return ret
        else:
            return None


######################################
# Imported functions points
######################################


def DEBUG1(num0):
    print("DEBUG: %s" % num0)
def DEBUG2(num0, num1):
    print("DEBUG: %s %s" % (num0, num1))

def writeline(s):
    print(s)

def readline(prompt):
    res = ''
    os.write(1, prompt)
    while True:
        buf = os.read(0, 255)
        if not buf: raise EOFError()
        res += buf
        if res[-1] == '\n': return res[:-1]



# Marshall, unmarshall for the imported functions
# Current hard-coded for each function
def call_import(mem, module, field, args):
    fname = "%s.%s" % (module, field)
    host_args = [a.val for a in args]
    result = []
    if   fname == "core.DEBUG":
        if len(host_args) == 1:
            DEBUG1(host_args[0])
        elif len(host_args) == 2:
            DEBUG2(host_args[0], host_args[1])
        else:
            raise Exception("DEBUG called with > 2 args")
    elif fname == "core.writeline":
        addr = int(host_args[0])
        assert addr >= 0
        debug("writeline addr: %s" % addr)

        length = mem.read_I32(addr)
        assert length >= 0
        bytes = mem.bytes[addr+4:addr+4+length]
        str = "".join([chr(b) for b in bytes])
        writeline(str)
    elif fname == "core.readline":
        addr = int(host_args[0])
        max_length = int(host_args[1])
        assert addr >= 0
        assert max_length >= 0
        debug("readline addr: %s, max_length: %s" % (addr,
            max_length))

        try:
            res = readline("user> ")
            res = res[0:max_length]
            length = len(res)

            # first four bytes are length
            mem.write_I32(addr, 0)
            start = addr+4

            pos = start

            for i in range(length):
                mem.bytes[pos] = ord(res[i])
                pos += 1
            mem.write_I32(addr, length)

            result.append(I32(int(length)))
        except EOFError:
            result.append(I32(int(-1)))
    else:
        raise Exception("invalid import %s.%s" % (module, field))
    return result


######################################
# Entry points
######################################

def entry_point(argv):
    try:
        # Argument handling
        wasm = open(argv[1]).read()

        entry = "main"
        if len(argv) >= 3:
            entry = argv[2]

        args = []
        if len(argv) >= 4:
            args = argv[3:]

        #

        m = Module(wasm, call_import, {})
        m.read_magic()
        m.read_version()
        m.read_sections()

        m.dump()

        # Args are strings at this point
        res = m.run(entry, args)
        if res:
            info("%s(%s) = %s" % (
                entry, ",".join(args), value_repr(res)))
            print(value_repr(res))
        else:
            info("%s(%s)" % (
                entry, ",".join(args)))

    except WAException as e:
        print("%s" % e.message)
        return 1

    except Exception as e:
        if IS_RPYTHON:
            llop.debug_print_traceback(lltype.Void)
            os.write(2, "Exception: %s" % e)
        else:
            os.write(2, "".join(traceback.format_exception(*sys.exc_info())))
        return 1

    return 0

# _____ Define and setup target ___
def target(*args):
    return entry_point

# Just run entry_point if not RPython compilation
if not IS_RPYTHON:
    sys.exit(entry_point(sys.argv))

