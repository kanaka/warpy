#!/usr/bin/env python

import sys, os, math
IS_RPYTHON = sys.argv[0].endswith('rpython')

if IS_RPYTHON:
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

    def unpack_f32(i32):
        return float_unpack(i32, 4)
    def unpack_f64(i64):
        return float_unpack(i64, 8)

    def fround(val, digits):
        return round_double(val, digits)

else:
    sys.path.append(os.path.abspath('./pypy2-v5.6.0-src'))
    import traceback
    import struct

    def do_sort(a):
        a.sort()

    def unpack_f32(i32):
        return struct.unpack('f', struct.pack('I', i32))[0]
    def unpack_f64(i64):
        return struct.unpack('d', struct.pack('q', i64))[0]

    def fround(val, digits):
        return round(val, digits)

# TODO: do we need to track the stack size at each block/call and
# discard extras from the stack?

INFO  = True   # informational logging
DEBUG = False  # verbose logging
#DEBUG = True   # verbose logging
TRACE = False  # trace instruction codes
#TRACE = True   # trace instruction codes


MAGIC = 0x6d736100
VERSION = 0xc
CALLSTACK_MAX = 1024

# https://github.com/WebAssembly/design/blob/453320eb21f5e7476fb27db7874c45aa855927b7/BinaryEncoding.md#function-bodies

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

class WAException(Exception):
    def __init__(self, message):
        self.message = message

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

###

def bytes2uint32(b):
    return ((b[3]<<24) + (b[2]<<16) + (b[1]<<8) + b[0])

def bytes2uint64(b):
    return ((b[7]<<56) + (b[6]<<48) + (b[5]<<40) + (b[4]<<32) +
            (b[3]<<24) + (b[2]<<16) + (b[1]<<8) + b[0])


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

    def read_F32(self):
        bytes = self.read_bytes(4)
        bits = bytes2uint32(bytes)
        #debug("read_F32 bytes: %s, bits: %d" % (bytes, bits))
        return fround(unpack_f32(bits), 5)

    def read_F64(self):
        bytes = self.read_bytes(8)
        bits = bytes2uint64(bytes)
        #debug("read_F64 bytes: %s, bits: %d" % (bytes, bits))
        return unpack_f64(bits)

    # https://en.wikipedia.org/wiki/LEB128
    def read_LEB(self, maxbits=32, signed=False):
        result = 0
        shift = 0

        bcnt = 0
        startpos = self.pos
        while True:
            byte = self.read_byte()
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
        return result

    def eof(self):
        return self.pos >= len(self.bytes)

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

def stack_repr(vals):
    return "[" + " ".join([value_repr(v) for v in vals]) + "]"

def localstack_repr(vals):
    return "[" + " ".join([value_repr(v) for v in vals]) + "]"

def sig_repr(sig):
    if isinstance(sig, Block):
        return "%s<0->%d>" % (
                BLOCK_NAMES[sig.kind],
                len(sig.type.results))
    elif isinstance(sig, Function):
        return "fn%d<%d/%d->%d>" % (
                sig.index, len(sig.type.params),
                len(sig.locals), len(sig.type.results))

def blockstack_repr(vals):
    return "[" + " ".join(["%s(%d)" % (sig_repr(s),ss) for s,ss in vals]) + "]"

def returnstack_repr(vals):
    return "[" + " ".join([str(v) for v in vals]) + "]"

def byte_code_repr(bytes):
    res = []
    for val in bytes:
        if val < 16:
            res.append("%x" % val)
        else:
            res.append("%x" % val)
    return "[" + ",".join(res) + "]"

def drop_immediates(rdr, opcode):
    imtype = OPERATOR_INFO[opcode][1]
    if   'varint32' == imtype:
        rdr.read_LEB(32)
    elif 'varuint32' == imtype:
        rdr.read_LEB(32)
    elif 'varint64' == imtype:
        rdr.read_LEB(64)
    elif 'varuint64' == imtype:
        rdr.read_LEB(64)
    elif 'uint32' == imtype:
        rdr.read_bytes(4)
    elif 'uint64' == imtype:
        rdr.read_bytes(8)
    elif 'inline_signature_type' == imtype:
        rdr.read_byte()  # 1 byte signature
    elif 'memory_immediate' == imtype:
        rdr.read_LEB(32) # flags
        rdr.read_LEB(32) # offset
    elif 'br_table' == imtype:
        count = rdr.read_LEB(32) # target_count
        for i in range(count):
            rdr.read_LEB(32)  # targets
        rdr.read_LEB(32) # default taget
    elif '' == imtype:
        pass # no immediates
    else:
        raise Exception("unknown immediate type %s" % imtype)


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
        self.stack = []
        self.localstack = []
        self.returnstack = []
        self.blockstack = []

    def dump_stacks(self):
        if INFO:
            trace("      * stack:       %s" % (
                stack_repr(self.stack)))
            trace("      * localstack:  %s" % (
                localstack_repr(self.localstack)))
            trace("      * blockstack:    %s" % (
                blockstack_repr(self.blockstack)))
            trace("      * returnstack: %s" % (
                returnstack_repr(self.returnstack)))

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
            self.blockstack.append((block, len(self.stack)))
            # WARNING: running code here to get offset!
            self.run_code_v12()
            offset_val = self.stack.pop()
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
        self.find_blocks(func, start, end)

    def parse_Code(self, length):
        body_count = self.rdr.read_LEB(32)
        import_cnt = len(self.import_list)
        for idx in range(body_count):
            self.parse_Code_body(idx + import_cnt)

    def parse_Data(self, length):
        return self.rdr.read_bytes(length)

    ###

    def find_blocks(self, func, start, end):
        #debug("bytes: %s" % bytes)
        debug("find_blocks start: 0x%x, end: 0x%x" % (start, end))
        # TODO: remove extra reader
        rdr = Reader(self.rdr.bytes)
        rdr.pos = start

        # map of blocks: {start : (type, end), ...}
        block_end_map = {}
        # stack of blocks with current at top: (opcode, pos) tuples
        opstack = []

        #
        # Build the map of blocks
        #
        opcode = 0
        while rdr.pos <= end:
            pos = rdr.pos
            opcode = rdr.read_byte()
            #debug("0x%x: %s, opstack: %s" % (
            #    pos, OPERATOR_INFO[opcode][0],
            #    ["%d,%s,0x%x" % (o,s.index,p) for o,s,p in opstack]))
            if   0x01 <= opcode <= 0x03:  # block, loop, if
                block_sig = BLOCK_TYPE[rdr.read_byte()]
                block = Block(opcode, block_sig, pos)
                opstack.append(block)
                self.block_map[pos] = block
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
            else:
                drop_immediates(rdr, opcode)

        assert opcode == 0xf, "function block did not end with 0xf"

        #debug("block_map: %s" % block_map)
        #debug("block_end_map: %s" % block_end_map)

    ###

    # TODO: update for MVP
    def run_code_v12(self):
        while not self.rdr.eof():
            self.dump_stacks()
            cur_pos = self.rdr.pos
            opcode = self.rdr.read_byte()
            trace("    [0x%x %s (0x%x)] - " % (
                    cur_pos, OPERATOR_INFO[opcode][0], opcode),
                    end='')
            if   0x00 == opcode:  # unreachable
                trace("unreachable")
                raise WAException("unreachable")
            elif 0x01 == opcode:  # block
                self.rdr.read_LEB(32) # ignore block_type
                block = self.block_map[cur_pos]
                self.blockstack.append((block, len(self.stack)))
                trace("sig: %s at 0x%x" % (
                    sig_repr(block), cur_pos))
            elif 0x02 == opcode:  # loop
                self.rdr.read_LEB(32) # ignore block_type
                block = self.block_map[cur_pos]
                self.blockstack.append((block, len(self.stack)))
                trace("sig: %s at 0x%x" % (
                    sig_repr(block), cur_pos))
            elif 0x03 == opcode:  # if
                self.rdr.read_LEB(32) # ignore block_type
                block = self.block_map[cur_pos]
                self.blockstack.append((block, len(self.stack)))
                cond = self.stack.pop()
                if not cond.val:  # if false
                    # branch to else block or after end of if
                    if block.else_addr == 0:
                        # no else block so pop if block
                        self.blockstack.pop()
                        self.rdr.pos = block.br_addr
                    else:
                        self.rdr.pos = block.else_addr
                trace("cond: %s jump to %s, sig: %s at 0x%x" % (
                    value_repr(cond), hex(self.rdr.pos),
                    sig_repr(block), cur_pos))
            elif 0x04 == opcode:  # else
                block = self.pop_sig()
                self.rdr.pos = block.br_addr
                trace("of %s jump to %s" % (
                    sig_repr(block), hex(self.rdr.pos)))
            elif 0x05 == opcode:  # select
                cond = self.stack.pop()
                a, b = self.stack.pop(), self.stack.pop()
                if cond.val:
                    self.stack.append(b)
                else:
                    self.stack.append(a)
            elif 0x06 == opcode:  # br
                relative_depth = self.rdr.read_LEB(32)
                self.do_branch(relative_depth)
                trace("depth: 0x%x, to: 0x%x" % (
                    relative_depth, self.rdr.pos))
            elif 0x07 == opcode:  # br_if
                relative_depth = self.rdr.read_LEB(32)
                cond = self.stack.pop()
                trace("cond: %s, depth: 0x%x" % (
                    value_repr(cond), relative_depth))
                if cond.val:
                    self.do_branch(relative_depth)
            elif 0x08 == opcode:  # br_table
                target_count = self.rdr.read_LEB(32) # +1 to catch default
                depths = []
                for c in range(target_count):
                    depths.append(self.rdr.read_LEB(32))
                depth = self.rdr.read_LEB(32) # default
                expr = self.stack.pop()
                assert isinstance(expr, I32)
                didx = int(expr.val)
                if didx >= 0 and didx < len(depths):
                    depth = depths[didx]
                trace("depths: %s, index: %d, choosen depth: 0x%x" % (
                    depths, didx, depth))
                self.do_branch(depth)
            elif 0x09 == opcode:  # return
                # Pop blocks until reach Function signature
                while len(self.blockstack) > 0:
                    if isinstance(self.blockstack[-1][0], Function): break
                    # We don't use sig_pop because the end opcode
                    # handler will do this for us and catch the return
                    # value properly.
                    block = self.blockstack.pop()
                assert len(self.blockstack) > 0
                block = self.blockstack[-1][0]
                assert isinstance(block, Function)
                # Set instruction pointer to end of function
                # The actual sig_pop and return is handled by handling
                # the end opcode
                self.rdr.pos = block.end
                trace("to 0x%x" % block.br_addr)
            elif 0x0a == opcode:  # nop
                trace("")
            elif 0x0b == opcode:  # drop
                trace("%s" % value_repr(self.stack[-1]))
                self.stack.pop()
            elif 0x0f == opcode:  # end
                block = self.pop_sig()
                trace("of %s" % sig_repr(block))
                if isinstance(block, Function):
                    # Return to return address
                    return_addr = self.returnstack.pop()
                    self.rdr.pos = return_addr
                    if len(self.returnstack) == 0:
                        # Return to top-level, ignoring return_addr
                        return None
                    else:
                        if DEBUG:
                            trace("  Returning from function %d to %d" % (
                                block.index, return_addr))
                elif isinstance(block, Block) and block.kind == 0x00:
                    # this is an init_expr
                    return
                else:
                    pass # end of block/loop/if, keep going
            elif 0x10 == opcode:  # i32.const
                self.stack.append(I32(int(self.rdr.read_LEB(32, signed=True))))
                trace("%s" % value_repr(self.stack[-1]))
            elif 0x11 == opcode:  # i64.const
                self.stack.append(I64(self.rdr.read_LEB(64, signed=True)))
                trace("%s" % value_repr(self.stack[-1]))
            elif 0x12 == opcode:  # f64.const
                self.stack.append(F64(self.rdr.read_F64()))
                trace("%s" % value_repr(self.stack[-1]))
            elif 0x13 == opcode:  # f32.const
                self.stack.append(F32(self.rdr.read_F32()))
                trace("%s" % value_repr(self.stack[-1]))
            elif 0x14 == opcode:  # get_local
                arg = self.rdr.read_LEB(32)
                trace("0x%x" % arg)
                self.stack.append(self.localstack[-1-arg])
            elif 0x15 == opcode:  # set_local
                arg = self.rdr.read_LEB(32)
                val = self.stack.pop()
                self.localstack[-1-arg] = val
                trace("0x%x to %s" % (arg, value_repr(val)))
            elif 0x19 == opcode:  # tee_local
                arg = self.rdr.read_LEB(32)
                val = self.stack[-1] # like set_local but do not pop
                self.localstack[-1-arg] = val
                trace("0x%x to %s" % (arg, value_repr(val)))
            elif 0xbb == opcode:  # get_global
                raise Exception("get_global unimplemented")
            elif 0xbc == opcode:  # set_global
                raise Exception("set_global unimplemented")
            elif 0x16 == opcode:  # call
                fidx = self.rdr.read_LEB(32)
                func = self.function[fidx]

                if isinstance(func, FunctionImport):
                    t = func.type
                    trace("calling import %s.%s(%s)" % (
                        func.module, func.field,
                        ",".join([a.TYPE_NAME for a in t.params])))
                    self.do_call_import(fidx)
                elif isinstance(func, Function):
                    trace("calling function fidx: %d" % fidx)
                    self.do_call(fidx)
            elif 0x17 == opcode:  # call_indirect
                # TODO: what do we do with tidx?
                tidx = self.rdr.read_LEB(32)
                table_index_val = self.stack.pop()
                assert isinstance(table_index_val, I32)
                table_index = int(table_index_val.val)
                table = self.table[0x20] # TODO: fix 0x20 for MVP
                if table_index < 0 or table_index >= len(table):
                    raise WAException("undefined element")
                fidx = table[table_index]
                trace("table idx: 0x%x, tidx: 0x%x, calling function fidx: 0x%x" % (
                    table_index, tidx, fidx))
                self.do_call(fidx)

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
                a = self.stack.pop()
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
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)

            # i32 binary operations
            elif 0x40 <= opcode <= 0x5a or opcode in [0xb6, 0xb7]:
                b, a = self.stack.pop(), self.stack.pop()
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

                trace("(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)

            # i64 unary operations
            elif opcode in [0x72, 0x73, 0x74, 0xba]:
                a = self.stack.pop()
                assert isinstance(a, I64)
                if   0xba == opcode: # i64.eqz
                    res = I32(int(a.val == 0))
                else:
                    raise Exception("%s unimplemented"
                            % OPERATOR_INFO[opcode][0])
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)

            # i64 binary operations
            elif 0x5b <= opcode <= 0x74 or opcode in [0xb8, 0xb9]:
                b, a = self.stack.pop(), self.stack.pop()
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

                trace("(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)

            # f32 unary operations
            elif opcode in [0x7b, 0x7c]:
                a = self.stack.pop()
                assert isinstance(a, F32)
                if   0x7b == opcode: # f32.abs
                    res = F32(abs(a.val))
                elif  0x7c == opcode: # f32.neg
                    res = F32(-a.val)
                else:
                    raise Exception("%s unimplemented"
                            % OPERATOR_INFO[opcode][0])
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)

            # f32 binary operations
            elif 0x75 <= opcode <= 0x88:
                b, a = self.stack.pop(), self.stack.pop()
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

                trace("(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)

            # f64 unary operations
            elif opcode in [0x8f, 0x90]:
                a = self.stack.pop()
                assert isinstance(a, F64)
                if   0x8f == opcode: # f32.abs
                    res = F64(abs(a.val))
                elif  0x90 == opcode: # f32.neg
                    res = F64(-a.val)
                else:
                    raise Exception("%s unimplemented"
                            % OPERATOR_INFO[opcode][0])
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)

            # f64 binary operations
            elif 0x89 <= opcode <= 0x9c:
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, F64) and isinstance(b, F64)
                if   0x89 == opcode: # f64.add
                    res = F64(float(a.val + b.val))
                elif 0x8a == opcode: # f64.sub
                    res = F64(float(a.val - b.val))
                else:
                    raise Exception("%s unimplemented"
                            % OPERATOR_INFO[opcode][0])

                trace("(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)

#            # conversion operations
#            elif 0x9d <= opcode <= 0xb5:

            # conversion operations
            elif 0xa1 == opcode: # i32.wrap/i64
                a = self.stack.pop()
                assert isinstance(a, I64)
                res = I32(int(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xa3 == opcode: # i64.trunc_s/f64
                a = self.stack.pop()
                assert isinstance(a, F64)
                res = I64(int(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xa6 == opcode: # i64.extend_s/i32
                a = self.stack.pop()
                assert isinstance(a, I32)
                res = I64(int(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xa7 == opcode: # i64.extend_u/i32
                a = self.stack.pop()
                assert isinstance(a, I32)
                res = I64(int(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xae == opcode: # f64.convert_s/i32
                a = self.stack.pop()
                assert isinstance(a, I32)
                res = F64(float(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xaf == opcode: # f64.convert_u/i32
                a = self.stack.pop()
                assert isinstance(a, I32)
                res = F64(float(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xb0 == opcode: # f64.convert_s/i64
                a = self.stack.pop()
                assert isinstance(a, I64)
                res = F64(float(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xb1 == opcode: # f64.convert_u/i64
                a = self.stack.pop()
                assert isinstance(a, I64)
                res = F64(float(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xb2 == opcode: # f64.promote/f32
                a = self.stack.pop()
                assert isinstance(a, F32)
                res = F64(float(a.val))
                trace("(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)

            else:
                raise Exception("unrecognized opcode %d" % opcode)

    def pop_sig(self):
        block, stacksize = self.blockstack.pop()
        t = block.type
        local_cnt = len(block.type.params) + len(block.locals)

        # Restore local stack
        for i in range(local_cnt):
            self.localstack.pop()

        # Validate return value if there is one
        if len(t.results) > 1:
            raise Exception("multiple return values unimplemented")
        if len(self.stack) < len(t.results):
            raise Exception("stack underflow")

        if len(t.results) == 1:
            # Restore main value stack, saving top return value
            save = self.stack.pop()
            if not isinstance(save, t.results[0]):
                raise WAException("call signature mismatch: %s != %s" % (
                    t.results[0].TYPE_NAME, save.TYPE_NAME))

            # Restore value stack to original size prior to call/block
            while len(self.stack) > stacksize:
                self.stack.pop()

            # Put back return value if we have one
            self.stack.append(save)
        else:
            # Restore value stack to original size prior to call/block
            while len(self.stack) > stacksize:
                self.stack.pop()

        return block


    def do_branch(self, depth):
        assert len(self.blockstack) > depth
        target_block, stack_size = self.blockstack[-1-depth]
        if isinstance(target_block, Block):
            block = self.pop_sig()
            for r in range(depth):
                block = self.pop_sig()
            self.rdr.pos = block.br_addr
        else:
            for r in range(depth):
                block = self.pop_sig()
            self.rdr.pos = target_block.end
            #raise Exception("br* from function unimplemented")


    def do_call(self, fidx):
        func = self.function[fidx]

        # Push type onto blockstack
        t = func.type
        self.blockstack.append((func, len(self.stack)))

        # Push return address onto returnstack
        assert len(self.returnstack) < CALLSTACK_MAX, "call stack exhausted"
        self.returnstack.append(self.rdr.pos)

        # Update the pos/instruction counter to the function
        self.rdr.pos = func.start

        if TRACE:
            info("  Calling function 0x%x, start: 0x%x, end: 0x%x, %d locals, %d params, %d results" % (
                fidx, func.start, func.end,
                len(func.locals), len(t.params), len(t.results)))
            debug("    bytes: %s" % (
                byte_code_repr(self.rdr.bytes[func.start:func.end])))

        # push locals onto localstack (dropping extras)
        for lidx in range(len(func.locals)-1, -1, -1):
            LType = func.locals[lidx]
            if   LType.TYPE_NAME == "i32": val = I32(0)
            elif LType.TYPE_NAME == "i64": val = I64(0)
            elif LType.TYPE_NAME == "f32": val = F32(0.0)
            elif LType.TYPE_NAME == "f64": val = F64(0.0)
            else: raise Exception("invalid locals signature")
            self.localstack.append(val)

        for tidx in range(len(t.params)-1, -1, -1):
            PType = t.params[tidx]
            val = self.stack.pop()
            if PType.TYPE_NAME != val.TYPE_NAME:
                raise WAException("call signature mismatch: %s != %s" % (
                    PType.TYPE_NAME, val.TYPE_NAME))
            self.localstack.append(val)

        self.rdr.pos = func.start


    def do_call_import(self, fidx):
        func = self.function[fidx]
        t = func.type

        # make sure args match signature
        args = []
        for idx in range(len(t.params)-1, -1, -1):
            PType = t.params[idx]
            arg = self.stack.pop()
            if PType.TYPE_NAME != arg.TYPE_NAME:
                raise WAException("call signature mismatch: %s != %s" % (
                    PType.TYPE_NAME, arg.TYPE_NAME))
            args.append(arg)

        # Workaround rpython failure to identify type
        results = [I32(0)]
        results.pop()
        results.extend(self.host_import_func(self.memory,
                func.module, func.field, args))

        # make sure returns match signature
        for idx, RType in enumerate(t.results):
            if idx < len(results):
                res = results[idx]
                assert isinstance(res, NumericValueType)
                if RType.TYPE_NAME != res.TYPE_NAME:
                    raise Exception("return signature mismatch")
                self.stack.append(res)
            else:
                raise Exception("return signature mismatch")


    def run(self, name, args):
        # Reset stacks
        self.stack = []
        self.localstack = []
        self.returnstack = []
        self.blockstack = []

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
            self.stack.append(val)

        info("Running function %s (0x%x)" % (name, fidx))
        self.dump_stacks()
        self.do_call(fidx)

        self.run_code_v12()
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            return None


### Imported functions

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



### Main

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

