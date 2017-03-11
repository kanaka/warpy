#!/usr/bin/env python

import sys, os, math
IS_RPYTHON = sys.argv[0].endswith('rpython')

if IS_RPYTHON:
    from rpython.rtyper.lltypesystem import lltype
    from rpython.rtyper.lltypesystem.lloperation import llop
    from rpython.rlib.listsort import TimSort
    from rpython.rlib.rstruct.ieee import float_unpack

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

# TODO: do we need to track the stack size at each block/call and
# discard extras from the stack?


MAGIC = 0x6d736100
VERSION = 0xc

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
        assert isinstance(val, int)
        self.val = val

class F32(NumericValueType):
    TYPE_NAME = "f32"
    def __init__(self, val):
        assert isinstance(val, float)
        self.val = val

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
                0x03 : "if" }


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


class Code():
    pass

class Block(Code):
    def __init__(self, kind, type, start, end):
        self.kind = kind # block opcode
        self.type = type # value_type
        self.locals = []
        self.start = start
        self.end = end
        self.label_addr = 0

    def update(self, label_addr):
        self.label_addr = label_addr

class Function(Code):
    def __init__(self, type, index):
        self.type = type # value_type
        self.index = index
        self.locals = []
        self.start = 0
        self.end = 0
        self.label_addr = 0

    def update(self, locals, start, end):
        self.locals = locals
        self.start = start
        self.end = end
        self.label_addr = end

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
        assert cnt > 0
        assert self.pos >= 0
        bytes = self.bytes[self.pos:self.pos+cnt]
        self.pos += cnt
        return bytes

    def read_F32(self):
        bytes = self.read_bytes(4)
        bits = bytes2uint32(bytes)
        #print("read_F32 bytes: %s, bits: %d" % (bytes, bits))
        return unpack_f32(bits)

    def read_F64(self):
        bytes = self.read_bytes(8)
        bits = bytes2uint64(bytes)
        #print("read_F64 bytes: %s, bits: %d" % (bytes, bits))
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
            if (byte & 0x80) == 0:
                break
            shift +=7
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
        return "0x%x:%s" % (int(val.val), val.TYPE_NAME)
    elif isinstance(val,I64):
        return "0x%x:%s" % (int(val.val), val.TYPE_NAME)
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

def sigstack_repr(vals):
    return "[" + " ".join([sig_repr(s) for s in vals]) + "]"

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
    if 0x01 <= opcode <= 0x03:  # blook/loop/if
        rdr.read_LEB(32) # block_type
    elif 0x06 <= opcode <= 0x07:  # br/br_if
        rdr.read_LEB(32) # relative_depth
    elif 0x08 == opcode:  # br_table
        count = rdr.read_LEB(32) # target_count
        for i in range(count):
            rdr.read_LEB(32)  # br_table targets
        rdr.read_LEB(32) # default taget
    elif 0x10 == opcode:  # i32.const
        rdr.read_LEB(32, signed=True) # varint32
    elif 0x11 == opcode:  # i64.const
        rdr.read_LEB(64, signed=True) # varint64
    elif 0x12 == opcode:  # f64.const
        rdr.read_bytes(8) # uint64
    elif 0x13 == opcode:  # f32.const
        rdr.read_bytes(4) # uint32
    elif 0x14 <= opcode <= 0x17:  # get_local/set_local/call/call_indirect
        rdr.read_LEB(32) # index
    elif 0x19 == opcode:  # tee_local
        rdr.read_LEB(32) # index
    elif 0x20 <= opcode <= 0x36:
        rdr.read_LEB(32) # memory_immediate flags
        rdr.read_LEB(32) # memory_immediate offset
    elif 0xbb <= opcode <= 0xbc:  # get_global/set_global
        rdr.read_LEB(32) # memory_immediate flags


class Module():
    def __init__(self, data, host_import_func):
        assert isinstance(data, str)
        self.data = data
        self.rdr = Reader([ord(b) for b in data])
        self.host_import_func = host_import_func

        # Sections
        self.type = []
        self.import_list = []
        self.function = []
        self.export_list = []
        self.export_map = {}

        # block/loop/if blocks {start addr: Block, ...}
        self.block_map = {}
        # references back to blocks for each br/br_if/br_table
        self.branch_map = {}

        # Execution state
        self.stack = []
        self.localstack = []
        self.returnstack = []
        self.sigstack = []

    def dump_stacks(self):
        print("      * stack:       %s" % (
            stack_repr(self.stack)))
        print("      * localstack:  %s" % (
            localstack_repr(self.localstack)))
        print("      * sigstack:    %s" % (
            sigstack_repr(self.sigstack)))
        print("      * returnstack: %s" % (
            returnstack_repr(self.returnstack)))

    def dump(self):
        #print("data: %s" % self.data)
        #print("rdr.pos: %s" % self.rdr.pos)
        print("bytes: %s" % byte_code_repr(self.rdr.bytes))
        bl = self.block_map
        block_keys = bl.keys()
        do_sort(block_keys)
        print("block_map: %s" % (
            ["%s[0x%x->0x%x]" % (sig_repr(bl[k]), bl[k].start, bl[k].end)
             for k in block_keys]))
        br = self.branch_map
        branch_keys = br.keys()
        do_sort(branch_keys)
        print("branch_map: %s" % (
            ["0x%x->0x%x" % (k, br[k].start)
             for k in branch_keys]))
        print("")

        print("Types:")
        for i, t in enumerate(self.type):
            print("  %d [form: %s, params: %s, results: %s]" % (
                i, t.form.TYPE_NAME,
                [p.TYPE_NAME for p in t.params],
                [r.TYPE_NAME for r in t.results]))

        print("Imports:")
        for i, imp in enumerate(self.import_list):
            if imp.kind == 0x0:  # Function
                print("  %d [type: %d, '%s.%s', kind: %s (%d)]" % (
                    i, imp.type, imp.module, imp.field,
                    EXTERNAL_KIND_NAMES[imp.kind], imp.kind))
            elif imp.kind in [0x1,0x2]:  # Table & Memory
                print("  %d ['%s.%s', kind: %s (%d), initial: %d, maximum: %d]" % (
                    i, imp.module, imp.field,
                    EXTERNAL_KIND_NAMES[imp.kind], imp.kind,
                    imp.initial, imp.maximum))
            elif imp.kind == 0x3:  # Global
                print("  %d ['%s.%s', kind: %s (%d), type: %d, mutability: %d]" % (
                    i, imp.module, imp.field,
                    EXTERNAL_KIND_NAMES[imp.kind], imp.kind,
                    imp.type, imp.mutability))

        print("Functions:")
        for i, f in enumerate(self.function):
            if isinstance(f, FunctionImport):
                print("  %d [type: %d, import: '%s.%s']" % (
                    i, f.type.index, f.module, f.field))
            else:
                print("  %d [type: %d, locals: %s, start: 0x%x, end: 0x%x]" % (
                    i, f.type.index, [p.TYPE_NAME for p in f.locals],
                    f.start, f.end))

        print("Exports:")
        for i, e in enumerate(self.export_list):
            print("  %d [kind: %s, field: %s, index: %d]" % (
                i, EXTERNAL_KIND_NAMES[e.kind], e.field, e.index))


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
        return self.rdr.read_bytes(length)

    def parse_Memory(self, length):
        return self.rdr.read_bytes(length)

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
        return self.rdr.read_bytes(length)

    def parse_Code_body(self, idx):
        body_size = self.rdr.read_LEB(32)
        payload_start = self.rdr.pos
        #print("body_size %d" % body_size)
        local_count = self.rdr.read_LEB(32)
        #print("local_count %d" % local_count)
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
        #print("bytes: %s" % bytes)
        #print("start: 0x%x, end: 0x%x" % (start, end))
        # TODO: remove extra reader
        rdr = Reader(self.rdr.bytes)
        rdr.pos = start

        # map of blocks: {start : (type, end), ...}
        block_start_map = {}
        block_end_map = {}
        # stack of blocks with current at top: (opcode, pos) tuples
        #opstack = [(-1, BLOCK_TYPE[0], start)]  # implicit function block
        opstack = []  # implicit function block

        #
        # Build the map of blocks
        #
        opcode = 0
        while rdr.pos <= end:
            pos = rdr.pos
            opcode = rdr.read_byte()
            #print("0x%x: opcode 0x%x, opstack: %s" % (pos, opcode, opstack))
            if   0x01 <= opcode <= 0x03:  # block, loop, if
                block_sig = BLOCK_TYPE[rdr.read_byte()]
                opstack.append((opcode, block_sig, pos))
            elif 0x04 == opcode:  # else is end of if and start of end
                block_opcode, block_sig, block_start = opstack.pop()
                assert block_opcode == 0x03, "else not matched with if"
                block_start_map[block_start] = (block_opcode, block_sig, pos)
                block_end_map[pos] = True
                opstack.append((opcode, block_sig, pos))
            elif 0x0f == opcode:  # end
                if pos == end: break
                block_opcode, block_sig, block_start = opstack.pop()
                block_start_map[block_start] = (block_opcode, block_sig, pos)
                block_end_map[pos] = True
            else:
                drop_immediates(rdr, opcode)

        assert opcode == 0xf, "function block did not end with 0xf"

        #print("block_start_map: %s" % block_start_map)
        #print("block_end_map: %s" % block_end_map)

        # Create the blocks
        for start, (kind, sig, end) in block_start_map.items():
            if kind == -1: # function
                block = func
            else: # block
                block = Block(kind, sig, start, end)
                if   0x02 == kind:  # loop
                    block.update(block.start) # label at top
                elif 0x04 == kind:  # else
                    block.update(block.end) # label at else
                else: # block, if
                    block.update(block.end+1) # label after end
            self.block_map[start] = block

        #
        # Scan for branch instructions and update Blocks with label
        #
        rdr.pos = start  # reset to beginning of function
        blockstack = []

        while rdr.pos < end:
            pos = rdr.pos
            opcode = rdr.read_byte()
            #print("%d: opcode 0x%x, blockstack: %s" % (pos, opcode, blockstack))
            if pos in block_start_map:
                block = self.block_map[pos]
                blockstack.append(block)
            elif pos in block_end_map:
                blockstack.pop()
            elif 0x06 <= opcode <= 0x08:  # br, br_if, br_table
                target_count = 1
                if 0x08 == opcode: # br_table
                    target_count = rdr.read_LEB(32)+1 # +1 to catch default
                for c in range(target_count):
                    relative_depth = rdr.read_LEB(32)
                    print("0x%x: branch opcode 0x%x, depth: %d" % (
                        pos, opcode, relative_depth))
                    block = blockstack[-1-relative_depth]
                    self.branch_map[pos] = block
                continue # already skipped immediate

            drop_immediates(rdr, opcode)

        #print("block_map: %s" % self.block_map)
        #print("branch_map: %s" % self.branch_map)

    ###

    # TODO: update for MVP
    def run_code_v12(self):
        while not self.rdr.eof():
            self.dump_stacks()
            cur_pos = self.rdr.pos
            opcode = self.rdr.read_byte()
            print "    [0x%x op 0x%x] -" % (cur_pos, opcode),
            if   0x00 == opcode:  # unreachable
                print("trap")
                raise Exception("Immediate trap")
            elif 0x01 == opcode:  # block
                self.rdr.read_LEB(32) # ignore block_type
                block = self.block_map[cur_pos]
                self.sigstack.append(block)
                print("block sig: %s at 0x%x" % (
                    sig_repr(block), cur_pos))
            elif 0x02 == opcode:  # loop
                self.rdr.read_LEB(32) # ignore block_type
                block = self.block_map[cur_pos]
                self.sigstack.append(block)
                print("loop sig: %s at 0x%x" % (
                    sig_repr(block), cur_pos))
            elif 0x03 == opcode:  # if
                self.rdr.read_LEB(32) # ignore block_type
                block = self.block_map[cur_pos]
                assert isinstance(block, Block)
                cond = self.stack.pop()
                if cond.val:  # if true
                    self.sigstack.append(block)
                if not cond.val:  # if false
                    # Branch to else or to after end
                    if (block.label_addr in self.block_map and
                        self.block_map[block.label_addr].kind == 0x04):
                        # pop if, push else
                        block = self.block_map[block.label_addr]
                        self.sigstack.append(block)
                    else:
                        # branch to after end of if
                        self.rdr.pos = block.label_addr
                print("if cond: %s, sig: %s at 0x%x" % (
                    value_repr(cond), sig_repr(block), cur_pos))
            # NOTE: See end (0x07) for else (0x04)
            elif 0x05 == opcode:  # select
                raise Exception("select unimplemented")
            elif 0x06 == opcode:  # br
                relative_depth = self.rdr.read_LEB(32)
                self.do_branch(relative_depth)
                print("br depth: 0x%x, to: 0x%x" % (
                    relative_depth, self.rdr.pos))
            elif 0x07 == opcode:  # br_if
                relative_depth = self.rdr.read_LEB(32)
                cond = self.stack.pop()
                if cond.val:
                    self.do_branch(relative_depth)
                print("br_if cond: %s, depth: 0x%x" % (
                    value_repr(cond), relative_depth))
            elif 0x08 == opcode:  # br_table
                raise Exception("br_table unimplemented")
            elif 0x09 == opcode:  # return
                # Pop blocks until reach Function signature
                while len(self.sigstack) > 0:
                    if isinstance(self.sigstack[-1], Function): break
                    block = self.sigstack.pop()
                    local_cnt = len(block.type.params) + len(block.locals)
                    for i in range(local_cnt):
                        self.localstack.pop()
                assert len(self.sigstack) > 0
                block = self.sigstack[-1]
                assert isinstance(block, Function)
                # Set instruction pointer to end of function
                self.rdr.pos = block.label_addr
            elif 0x0a == opcode:  # nop
                pass
            elif 0x0b == opcode:  # drop
                print("drop: %s" % value_repr(self.stack[-1]))
                self.stack.pop()
            elif 0x0f == opcode or 0x04 == opcode:  # end (and else)
                block = self.sigstack.pop()
                t = block.type
                print("end of %s" % sig_repr(block))
                local_cnt = len(block.locals)

                # Get and validate return value if there is one
                res = None
                if len(self.stack) >= len(t.results):
                    if len(t.results) == 1:
                        res = self.stack.pop()
                        assert isinstance(res, t.results[0])
                    elif len(t.results) > 1:
                        raise Exception("multiple return values unimplemented")
                else:
                    raise Exception("stack underflow")

                # Restore local stack
                for i in range(len(t.params)+local_cnt):
                    self.localstack.pop()

                if isinstance(block, Function):
                    # Handle return value and return address
                    return_addr = self.returnstack.pop()
                    if len(self.returnstack) == 0:
                        # Return to top-level, ignoring return_addr
                        return res
                    else:
                        print("  Returning from function %d to %d" % (
                            block.index, return_addr))
                        # Return to return address
                        self.rdr.pos = return_addr
                        # Push return value if there is one
                        if res:
                            self.stack.append(res)
                else:
                    pass # end of block/loop/if/else, keep going


            elif 0x10 == opcode:  # i32.const
                self.stack.append(I32(int(self.rdr.read_LEB(32, signed=True))))
                print("i32.const: %s" % value_repr(self.stack[-1]))
            elif 0x11 == opcode:  # i64.const
                self.stack.append(I64(int(self.rdr.read_LEB(64, signed=True))))
                print("i64.const: %s" % value_repr(self.stack[-1]))
            elif 0x12 == opcode:  # f64.const
                self.stack.append(F64(self.rdr.read_F64()))
                print("f64.const: %s" % value_repr(self.stack[-1]))
            elif 0x13 == opcode:  # f32.const
                self.stack.append(F32(self.rdr.read_F32()))
                print("f32.const: %s" % value_repr(self.stack[-1]))
            elif 0x14 == opcode:  # get_local
                arg = self.rdr.read_LEB(32)
                print("get_local: 0x%x" % arg)
                self.stack.append(self.localstack[-1-arg])
            elif 0x15 == opcode:  # set_local
                arg = self.rdr.read_LEB(32)
                val = self.stack.pop()
                self.localstack[-1-arg] = val
                print("set_local 0x%x to %s" % (arg, value_repr(val)))
            elif 0x19 == opcode:  # tee_local
                arg = self.rdr.read_LEB(32)
                val = self.stack[-1] # like set_local but do not pop
                self.localstack[-1-arg] = val
                print("tee_local 0x%x to %s" % (arg, value_repr(val)))
            elif 0xbb == opcode:  # get_global
                raise Exception("get_global unimplemented")
            elif 0xbc == opcode:  # set_global
                raise Exception("set_global unimplemented")
            elif 0x16 == opcode:  # call
                fidx = self.rdr.read_LEB(32)
                func = self.function[fidx]
                t = func.type
                args = []
                arg_cnt = len(t.params)
                res_cnt = len(t.results)

                # make args match
                for idx, PType in enumerate(t.params):
                    #assert issubclass(PType, NumericValueType)
                    arg = self.stack.pop()
                    if PType.TYPE_NAME != arg.TYPE_NAME:
                        raise Exception("call signature mismatch")
                    args.append(arg)

                if isinstance(func, FunctionImport):
                    print("call: %s.%s(%s)" % (
                        func.module, func.field,
                        ",".join([a.TYPE_NAME for a in args])))
                    results = self.host_import_func(
                            func.module, func.field, args)

                    for idx, RType in enumerate(t.results):
                        res = results[idx]
                        if RType.TYPE_NAME != res.TYPE_NAME:
                            raise Exception("return signature mismatch")
                        self.stack.append(res)
                elif isinstance(func, Function):
                    print("calling function fidx: %d" % fidx)
                    self.call_setup(fidx, args)
            elif 0x17 == opcode:  # call_indirect
                raise Exception("call_indirect unimplemented")

            # Memory immediates
            elif 0x20 <= opcode <= 0x36:
                raise Exception("memory immediates unimplemented")

            # Other Memory
            elif 0x3b == opcode:  # current_memory
                raise Exception("current_memory unimplemented")
            elif 0x39 == opcode:  # grow_memory
                raise Exception("grow_memory unimplemented")

            # Simple operations

#            # i32 operations
#            elif 0x40 <= opcode <= 0x5a or opcode in [0xb6, 0xb7]:
#                b, a = self.stack.pop(), self.stack.pop()
#                # TODO: add/use operation types, combine all ops
#                # sections
#                assert isinstance(a, I32)
#                assert isinstance(b, I32)
#                res = OPERATIONS_2[opcode][1](a,b)
#                print("i32 op: %s(%s, %s) = %s" % (
#                    OPERATIONS_2[opcode][0], value_repr(a),
#                    value_repr(b), value_repr(res)))
#                self.stack.append(res)

            # i32 operations
            elif 0x40 == opcode: # i32.add
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I32) and isinstance(b, I32)
                res = I32(int(a.val + b.val))
                print("i32.add(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x41 == opcode: # i32.sub
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I32) and isinstance(b, I32)
                res = I32(int(a.val - b.val))
                print("i32.sub(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x42 == opcode: # i32.mul
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I32) and isinstance(b, I32)
                res = I32(int(a.val * b.val))
                print("i32.mul(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x4d == opcode: # i32.eq
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I32) and isinstance(b, I32)
                res = I32(int(a.val == b.val))
                print("i32.eq(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x4e == opcode: # i32.ne
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I32) and isinstance(b, I32)
                res = I32(int(a.val != b.val))
                print("i32.ne(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x4f == opcode: # i32.lt_s
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I32) and isinstance(b, I32)
                res = I32(int(a.val < b.val))
                print("i32.lt_s(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)

#            # i64 operations
#            elif 0x5b <= opcode <= 0x74 or opcode in [0xb8, 0xb9, 0xba]:
#                b, a = self.stack.pop(), self.stack.pop()
#                # TODO: add/use operation types, combine all ops
#                # sections
#                assert isinstance(a, I64)
#                assert isinstance(b, I64)
#                res = OPERATIONS_2[opcode][1](a,b)
#                print("i64 op: %s(%s, %s) = %s" % (
#                    OPERATIONS_2[opcode][0], value_repr(a),
#                    value_repr(b), value_repr(res)))
#                self.stack.append(res)

            # i64 operations
            elif 0x5b == opcode: # i64.add
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I64) and isinstance(b, I64)
                res = I64(int(a.val + b.val))
                print("i64.add(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x5c == opcode: # i64.sub
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I64) and isinstance(b, I64)
                res = I64(int(a.val - b.val))
                print("i64.sub(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x5d == opcode: # i64.mul
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I64) and isinstance(b, I64)
                res = I64(int(a.val * b.val))
                print("i64.mul(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x5e == opcode: # i64.div_s
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I64) and isinstance(b, I64)
                res = I64(int(a.val / b.val))
                print("i64.div_s(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)
            elif 0x6e == opcode: # i64.gt_s
                b, a = self.stack.pop(), self.stack.pop()
                assert isinstance(a, I64) and isinstance(b, I64)
                res = I32(int(a.val > b.val))
                print("i64.div_s(%s, %s) = %s" % (
                    value_repr(a), value_repr(b), value_repr(res)))
                self.stack.append(res)

            # f32 operations
            elif 0x75 <= opcode <= 0x88:
                raise Exception("f32 ops unimplemented")

            # f64 operations
            elif 0x89 <= opcode <= 0x9c:
                raise Exception("f64 ops unimplemented")

#            # conversion operations
#            elif 0x9d <= opcode <= 0xb5:
#                a = self.stack.pop()
#                #assert isinstance(a, I32)
#                res = OPERATIONS_1[opcode][1](a)
#                print("conv op: %s(%s) = %s" % (
#                    OPERATIONS_1[opcode][0], value_repr(a),
#                    value_repr(res)))
#                self.stack.append(res)

            # conversion operations
            elif 0xa6 == opcode: # i64.extend_s/i32
                a = self.stack.pop()
                assert isinstance(a, I32)
                res = I64(int(a.val))
                print("i64.extend_s/i32(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)
            elif 0xb0 == opcode: # f64.convert_s/i64
                a = self.stack.pop()
                assert isinstance(a, I64)
                res = F64(float(a.val))
                print("f64.convert_s/i64(%s) = %s" % (
                    value_repr(a), value_repr(res)))
                self.stack.append(res)

            else:
                raise Exception("unrecognized opcode %d" % opcode)

    def do_branch(self, depth):
        assert len(self.sigstack) > 0
        block = self.sigstack.pop() # Always get at least one
        for r in range(depth+1):
            local_cnt = len(block.type.params) + len(block.locals)
            for i in range(local_cnt):
                self.localstack.pop()
            if r < depth:
                block = self.sigstack.pop()
            # TODO: return values/normal stack?
        if isinstance(block, Block):
            self.rdr.pos = block.label_addr
        else:
            #self.rdr.pos = block.label_addr
            raise Exception("br* in function unimplemented")

    def call_setup(self, fidx, args):
        func = self.function[fidx]

        # Push type onto sigstack
        t = func.type
        self.sigstack.append(func)

        # Push return address onto returnstack
        self.returnstack.append(self.rdr.pos)

        # Update the pos/instruction counter to the function
        self.rdr.pos = func.start

        print("  Calling function %d, start: 0x%x, end: 0x%x, %d locals, %d params, %d results" % (
            fidx, func.start, func.end,
            len(func.locals), len(t.params), len(t.results)))
        print("    bytes: %s" % (
            byte_code_repr(self.rdr.bytes[func.start:func.end])))

        # push locals onto localstack (dropping extras)
        idx = len(func.locals)-1
        while idx > -1:
            LType = func.locals[idx]
            if   LType.TYPE_NAME == "i32": val = I32(0)
            elif LType.TYPE_NAME == "i64": val = I64(0)
            elif LType.TYPE_NAME == "f32": val = F32(0.0)
            elif LType.TYPE_NAME == "f64": val = F64(0.0)
            else: raise Exception("invalid locals signature")
            self.localstack.append(val)
            idx -= 1

        # push args onto localstack as locals (dropping extras)
        aidx = 0
        idx = len(t.params)-1
        while idx > -1:
            val = args[aidx]
            PType = t.params[idx]
            assert PType.TYPE_NAME == val.TYPE_NAME, "Call signature mismatch"
            self.localstack.append(val)
            idx -= 1
            aidx += 1

        self.rdr.pos = func.start


    def run(self, name, args):
        # Reset stacks
        self.stack = []
        self.localstack = []
        self.returnstack = []
        self.sigstack = []

        fargs = []
        for arg in args:
            # TODO: accept other argument types
            assert isinstance(arg, str)
            fargs.append(I32(int(arg)))

        fidx = self.export_map[name].index
        self.call_setup(fidx, fargs)

        print("Running function %s (%d)" % (name, fidx))
        return self.run_code_v12()

### Imported functions

def DEBUG1(num0):
    print("DEBUG: %s" % num0)
def DEBUG2(num0, num1):
    print("DEBUG: %s %s" % (num0, num1))

def writeline(addr):
    print("writeline addr: %s" % addr)

def readline(addr, max_length):
    print("readline addr: %s, max_length: %s" % (addr,
        max_length))

    res = ''
    os.write(1, "user> ")
    while True:
        buf = os.read(0, 255)
        if not buf: return -1
        res += buf
        if res[-1] == '\n': return len(res)

#    res = ''
#    os.write(1, prompt)
#    while True:
#        buf = os.read(0, 255)
#        if not buf: raise EOFError()
#        res += buf
#        if res[-1] == '\n': return res[:-1]


def entry_point(argv):
    try:

        def call_import(module, field, args):
            fname = "%s.%s" % (module, field)
            host_args = [a.val for a in args]
            host_args.reverse()
            if fname == "core.DEBUG":
                if len(host_args) == 1:
                    DEBUG1(host_args[0])
                elif len(host_args) == 2:
                    DEBUG2(host_args[0], host_args[1])
                else:
                    raise Exception("DEBUG called with > 2 args")
                return []
            if fname == "core.writeline":
                writeline(host_args[0])
                return []
            if fname == "core.readline":
                res = readline(host_args[0], host_args[1])
                return [I32(int(res))]


        # Argument handling
        wasm = open(argv[1]).read()

        entry = "main"
        if len(argv) >= 3:
            entry = argv[2]

        args = []
        if len(argv) >= 4:
            args = argv[3:]

        #

        m = Module(wasm, call_import)
        m.read_magic()
        m.read_version()
        m.read_sections()

        m.dump()
        print("")

        # Assumption is that args are I32s
        res = m.run(entry, args)
        assert isinstance(res, NumericValueType)
        print("%s(%s) = %s" % (
            entry, ",".join(args), value_repr(res)))

    except Exception as e:
	if IS_RPYTHON:
	    llop.debug_print_traceback(lltype.Void)
            print("Exception: %s" % e)
	else:
	    print("".join(traceback.format_exception(*sys.exc_info())))
        return 1

    return 0

# _____ Define and setup target ___
def target(*args):
    return entry_point

# Just run entry_point if not RPython compilation
if not IS_RPYTHON:
    sys.exit(entry_point(sys.argv))

