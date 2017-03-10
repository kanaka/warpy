#!/usr/bin/env python

import sys, os, math
IS_RPYTHON = sys.argv[0].endswith('rpython')

if IS_RPYTHON:
    from rpython.rtyper.lltypesystem import lltype
    from rpython.rtyper.lltypesystem.lloperation import llop
else:
    sys.path.append(os.path.abspath('./pypy2-v5.6.0-src'))
    import traceback



MAGIC = 0x6d736100
VERSION = 0xc

# https://github.com/WebAssembly/design/blob/453320eb21f5e7476fb27db7874c45aa855927b7/BinaryEncoding.md#function-bodies

class ValueType():
    pass

class NumericValueType(ValueType):
    pass

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


VALUE_TYPE = { 0x01 : I32,
               0x02 : I64,
               0x03 : F32,
               0x04 : F64,
               0x10 : AnyFunc,
               0x20 : Func,
               0x40 : EmptyBlockType }

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

OPERATIONS = { 0x40: ("i32.add", lambda a,b: I32(a.val+b.val)),
               0x41: ("i32.sub", lambda a,b: I32(a.val-b.val)),
               0x42: ("i32.mul", lambda a,b: I32(a.val*b.val)) }



class Type():
    def __init__(self, form, params, results):
        self.form = form
        self.params = params
        self.results = results

class Function():
    def __init__(self, type):
        self.type = type

    def update(self, locals, byte_code):
        self.locals = locals
        self.byte_code = byte_code

class FunctionImport(Function):
    def __init__(self, type, module, field):
        self.type = type
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

def bytes2word(w):
    return (w[3]<<24) + (w[2]<<16) + (w[1]<<8) + w[0]

class Reader():
    def __init__(self, bytes):
        self.bytes = bytes
        self.pos = 0

    def read_byte(self):
        b = self.bytes[self.pos]
        self.pos += 1
        return b

    def read_word(self):
        w = bytes2word(self.bytes[self.pos:self.pos+4])
        self.pos += 4
        return w

    def read_bytes(self, cnt):
        assert cnt > 0
        bytes = self.bytes[self.pos:self.pos+cnt]
        self.pos += cnt
        return bytes

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

def stack_repr(vals):
    res = []
    for val in vals:
        assert isinstance(val, NumericValueType)
        res.append("%s %s" % (val.TYPE_NAME, val.val))
    return "[" + ", ".join(res) + "]"

def callstack_repr(vals):
    res = []
    for val in vals:
        assert isinstance(val, NumericValueType)
        res.append("%s %s" % (val.TYPE_NAME, val.val))
    return "[" + ", ".join(res) + "]"

def byte_code_repr(bytes):
    res = []
    for val in bytes:
        if val < 16:
            res.append("%x" % val)
        else:
            res.append("%x" % val)
    return "[" + ",".join(res) + "]"

class Module():
    def __init__(self, data, host_import_func):
        assert isinstance(data, str)
        self.data = data
        self.rdr = Reader([ord(b) for b in data])
        self.host_import_func = host_import_func

        # Sections
        self.type = []
        self.import_list = []
        self.import_map = {}
        self.function = []
        self.export_list = []
        self.export_map = {}

        # Execution state
        self.stack = []
        self.callstack = []
        self.typestack = []

    def dump(self):
        #print("data: %s" % self.data)
        #print("bytes: %s" % [hex(b) for b in self.rdr.bytes])
        #print("rdr.pos: %s" % self.rdr.pos)

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
            if isinstance(f,FunctionImport):
                print("  %d [type: %d, import: '%s.%s']" % (
                    i, f.type, f.module, f.field))
            else:
                print("  %d [type: %d, locals: %s]" % (
                    i, f.type, [p.TYPE_NAME for p in f.locals]))
                print("    byte_code  : %s" % byte_code_repr(f.byte_code))

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
            self.type.append(Type(form, params, results))
        

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
                imp = Import(module, field, kind, type=sig_index)
                self.import_list.append(imp)
                func = FunctionImport(
                        sig_index, module, field)
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
            type = self.rdr.read_LEB(32)
            self.function.append(Function(type))

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
        byte_code = self.rdr.read_bytes(
                body_size - (self.rdr.pos-payload_start))
        #print("byte code: %s" % byte_code)
        self.function[idx].update(locals, byte_code)

    def parse_Code(self, length):
	body_count = self.rdr.read_LEB(32)
        import_cnt = len(self.import_list)
        for idx in range(body_count):
            self.parse_Code_body(idx + import_cnt)

    def parse_Data(self, length):
        return self.rdr.read_bytes(length)

    ###

    # TODO: update for MVP
    def run_code_v12(self):
        while not self.code_rdr.eof():
            print("    * stack: %s, callstack: %s" % (
                stack_repr(self.stack), callstack_repr(self.callstack)))
            opcode = self.code_rdr.read_byte()
            print "    [0x%x] - " % opcode,
            if   0x00 == opcode:  # unreachable
                raise Exception("Immediate trap")
            elif 0x01 == opcode:  # block
                inline_signature_type = self.code_rdr.read_LEB(32)
                print("block sig: 0x%x" % inline_signature_type)
            elif 0x02 == opcode:  # loop
                inline_signature_type = self.code_rdr.read_LEB(32)
                print("loop sig: 0x%x" % inline_signature_type)
            elif 0x03 == opcode:  # if
                inline_signature_type = self.code_rdr.read_LEB(32)
                print("if sig: 0x%x" % inline_signature_type)
            elif 0x04 == opcode:  # else
                pass
            elif 0x05 == opcode:  # select
                pass
            elif 0x06 == opcode:  # br
                relative_depth = self.code_rdr.read_LEB(32)
                print("br depth: 0x%x" % relative_depth)
            elif 0x07 == opcode:  # br_if
                relative_depth = self.code_rdr.read_LEB(32)
                print("br_if depth: 0x%x" % relative_depth)
            elif 0x08 == opcode:  # br_table
                pass
            elif 0x09 == opcode:  # return
                pass
            elif 0x0a == opcode:  # nop
                pass
            elif 0x0b == opcode:  # drop
                pass
            elif 0x0f == opcode:  # end
                print("end")
                t = self.typestack.pop()
                for i in range(len(t.params)):
                    self.callstack.pop()
                if len(self.callstack) == 0:
                    #assert len(self.stack) >= 1
                    #assert isinstance(self.stack[-1], NumericValueType)
                    if len(self.stack) > 0:
                        res = self.stack.pop()
                        assert isinstance(res, t.results[0])
                        return res
                    else:
                        res = None
                        return res
                else:
                    pass

            elif 0x10 == opcode:  # i32.const
                self.stack.append(I32(self.code_rdr.read_LEB(32, signed=True)))
                print("i32.const: 0x%x" % self.stack[-1].val)
            elif 0x11 == opcode:  # i64.const
                self.stack.append(I32(self.code_rdr.read_LEB(64, signed=True)))
                print("i64.const: 0x%x" % self.stack[-1].val)
            elif 0x12 == opcode:  # f32.const
                pass
            elif 0x13 == opcode:  # f64.const
                pass
            elif 0x14 == opcode:  # get_local
                arg = self.code_rdr.read_LEB(32)
                print("get_local: 0x%x" % arg)
                self.stack.append(self.callstack[-1-arg])
            elif 0x15 == opcode:  # set_local
                arg = self.code_rdr.read_LEB(32)
                print("set_local: 0x%x" % arg)
            elif 0x19 == opcode:  # tee_local
                pass
            elif 0xbb == opcode:  # get_global
                pass
            elif 0xbc == opcode:  # set_global
                pass
            elif 0x16 == opcode:  # call
                fidx = self.code_rdr.read_LEB(32)
                func = self.function[fidx]
                args = []
                t = self.type[func.type]
                arg_cnt = len(t.params)
                res_cnt = len(t.results)

                # make args match
                for idx, PType in enumerate(t.params):
                    #assert issubclass(PType, NumericValueType)
                    arg = self.stack.pop()
                    if PType.TYPE_NAME != arg.TYPE_NAME:
                        raise Exception("call signature mismatch")
                    args.append(arg)
                
                print("call: %s.%s(%s)" % (
                    func.module, func.field, args))
                results = self.host_import_func(
                        func.module, func.field, args)

                for idx, RType in enumerate(t.results):
                    res = results[idx]
                    if RType.TYPE_NAME != res.TYPE_NAME:
                        raise Exception("return signature mismatch")
                    self.stack.append(res)

            elif 0x17 == opcode:  # call_indirect
                pass

            # Memory immediates
            elif 0x20 <= opcode <= 0x36:
                pass

            # Other Memory
            elif 0x3b == opcode:  # current_memory
                pass
            elif 0x39 == opcode:  # grow_memory
                pass
            
            # Simple operations

            # i32 operations
            elif 0x40 <= opcode <= 0x5a or opcode in [0xb6, 0xb7]:
#                assert len(self.stack) >= 2
                a = self.stack.pop()
                b = self.stack.pop()
                # TODO: add operation types
                assert isinstance(a, NumericValueType)
                assert isinstance(b, NumericValueType)
                res = OPERATIONS[opcode][1](a,b)
                print("i32 op: %s(0x%x, 0x%x) = 0x%x" % (
                    OPERATIONS[opcode][0], a.val, b.val, res.val))
                self.stack.append(res)

            # i64 operations
            elif 0x5b <= opcode <= 0x74 or opcode in [0xb8, 0xb9, 0xba]:
                pass

            # f32 operations
            elif 0x75 <= opcode <= 0x88:
                pass

            # f64 operations
            elif 0x89 <= opcode <= 0x9c:
                pass

            # conversion operations
            elif 0x9d <= opcode <= 0xb5:
                pass
        

    def run(self, name, args):
        self.stack = []
        self.callstack = []
        self.typestack = []

        idx = self.export_map[name].index
        func = self.function[idx]
        self.code_rdr = Reader(func.byte_code)

        # Push type onto typestack and args onto callstack (drop extras)
        t = self.type[func.type]
        self.typestack.append(t)
        for idx, PType in enumerate(t.params):
            #assert issubclass(PType, NumericValueType)
            arg = args[idx]
            if   PType.TYPE_NAME == "i32": val = I32(arg)
            elif PType.TYPE_NAME == "i64": val = I64(arg)
            elif PType.TYPE_NAME == "f32": val = I64(arg)
            elif PType.TYPE_NAME == "f64": val = I64(arg)
            else: raise Exception("invalid function signature")
            self.callstack.append(val)

        print ("Running function %s (%d), code: %s" % (
            name, idx, [hex(b) for b in self.code_rdr.bytes]))
        return self.run_code_v12()

### Imported functions

def DEBUG1(num0):
    print("DEBUG: %s" % num0)
def DEBUG2(num0, num1):
    print("DEBUG: %s %s" % (num0, num1))

def writeline(addr):
    print("writeline addr: %d" % addr)

def readline(addr, max_length):
    print("readline addr: %d, max_length: %d" % (addr,
        max_length))
    return 0

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
                return [I32(res)]


        wasm = open(argv[1]).read()
        m = Module(wasm, call_import)

        m.read_magic()
        m.read_version()
        m.read_sections()

        m.dump()


        print("")

        #res = m.run("addTwo", [7, 8])
        #assert isinstance(res, NumericValueType)
        #print("addTwo(7,8) = %s" % res.val)

        m.run("debugTwo", [7, 8])

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

