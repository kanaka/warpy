#!/usr/bin/env python

from __future__ import print_function
import os, sys, re, subprocess, tempfile
from subprocess import Popen, PIPE

CLEANUP = False

WAST2WASM = os.environ.get("WAST2WASM", "wast2wasm")
WARPY = os.environ.get("WARPY", "./warpy.py")

# regex patterns of tests to skip
SKIP_TESTS = (
              # names.wast
              'invoke \"~!',
              # conversions.wast
              '18446742974197923840.0',
              '18446744073709549568.0',
              '9223372036854775808',
              'reinterpret_f.*nan',
              # endianness
              '.const 0x1.fff'
             )

def read_forms(string):
    forms = []
    form = ""
    depth = 0
    line = 0
    pos = 0
    while pos < len(string):
        # Keep track of line number
        if string[pos] == '\n': line += 1

        # Handle top-level elements
        if depth == 0:
            # Add top-level comments
            if string[pos:pos+2] == ";;":
                end = string.find("\n", pos)
                if end == -1: end == len(string)
                forms.append(string[pos:end])
                pos = end
                continue

            # TODO: handle nested multi-line comments
            if string[pos:pos+2] == "(;":
                # Skip multi-line comment
                end = string.find(";)", pos)
                if end == -1:
                    raise Exception("mismatch multiline comment on line %d: '%s'" % (
                        line, string[pos:pos+80]))
                pos = end+2
                continue

            # Ignore whitespace between top-level forms
            if string[pos] in (' ', '\n', '\t'):
                pos += 1
                continue

        # Read a top-level form
        if string[pos] == '(': depth += 1
        if string[pos] == ')': depth -= 1
        if depth == 0 and not form:
            raise Exception("garbage on line %d: '%s'" % (
                line, string[pos:pos+80]))
        form += string[pos]
        if depth == 0 and form:
            forms.append(form)
            form = ""
        pos += 1
    return forms

def parse_const(val):
    if   val == '':
        return (None, '')
    type = val[0:3]
    if type in ["i32", "i64"]:
        if val[10:12] == "0x":
            return (int(val[10:], 16),
                    "%s:%s" % (val[10:].lower(), type))
        else:
            return (int(val[10:]),
                    "%s:%s" % (hex(int(val[10:])), type))
    elif type in ["f32", "f64"]:
        if val.find("nan:") >= 0:
            # TODO: how to handle this correctly
            return (float.fromhex(val[10:].split(':')[1]),
                    "%s:%s" % (val[10:].split(':')[0], type))
        elif val[10:12] == "0x" or val[10:13] == "-0x":
            return (float.fromhex(val[10:]),
                    "%.7g:%s" % (float.fromhex(val[10:]), type))
        else:
            return (float(val[10:]),
                    "%.7g:%s" % (float(val[10:]), type))
    else:
        raise Exception("invalid value '%s'" % val)

def int2uint32(i):
    return i & 0xffffffff

def int2int32(i):
    val = i & 0xffffffff
    if val & 0x80000000:
        return val - 0x100000000
    else:
        return val

def int2uint64(i):
    return i & 0xffffffffffffffff

def int2int64(i):
    val = i & 0xffffffffffffffff
    if val & 0x8000000000000000:
        return val - 0x10000000000000000
    else:
        return val


def num_repr(i):
    if isinstance(i, int) or isinstance(i, long):
        return re.sub("L$", "", hex(i))
    else:
        return "%.16g" % i

def hexpad16(i):
    return "0x%04x" % i

def hexpad24(i):
    return "0x%06x" % i

def hexpad32(i):
    return "0x%08x" % i

def hexpad64(i):
    return "0x%016x" % i

def invoke(wasm, func, args, returncode=0):
    cmd = [WARPY, wasm, func] + args
    #print("Running: %s" % " ".join(cmd))

    sp = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (out, err) = sp.communicate()
    if sp.returncode != returncode:
        raise Exception("Failed (retcode expected: %d, got: %d)\n%s" % (
            returncode, sp.returncode, err))
    return out, err

def test_assert(mode, wasm, func, args, expected, returncode=0):
    print("Testing(%s) %s(%s) = %s" % (
        mode, func, ", ".join(args), expected))


    expects = set([expected])
    m0 = re.search("^(-?[0-9\.e-]+):f32$", expected)
    if m0:
        if m0.group(1) == "-0":
            expects.add("0:f32")
        expects.add('%f:f32' % float(m0.group(1)))
        expects.add('%f:f32' % round(float(m0.group(1)),5))
    if expected == "-nan:f32":
        expects.add("nan:f32")
    if expected == "-nan:f64":
        expects.add("nan:f64")

    out, err = invoke(wasm, func, args, returncode)

    # munge the output some
    out = out.rstrip("\n")
    out = re.sub("L:i32$", ':i32', out)
    out = re.sub("L:i64$", ':i64', out)
    results = set([out])
    # create alternate representations
    m1 = re.search("^(-?[0-9a-fx]+):i32$", out)
    m2 = re.search("^(-?[0-9a-fx]+):i64$", out)
    m3 = re.search("^(-?[0-9\.e-]+):f32$", out)
    m4 = re.search("^(-?0x[0-9a-fp+\.]+):f32$", out)
    m5 = re.search("^(-?[0-9\.e-]+):f64$", out)
    m6 = re.search("^(-?0x[0-9a-fp+\.]+):f64$", out)
    if m1:
        val = int(m1.group(1), 16)
        results.add(num_repr(int2int32(val))+":i32")
        results.add(num_repr(int2uint32(val))+":i32")
        results.add(hexpad16(int2uint32(val))+":i32")
        results.add(hexpad24(int2uint32(val))+":i32")
        results.add(hexpad32(int2uint32(val))+":i32")
    elif m2:
        val = int(m2.group(1), 16)
        results.add(num_repr(int2int64(val))+":i64")
        results.add(num_repr(int2uint64(val))+":i64")
        results.add(hexpad32(int2uint64(val))+":i64")
        results.add(hexpad64(int2uint64(val))+":i64")
    elif m3:
        val = float(m3.group(1))
        if re.search("^.*\.0+$", m3.group(1)):
            # Zero
            results.add('%d:f32' % int(val))
            results.add('%.7g:f32' % val)
        else:
            results.add('%.7g:f32' % val)
    elif m4:
        val = float.fromhex(m4.group(1))
        results.add("%f:f32" % val)
        results.add("%.7g:f32" % val)
    elif m5:
        val = float(m5.group(1))
        if re.search("^.*\.0+$", m5.group(1)):
            # Zero
            results.add('%d:f64' % int(val))
            results.add('%.7g:f64' % val)
        else:
            results.add('%.7g:f64' % val)
    elif m6:
        val = float.fromhex(m6.group(1))
        results.add("%f:f64" % val)
        results.add("%.7g:f64" % val)

    #print("  out: '%s'" % out)
    #print("  err: %s" % err)
    #print("expects: '%s', results: %s, returncode: %d" % (
    #    expects, results, returncode))
    if (expected.find("unreachable") > -1
            and err.find("unreachable") > -1):
        pass
    elif (expected.find("call signature mismatch") > -1
            and err.find("call signature mismatch") > -1):
        pass
    elif returncode==1 and err.find(expected) > -1:
        pass
    elif not expects.intersection(results):
        if returncode==1:
            raise Exception("Failed:\n  expected: '%s'\n  got: '%s'" % (
                expected, err))
        else:
            raise Exception("Failed:\n  expected: '%s' %s\n  got: '%s' %s" % (
                expected, expects, out, results))

def test_assert_return(wasm, form):
    # params, return
    m = re.search('^\(assert_return\s+\(invoke\s+"((?:[^"]|\\\")*)"\s+(\(.*\))\s*\)\s*(\([^)]+\))\s*\)\s*$', form, re.S)
    if not m:
        # no params, return
        m = re.search('^\(assert_return\s+\(invoke\s+"((?:[^"]|\\\")*)"\s*\)\s+()(\([^)]+\))\s*\)\s*$', form, re.S)
    if not m:
        # params, no return
        m = re.search('^\(assert_return\s+\(invoke\s+"([^"]*)"\s+(\(.*\))()\s*\)\s*\)\s*$', form, re.S)
    if not m:
        # no params, no return
        m = re.search('^\(assert_return\s+\(invoke\s+"([^"]*)"\s*()()\)\s*\)\s*$', form, re.S)
    if not m:
        raise Exception("unparsed assert_return: '%s'" % form)
    func = m.group(1)
    if m.group(2) == '':
        args = []
    else:
        args = [v.split(' ')[1] for v in re.split("\)\s*\(", m.group(2)[1:-1])]
    result, expected = parse_const(m.group(3)[1:-1])

    test_assert("return", wasm, func, args, expected)

def test_assert_trap(wasm, form):
    # params
    m = re.search('^\(assert_trap\s+\(invoke\s+"([^"]*)"\s+(\(.*\))\s*\)\s*"([^"]+)"\s*\)\s*$', form)
    if not m:
        # no params
        m = re.search('^\(assert_trap\s+\(invoke\s+"([^"]*)"\s*()\)\s*"([^"]+)"\s*\)\s*$', form)
    if not m:
        raise Exception("unparsed assert_trap: '%s'" % form)
    func = m.group(1)
    if m.group(2) == '':
        args = []
    else:
        args = [v.split(' ')[1] for v in re.split("\)\s*\(", m.group(2)[1:-1])]
    expected = m.group(3)

    test_assert("trap", wasm, func, args, expected, returncode=1)

def do_invoke(wasm, form):
    # params
    m = re.search('^\(invoke\s+"([^"]+)"\s+(\(.*\))\s*\)\s*$', form)
    if not m:
        # no params
        m = re.search('^\(invoke\s+"([^"]+)"\s*()\)\s*$', form)
    if not m:
        raise Exception("unparsed invoke: '%s'" % form)
    func = m.group(1)
    if m.group(2) == '':
        args = []
    else:
        args = [v.split(' ')[1] for v in re.split("\)\s*\(", m.group(2)[1:-1])]

    print("Invoking %s(%s)" % (
        func, ", ".join([str(a) for a in args])))

    invoke(wasm, func, args)

def skip_test(form):
    for s in SKIP_TESTS:
        if re.search(s, form):
            return True
    return False

def run_test_file(test_file):
    print("WAST2WASM: '%s'" % WAST2WASM)
    (t1fd, wast_tempfile) = tempfile.mkstemp(suffix=".wast")
    (t2fd, wasm_tempfile) = tempfile.mkstemp(suffix=".wasm")
    print("wast_tempfile: '%s'" % wast_tempfile)
    print("wasm_tempfile: '%s'" % wasm_tempfile)

    try:
        forms = read_forms(file(test_file).read())

        for form in forms:
            if  ";;" == form[0:2]:
                print(form)
            elif re.match("^\(module\\b.*", form):
                print("Writing WAST module to '%s'" % wast_tempfile)
                file(wast_tempfile, 'w').write(form)
                print("Compiling WASM to '%s'" % wasm_tempfile)
                subprocess.check_call([
                    WAST2WASM,
                    #"--no-check-assert-invalid-and-malformed",
                    "--no-check",
                    wast_tempfile,
                    "-o",
                    wasm_tempfile])
            elif skip_test(form):
                print("Skipping test: %s" % form[0:60])
            elif re.match("^\(assert_return\\b.*", form):
                #print("%s" % form)
                test_assert_return(wasm_tempfile, form)
            elif re.match("^\(assert_trap\\b.*", form):
                #print("%s" % form)
                test_assert_trap(wasm_tempfile, form)
            elif re.match("^\(invoke\\b.*", form):
                do_invoke(wasm_tempfile, form)
            elif re.match("^\(assert_invalid\\b.*", form):
                #print("ignoring assert_invalid")
                pass
            elif re.match("^\(assert_exhaustion\\b.*", form):
                print("ignoring assert_exhaustion")
                pass
            elif re.match("^\(assert_unlinkable\\b.*", form):
                print("ignoring assert_unlinkable")
                pass
            else:
                raise Exception("unrecognized form '%s...'" % form[0:40])
    finally:
        if CLEANUP:
            print("Removing tempfiles")
            os.remove(wast_tempfile)
            os.remove(wasm_tempfile)
        else:
            print("Leaving tempfiles: %s" % (
                [wast_tempfile, wasm_tempfile]))

if __name__ == "__main__":
    run_test_file(sys.argv[1])
