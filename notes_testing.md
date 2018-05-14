BASIC_TESTS="block if br br_if br_table return loop select labels switch unwind stack get_local set_local tee_local unreachable func call call_indirect address memory left-to-right int_literals int_exprs i32 i64 endianness"

# {f32,f64}_cmp take a long time
OTHER_TESTS="break-drop forward func_ptrs fac unreached-invalid traps f32_cmp f64_cmp conversions"

C_TESTS="nop memory_redundancy memory_trap start globals f32 f64 float_literals float_misc float_exprs float_memory"

SKIP_TESTS="store_retval binary custom_section exports linking imports typecheck skip-stack-guard-page"

export WA_CMD=./warpy.py WAST2WASM=./wabt/out/gcc/Release/wast2wasm 
time for t in ${BASIC_TESTS} ${OTHER_TESTS}; do
    echo "TESTING ${t}"
    ./runtest.py ./wabt/third_party/testsuite/${t}.wast || break
    echo
done

No actual tests:
- 1.6K  store_retval.wast
-  968  binary.wast
*  270  break-drop.wast
*  643  forward.wast
X  652  comments.wast      (nested comments)
X 1.4K  memory_redundancy.wast  (multiple invokes need same context)
X 1.5K  memory_trap.wast   (multiple invokes need same context)
* 1.6K  address.wast
X 1.8K  start.wast         (multiple invokes need same context)
* 2.5K  fac.wast
X 2.6K  resizing.wast      (doesn't throw out of bound exception)
- 2.7K  custom_section.wast
* 2.8K  names.wast         (skip symbols name)
* 3.0K  stack.wast
* 3.1K  int_literals.wast
* 3.8K  func_ptrs.wast
* 4.0K  select.wast
* 4.1K  get_local.wast
* 4.3K  traps.wast
X 4.4K  globals.wast       (multiple invokes need same context)
* 4.7K  switch.wast
* 5.5K  set_local.wast
- 6.2K  exports.wast       (testing compiler/textual format)
X 6.7K  float_memory.wast  (multiple invokes need same context)
* 6.9K  block.wast
* 6.9K  tee_local.wast
* 7.0K  call.wast
* 7.3K  unwind.wast
* 8.3K  labels.wast
* 8.8K  br_if.wast
* 8.9K  unreachable.wast
* 9.0K  loop.wast
- 9.1K  linking.wast
X 9.3K  nop.wast           (multiple invokes need same context)
* 9.3K  return.wast
* 9.9K  endianness.wast
X  10K  float_literals.wast (float mismatch)
*  12K  memory.wast
*  12K  call_indirect.wast
*  12K  br.wast
*  13K  if.wast
-  14K  imports.wast
*  14K  int_exprs.wast
*  15K  func.wast
*  15K  unreached-invalid.wast  (all ignored)
*  19K  typecheck.wast     (all ignored)
*  21K  left-to-right.wast
*  31K  i32.wast
*  33K  i64.wast
*  36K  conversions.wast   (skipping several nan float tests)
X  56K  float_misc.wast    (lots of float mismatches)
*  79K  br_table.wast
X 138K  float_exprs.wast   (float mismatch)
* 158K  skip-stack-guard-page.wast  (just assert_exhaustion)
* 167K  f32_cmp.wast
* 180K  f64_cmp.wast
X 222K  f32.wast           (float mismatch)
X 244K  f64.wast           (float mismatch)
  

