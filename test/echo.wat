(module $echo

  (import "env" "printline" (func $printline (param i32)))
  (import "env" "readline" (func $readline (param i32 i32 i32) (result i32)))

  (memory $mem (export "memory") 16)

  (data (i32.const 0) "prompt> \00")
  (data (i32.const 10) "line: \00")
  (data (i32.const 17) "\n\00")

  ;; Constant value settings
  (global $line i32 (i32.const 1024))
  (global $line_max i32 (i32.const 1024))

  (func $main (result i32)
    (local $res i32)

    (loop $repl_loop
      (local.set $res
        (call $readline (i32.const 0) ;; prompt string
                        (global.get $line) ;; line buffer
                        (global.get $line_max)))
      (if (i32.ne (local.get $res) (i32.const 0))
        (then
          (call $printline (i32.const 10)) ;; line: 
          (call $printline (global.get $line))
          (call $printline (i32.const 17)) ;; \n
          (br $repl_loop))))

    (i32.const 0))

  (export "_main" (func $main))
)
