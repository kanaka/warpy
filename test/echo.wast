(module $echo

  (import "env" "fputs" (func $fputs (param i32 i32)))
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
      (set_local $res
        (call $readline (i32.const 0) ;; prompt string
                        (get_global $line) ;; line buffer
                        (get_global $line_max)))
      (if (i32.ne (get_local $res) (i32.const 0))
        (then
          (call $fputs (i32.const 10) (i32.const 0)) ;; line: 
          (call $fputs (get_global $line) (i32.const 0))
          (call $fputs (i32.const 17) (i32.const 0)) ;; \n
          (br $repl_loop))))

    (i32.const 0))

  (export "main" (func $main))
)
