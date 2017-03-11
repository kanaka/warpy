(module
  (import "core" "DEBUG" (func $DEBUG (param i32) (param i32)))
  (import "core" "writeline" (func $writeline (param i32)))
  (import "core" "readline" (func $readline (param i32) (param i32)
                                            (result i32)))

  ;; TODO: use passed in memory rather than exporting.
  ;;       this should enable dynamic linking of other
  ;;       module imports rather than textual inclusion
  ;;(import "reader" "read_str" (func $read_str (param i32)
  ;;                                            (result i32)))
  ;;(import "printer" "pr_str" (func $pr_str (param i32)
  ;;                                         (result i32)))

  ;;(import "env" "memory" (memory $mem 16))
  (memory $mem (export "memory") 16)

  ;;(data (i32.const 0) "\00\00\00\03mal")
  ;;(data (i32.const 10) "\00\00\00\05Line:")

  (func $read_str (param $str i32) (result i32)
    (get_local $str))

  (func $READ (param $str i32) (result i32)
    (call $read_str (get_local $str)))

  (func $EVAL (param $ast i32) (param $env i32) (result i32)
    (get_local $ast))

  (func $PRINT (param $ast i32) (result i32)
    (get_local $ast))

  (func $rep (param $str i32) (result i32)
    (call $PRINT
      (call $EVAL
        (call $READ (get_local $str))
        (i32.const 0))))

  (func $main (result i32)
    ;; Constant location/value definitions
    (local $line i32)
    (local $line_max i32)

    ;; Variable definitions
    (local $line_len i32)

    ;; Constant value settings
    (set_local $line (i32.const 1024))
    (set_local $line_max (i32.const 1024))

    ;; Start
    ;;(call $writeline (i32.const 0))

    (loop $repl_loop
      (set_local $line_len
        (call $readline (get_local $line)
                        (get_local $line_max)))
      (if (i32.ne (get_local $line_len)
                  (i32.const -1))
        (then
          ;;(call $writeline (i32.const 10))
          (call $writeline (call $rep (get_local $line)))
          (br $repl_loop))))

    (i32.const 0))

  (export "main" (func $main)))
