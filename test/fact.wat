(module
  (memory $0 1)
  (export "memory" (memory $0))
  (export "fact" (func $fact))
  (func $fact (param $0 i32) (result f64)
    (local $1 i64)
    (local $2 i64)
    (block $label$0
      (br_if $label$0
        (i32.lt_s
          (local.get $0)
          (i32.const 1)
        )
      )
      (local.set $1
        (i64.add
          (i64.extend_i32_s
            (local.get $0)
          )
          (i64.const 1)
        )
      )
      (local.set $2
        (i64.const 1)
      )
      (loop $label$1
        (local.set $2
          (i64.mul
            (local.get $2)
            (local.tee $1
              (i64.add
                (local.get $1)
                (i64.const -1)
              )
            )
          )
        )
        (br_if $label$1
          (i64.gt_s
            (local.get $1)
            (i64.const 1)
          )
        )
      )
      (return
        (f64.convert_i64_s
          (local.get $2)
        )
      )
    )
    (f64.const 1)
  )
)

