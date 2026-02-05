Fix common/README.md: correct factual errors found during code review

Issues to fix:
1. long_double.rs x87 arithmetic described as using inline asm — actually pure software
2. x87_cmp/x87_rem described as using fucompp/fprem asm — actually software implementations
3. Fallback "lossy f64 approximation" claim is wrong — all implementations are full-precision software
4. CType variant breakdown incorrect (says "20 primitive scalar" but there are 17)
5. IrType section claims "int and long on LP64 both become I64" but int is always I32
