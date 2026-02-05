Fix factual errors in x86 codegen README.md:
- Caller-saved register table has notes swapped between r10 and r11 rows
- Paragraph text correctly says r10 is excluded but table says r11
- Peephole optimizer description says 3 phases but there are actually 7
- globals.rs file description mentions PLT but PLT is emitted from calls.rs
- Kernel code model says "movabs" but code emits "movq $symbol"
- Minor wording improvements
