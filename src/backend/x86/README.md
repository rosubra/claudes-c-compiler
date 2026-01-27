# x86-64 Backend

Code generation targeting the x86-64 (AMD64) architecture with System V ABI.

## Structure

- `codegen/` - Code generation implementation
  - `codegen.rs` - Main `X86Codegen` struct implementing the `ArchCodegen` trait. Handles instruction selection, linear scan register allocation (callee-saved: rbx, r12-r15), calling convention, atomics, and varargs.
  - `asm_emitter.rs` - `InlineAsmEmitter` trait implementation: constraint classification (multi-alternative, x87, condition codes), scratch register allocation, operand loading/storing for inline asm.
  - `f128.rs` - F128 (long double) operations via x87 FPU: load/store helpers for `SlotAddr`, x87-specific cast instructions (`emit_cast_instrs_x86`), and `emit_f128_load_to_x87`.
  - `inline_asm.rs` - x86 inline assembly template substitution and register formatting (AT&T syntax with `%` operand references).
  - `peephole/` - Post-codegen peephole optimizer. Operates on assembly text to eliminate redundant patterns from stack-based codegen.
    - `types.rs` - Core data structures (LineInfo, LineKind, LineStore), line classification (`classify_line`), register utilities, and parsing helpers.
    - `passes.rs` - Optimization passes: local store/load forwarding, push/pop elimination, compare-branch fusion, dead store elimination, global store forwarding, and unused callee-save removal.
  - `register.rs` - Register name definitions and utilities.
