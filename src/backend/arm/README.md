# AArch64 Backend

Code generation targeting the AArch64 (ARM64) architecture with AAPCS64 calling convention.

## Structure

- `codegen/` - Code generation implementation
  - `codegen.rs` - Main `ArmCodegen` struct implementing the `ArchCodegen` trait. Handles instruction selection, stack frame management, calling convention (x0-x7 for integer args, d0-d7 for FP), atomics (LDXR/STXR exclusive access), and varargs.
  - `inline_asm.rs` - AArch64 inline assembly template substitution, register formatting (w/x/s/d modifiers), and exclusive load/store instruction helpers.
