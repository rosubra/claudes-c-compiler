# RISC-V 64 Backend

Code generation targeting the RISC-V 64-bit (RV64GC) architecture.

## Structure

- `codegen/` - Code generation implementation
  - `codegen.rs` - Main `RiscvCodegen` struct implementing the `ArchCodegen` trait. Handles instruction selection, stack frame management (s0 frame pointer), calling convention (a0-a7 for integer args, fa0-fa7 for FP), and varargs.
  - `inline_asm.rs` - RISC-V inline assembly constraint classification and template substitution (%0, %[name], %lo/%hi operands).
  - `atomics.rs` - Sub-word atomic RMW/CAS via LR.W/SC.W loops with bit masking, and software implementations of CLZ, CTZ, BSWAP, POPCOUNT builtins.
