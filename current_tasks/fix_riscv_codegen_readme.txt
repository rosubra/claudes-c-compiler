Fix RISC-V codegen README: correct factual errors and improve accuracy

Issues found:
1. Inline asm scratch register count says "13" but actual RISCV_GP_SCRATCH has 15 (includes a0, a1)
2. Register allocation table describes s2-s6 as "Conditionally available" but they are always
   added to the allocation pool (the only "condition" is inline asm clobbering, which applies
   to all registers equally)
3. Minor readability improvements
