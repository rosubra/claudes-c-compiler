# Task: fix_asm_goto_liveness
Status: completed
Branch: master
Description: Fix asm goto (InlineAsm with goto_labels) handling across the compiler: CFG analysis, mem2reg SSA construction, phi elimination, and backend liveness analysis all now correctly include asm goto label targets as control flow edges.
