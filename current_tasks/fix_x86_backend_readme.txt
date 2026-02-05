Fix x86 backend top-level README (src/backend/x86/README.md):
- "13-pass pipeline" is wrong, should be 15 pass functions in 7 phases
- Assembler description says "SSE/AES-NI encoding" but misses AVX/AVX2/EVEX/BMI2/VEX
- Peephole parenthetical list is incomplete (missing tail call, frame compaction)
- Linker description misses shared library (.so) output support
- Overall too terse for a backend overview; add more useful context
