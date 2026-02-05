Task: Fix RISC-V assembler README (code/src/backend/riscv/assembler/README.md)

The README has drifted significantly from the actual code. Fixing:
- Wrong data structure names (ParsedLine vs AsmStatement, EncodeResult variants)
- Stale line counts
- Missing Operand variants and ElfWriter fields
- Missing relocation types (TlsGdHi20, TlsGotHi20, Add32/64, Sub32/64)
- False claims about unimplemented features (.macro, .if/.endif, .irp are all implemented)
- Wrong compression ordering claim
- Compression is actually disabled (commented out)
