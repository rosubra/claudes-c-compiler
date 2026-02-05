Task: Fix ARM assembler README.md accuracy and completeness

The README at src/backend/arm/assembler/README.md has several factual errors:
- Line counts are out of date (parser=2193 vs actual 2342, encoder=5604 vs 5983, etc.)
- Relocation table says "18 types" but there are actually 20 (missing AdrPrelLo21)
- TlsLeAddTprelLo12 ELF number listed as 550, should be 551 (_NC variant)
- AsmDirective count says "24 kinds" but there are 27
- Various small inaccuracies to fix

Will update the README to be accurate and well-structured.
