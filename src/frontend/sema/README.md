# Semantic Analysis

Performs semantic checks on the parsed AST before IR lowering.

## Files

- `sema.rs` - Main semantic analysis pass. Resolves typedefs, validates type specifiers, checks declarations, and normalizes the AST for lowering.
- `builtins.rs` - Builtin function signature definitions (e.g., `__builtin_va_start`, `__builtin_memcpy`). Provides type information for compiler builtins during semantic analysis.
