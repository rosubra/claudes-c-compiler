# Fix fn_ptr()->member always accessing offset 0 for indirect calls

## Summary

Accessing struct members through function pointer call results (e.g.,
`fn_ptr()->field`) always read offset 0 regardless of which member was
requested. This caused Redis's `pubsub.c` to crash with a segfault in
`pubsubUnsubscribeAllChannelsInternal` where `pubsubtype` function pointer
members were called and the result struct members accessed at wrong offsets.

## Root Cause

Two functions in `src/ir/lowering/structs.rs` resolve struct layouts for
member access expressions:

1. `get_pointed_struct_layout()` — for `expr->member` (pointer member access)
2. `get_layout_for_expr()` — for `expr.member` (direct member access)

Both functions' `FunctionCall` arms only handled direct function calls by
looking up the function name in `func_meta.sigs`. For indirect calls through
function pointers (where the callee is a variable, not a named function),
the lookup failed and fell through to return `None`, which the callers
treated as offset 0 with type `IrType::I32`.

This meant `fn_ptr()->a`, `fn_ptr()->b`, `fn_ptr()->c` all generated code
to access offset 0, returning the first member's value for every access.

## Fix

Added a fallback in both functions' `FunctionCall` arms: when the direct
`func_meta.sigs` lookup fails (indicating an indirect call), use
`get_expr_ctype()` to resolve the return type from the function pointer's
CType, then derive the struct layout from that type.

`get_expr_ctype()` already correctly handles indirect function pointer calls
by stripping `Deref` layers, looking up the variable's CType, and using
`extract_func_ptr_return_ctype()`.

## Files Changed

- `src/ir/lowering/structs.rs` — added `get_expr_ctype()` fallback in both
  `get_pointed_struct_layout()` and `get_layout_for_expr()` FunctionCall arms
- `tests/fnptr-call-member-offset/` — new regression test covering:
  - Direct member access through function pointer (`fn()->a`, `fn()->b`, `fn()->c`)
  - Array member access at non-zero offset through function pointer (`fn()->val[0]`, `fn()->val[1]`)
  - Function pointer inside a struct calling and accessing result members

## Test Results

- Before: 220/222 tests passing (99.1%)
- After: 220/222 tests passing (99.1%) — no regressions
- New regression test `fnptr-call-member-offset` added and passing
- The 2 pre-existing failures (`va-arg-fp-overflow`, `wl-linker-flags-passthrough`) are unrelated
- Redis pubsub.c crash is fixed; Redis verification improves from 0/4 to 2/4 tests passing
  (remaining 2 failures are the separate bind EFAULT bug tracked in `fix_postgres_bind_efault`)

## Commit

`109d3a8` Fix fn_ptr()->member always accessing offset 0 for indirect calls
