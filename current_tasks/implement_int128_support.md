# Task: Implement __int128 / __uint128_t type support

## Goal
Add support for 128-bit integer types needed by mbedtls bignum library.

## What's needed
1. `__int128`, `__uint128_t`, `__int128_t` as type specifiers
2. `__attribute__((mode(TI)))` to transform types to 128-bit
3. IR types I128/U128
4. 128-bit arithmetic (at minimum: add, sub, mul, shift, or, and, cast)
5. Backend codegen for 128-bit values as register pairs

## Why
mbedtls rsa/ecp tests segfault because `mbedtls_t_udbl` (defined via mode(TI))
is treated as 4-byte int instead of 16-byte, breaking bignum division.
