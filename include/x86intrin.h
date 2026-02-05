/* CCC compiler bundled x86intrin.h - GCC-compatible x86 intrinsics */
#ifndef _X86INTRIN_H_INCLUDED
#define _X86INTRIN_H_INCLUDED

#if !defined(__x86_64__) && !defined(__i386__) && !defined(__i686__)
#error "x86 intrinsics (x86intrin.h) require an x86 target"
#endif

/* Include the main SIMD intrinsics header */
#include <immintrin.h>

#endif /* _X86INTRIN_H_INCLUDED */
