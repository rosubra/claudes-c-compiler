/* CCC compiler bundled xmmintrin.h - SSE intrinsics */
#ifndef _XMMINTRIN_H_INCLUDED
#define _XMMINTRIN_H_INCLUDED

typedef struct __attribute__((__aligned__(16))) {
    float __val[4];
} __m128;

static __inline__ void __attribute__((__always_inline__))
_mm_sfence(void)
{
    __builtin_ia32_sfence();
}

static __inline__ void __attribute__((__always_inline__))
_mm_pause(void)
{
    __builtin_ia32_pause();
}

#endif /* _XMMINTRIN_H_INCLUDED */
