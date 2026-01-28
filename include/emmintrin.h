/* CCC compiler bundled emmintrin.h - SSE2 intrinsics */
#ifndef _EMMINTRIN_H_INCLUDED
#define _EMMINTRIN_H_INCLUDED

#include <xmmintrin.h>

typedef struct __attribute__((__aligned__(16))) {
    long long __val[2];
} __m128i;

typedef struct __attribute__((__aligned__(1))) {
    long long __val[2];
} __m128i_u;

typedef struct __attribute__((__aligned__(16))) {
    double __val[2];
} __m128d;

typedef struct __attribute__((__aligned__(1))) {
    double __val[2];
} __m128d_u;

/* Internal vector types referenced by GCC system headers (wmmintrin.h, etc.).
 * These enable parsing of system headers that use (__v2di)expr casts.
 * Note: vector_size attribute is parsed but vectors are lowered as aggregates. */
typedef double __v2df __attribute__ ((__vector_size__ (16)));
typedef long long __v2di __attribute__ ((__vector_size__ (16)));
typedef unsigned long long __v2du __attribute__ ((__vector_size__ (16)));
typedef int __v4si __attribute__ ((__vector_size__ (16)));
typedef unsigned int __v4su __attribute__ ((__vector_size__ (16)));
typedef short __v8hi __attribute__ ((__vector_size__ (16)));
typedef unsigned short __v8hu __attribute__ ((__vector_size__ (16)));
typedef char __v16qi __attribute__ ((__vector_size__ (16)));
typedef signed char __v16qs __attribute__ ((__vector_size__ (16)));
typedef unsigned char __v16qu __attribute__ ((__vector_size__ (16)));

/* Helper to convert intrinsic result pointer to __m128i value.
 * Our SSE builtins return a pointer to 16-byte result data.
 * This macro dereferences that pointer to get the __m128i struct value. */
#define __CCC_M128I_FROM_BUILTIN(expr) (*(__m128i *)(expr))

/* === Load / Store === */

static __inline__ __m128i __attribute__((__always_inline__))
_mm_loadu_si128(__m128i_u const *__p)
{
    return *__p;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_load_si128(__m128i const *__p)
{
    return *__p;
}

static __inline__ void __attribute__((__always_inline__))
_mm_storeu_si128(__m128i_u *__p, __m128i __b)
{
    *__p = __b;
}

static __inline__ void __attribute__((__always_inline__))
_mm_store_si128(__m128i *__p, __m128i __b)
{
    *__p = __b;
}

/* === Set === */

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set1_epi8(char __b)
{
    return __CCC_M128I_FROM_BUILTIN(
        __builtin_ia32_vec_init_v16qi(__b, __b, __b, __b,
                                      __b, __b, __b, __b,
                                      __b, __b, __b, __b,
                                      __b, __b, __b, __b));
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set1_epi32(int __i)
{
    return __CCC_M128I_FROM_BUILTIN(
        __builtin_ia32_vec_init_v4si(__i, __i, __i, __i));
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_setzero_si128(void)
{
    return (__m128i){ { 0LL, 0LL } };
}

/* === Compare === */

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpeq_epi8(__m128i __a, __m128i __b)
{
    return __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pcmpeqb128(__a, __b));
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpeq_epi32(__m128i __a, __m128i __b)
{
    return __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pcmpeqd128(__a, __b));
}

/* === Arithmetic === */

static __inline__ __m128i __attribute__((__always_inline__))
_mm_subs_epu8(__m128i __a, __m128i __b)
{
    return __CCC_M128I_FROM_BUILTIN(__builtin_ia32_psubusb128(__a, __b));
}

/* === Bitwise === */

static __inline__ __m128i __attribute__((__always_inline__))
_mm_or_si128(__m128i __a, __m128i __b)
{
    return (__m128i){ { __a.__val[0] | __b.__val[0],
                        __a.__val[1] | __b.__val[1] } };
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_and_si128(__m128i __a, __m128i __b)
{
    return (__m128i){ { __a.__val[0] & __b.__val[0],
                        __a.__val[1] & __b.__val[1] } };
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_xor_si128(__m128i __a, __m128i __b)
{
    return (__m128i){ { __a.__val[0] ^ __b.__val[0],
                        __a.__val[1] ^ __b.__val[1] } };
}

/* === Shift operations === */

/* Byte-level shift left (PSLLDQ): shift __a left by __N bytes, zero-fill */
#define _mm_slli_si128(a, N) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pslldqi128((a), (N)))

/* Byte-level shift right (PSRLDQ): shift __a right by __N bytes, zero-fill */
#define _mm_srli_si128(a, N) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_psrldqi128((a), (N)))

/* Bit-level shift left on each 64-bit element (PSLLQ) */
#define _mm_slli_epi64(a, count) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_psllqi128((a), (count)))

/* Bit-level shift right on each 64-bit element (PSRLQ) */
#define _mm_srli_epi64(a, count) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_psrlqi128((a), (count)))

/* Shuffle 32-bit integers (PSHUFD) */
#define _mm_shuffle_epi32(a, imm) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pshufd128((a), (imm)))

/* Load low 64 bits into lower half, zero upper half (MOVQ) */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_loadl_epi64(__m128i const *__p)
{
    return __CCC_M128I_FROM_BUILTIN(__builtin_ia32_loadldi128(__p));
}

/* === Miscellaneous === */

static __inline__ int __attribute__((__always_inline__))
_mm_movemask_epi8(__m128i __a)
{
    return __builtin_ia32_pmovmskb128(__a);
}

/* === Streaming / Non-temporal stores === */

static __inline__ void __attribute__((__always_inline__))
_mm_stream_si128(__m128i *__p, __m128i __a)
{
    __builtin_ia32_movntdq(__p, __a);
}

static __inline__ void __attribute__((__always_inline__))
_mm_stream_si64(long long *__p, long long __a)
{
    __builtin_ia32_movnti64(__p, __a);
}

static __inline__ void __attribute__((__always_inline__))
_mm_stream_si32(int *__p, int __a)
{
    __builtin_ia32_movnti(__p, __a);
}

static __inline__ void __attribute__((__always_inline__))
_mm_stream_pd(double *__p, __m128d __a)
{
    __builtin_ia32_movntpd(__p, __a);
}

/* === Fence / Cache === */

static __inline__ void __attribute__((__always_inline__))
_mm_lfence(void)
{
    __builtin_ia32_lfence();
}

static __inline__ void __attribute__((__always_inline__))
_mm_mfence(void)
{
    __builtin_ia32_mfence();
}

static __inline__ void __attribute__((__always_inline__))
_mm_clflush(void const *__p)
{
    __builtin_ia32_clflush(__p);
}

#endif /* _EMMINTRIN_H_INCLUDED */
