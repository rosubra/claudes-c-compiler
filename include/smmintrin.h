/* CCC compiler bundled smmintrin.h - SSE4.1 / SSE4.2 intrinsics (CRC32) */
#ifndef _SMMINTRIN_H_INCLUDED
#define _SMMINTRIN_H_INCLUDED

#include <emmintrin.h>

/* === CRC32 intrinsics (SSE4.2) === */

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u8(unsigned int __crc, unsigned char __v)
{
    return __builtin_ia32_crc32qi(__crc, __v);
}

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u16(unsigned int __crc, unsigned short __v)
{
    return __builtin_ia32_crc32hi(__crc, __v);
}

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u32(unsigned int __crc, unsigned int __v)
{
    return __builtin_ia32_crc32si(__crc, __v);
}

static __inline__ unsigned long long __attribute__((__always_inline__))
_mm_crc32_u64(unsigned long long __crc, unsigned long long __v)
{
    return __builtin_ia32_crc32di(__crc, __v);
}

#endif /* _SMMINTRIN_H_INCLUDED */
