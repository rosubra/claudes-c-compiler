#!/bin/bash
# Setup i686 cross-compilation environment for building projects with C++ deps.
#
# The standard Docker/CI environment provides i686-linux-gnu-gcc but not
# i686-linux-gnu-g++ (the C++ cross-compiler). This script creates:
#
# 1. /usr/local/bin/i686-linux-gnu-g++ - A wrapper that handles fast_float
#    (Redis C++ dependency) by providing a C fallback via i686-linux-gnu-gcc.
#
# 2. /usr/i686-linux-gnu/lib/libstdc++.so - Symlink to libstdc++.so.6 so
#    that -lstdc++ resolves during linking.
#
# Usage: sudo ./scripts/setup_i686_cross.sh
#   (or run the commands within as root)
#
# This only needs to be run once per environment.

set -e

WRAPPER=/usr/local/bin/i686-linux-gnu-g++
LIBDIR=/usr/i686-linux-gnu/lib

# 1. Create i686-linux-gnu-g++ wrapper
if command -v i686-linux-gnu-g++ >/dev/null 2>&1; then
    echo "i686-linux-gnu-g++ already exists, skipping wrapper creation."
else
    echo "Creating i686-linux-gnu-g++ wrapper at $WRAPPER..."
    cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper for i686-linux-gnu-g++ when the real cross-compiler is not installed.
# Provides a C fallback for fast_float_strtod.cpp (Redis dependency).

for arg in "$@"; do
    if [[ "$arg" == *"fast_float_strtod.cpp" ]]; then
        dir=$(dirname "$arg")
        tmpfile="$dir/fast_float_strtod_c_fallback.c"
        cat > "$tmpfile" << 'EOF'
#include <stdlib.h>
#include <errno.h>
double fast_float_strtod(const char *nptr, char **endptr) {
    return strtod(nptr, endptr);
}
EOF
        new_args=()
        for a in "$@"; do
            case "$a" in
                *fast_float_strtod.cpp) new_args+=("$tmpfile") ;;
                -std=c++*) ;;
                *) new_args+=("$a") ;;
            esac
        done
        exec i686-linux-gnu-gcc "${new_args[@]}" -o fast_float_strtod.o
    fi
done
exec g++ -m32 "$@"
WRAPPER_EOF
    chmod +x "$WRAPPER"
    echo "  Created $WRAPPER"
fi

# 2. Create libstdc++.so symlink
if [ -d "$LIBDIR" ] && [ ! -e "$LIBDIR/libstdc++.so" ]; then
    if [ -e "$LIBDIR/libstdc++.so.6" ]; then
        echo "Creating libstdc++.so symlink in $LIBDIR..."
        ln -sf libstdc++.so.6 "$LIBDIR/libstdc++.so"
        echo "  Created $LIBDIR/libstdc++.so -> libstdc++.so.6"
    else
        echo "Warning: $LIBDIR/libstdc++.so.6 not found, skipping symlink."
    fi
else
    if [ -e "$LIBDIR/libstdc++.so" ]; then
        echo "libstdc++.so symlink already exists, skipping."
    else
        echo "Warning: $LIBDIR does not exist, skipping libstdc++ symlink."
    fi
fi

echo "i686 cross-compilation environment setup complete."
