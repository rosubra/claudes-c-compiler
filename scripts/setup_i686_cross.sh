#!/bin/bash
# Setup i686 cross-compilation environment for building projects.
#
# The standard Docker/CI environment provides i686-linux-gnu-gcc but has
# several gaps for full i686 cross-compilation support. This script creates:
#
# 1. A wrapper i686-linux-gnu-g++ that handles C++ compilation for known
#    libraries (e.g. Redis's fast_float) via pure-C fallback.
#
# 2. /usr/i686-linux-gnu/lib/libstdc++.so - Symlink for -lstdc++ resolution.
#
# 3. Sysroot symlinks at /usr/lib/i386-linux-gnu/ and /usr/include/i386-linux-gnu/
#    pointing to /usr/i686-linux-gnu/{lib,include}. These are needed because
#    configure scripts (e.g., TCC) detect the triplet as "i386-linux-gnu" and
#    look for CRT objects at /usr/lib/i386-linux-gnu/crti.o.
#
# 4. A smart uname wrapper that returns "i686" when called from configure
#    scripts that use ccc-i686 as the compiler. This is needed because
#    `uname -m` returns "x86_64" on the host, but TCC's configure uses
#    it to determine the target architecture.
#
# Usage:
#   sudo ./scripts/setup_i686_cross.sh              # installs to /usr/local/bin
#   sudo ./scripts/setup_i686_cross.sh ~/.cargo/bin  # installs to user dir
#
# Note: sudo is required for creating system-level symlinks.
# This only needs to be run once per environment.

set -e

# Default to /usr/local/bin, allow override via first argument
INSTALL_DIR="${1:-/usr/local/bin}"
WRAPPER="$INSTALL_DIR/i686-linux-gnu-g++"
LIBDIR=/usr/i686-linux-gnu/lib

# 1. Create i686-linux-gnu-g++ wrapper
if command -v i686-linux-gnu-g++ >/dev/null 2>&1; then
    echo "i686-linux-gnu-g++ already exists on PATH, skipping wrapper creation."
else
    echo "Creating i686-linux-gnu-g++ wrapper at $WRAPPER..."
    mkdir -p "$INSTALL_DIR"
    cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper for i686-linux-gnu-g++ when the real cross-compiler is not installed.
#
# For C++ source files: generates a C fallback stub (providing the same
# exported symbols) and compiles it with i686-linux-gnu-gcc. This avoids
# needing i686 C++ headers which are not available in the cross environment.
#
# Currently handles:
# - fast_float_strtod.cpp (Redis dependency) -> strtod() wrapper

compile_mode=false
has_cpp_input=false
output_file=""
prev_was_o=false

for arg in "$@"; do
    case "$arg" in
        -c) compile_mode=true ;;
        -o) prev_was_o=true; continue ;;
        *.cpp|*.cc|*.cxx|*.C) has_cpp_input=true ;;
    esac
    if $prev_was_o; then
        output_file="$arg"
        prev_was_o=false
    fi
done

if $compile_mode && $has_cpp_input; then
    # Compiling C++ source - provide a C fallback stub and compile with
    # the i686 C cross-compiler. This handles one source file at a time
    # (matching how make invokes the compiler for each .cpp file).
    cpp_src=""
    other_args=()
    for arg in "$@"; do
        case "$arg" in
            *.cpp|*.cc|*.cxx|*.C) cpp_src="$arg" ;;
            -std=c++*) ;; # strip C++ std flags
            *) other_args+=("$arg") ;;
        esac
    done

    base=$(basename "$cpp_src" | sed 's/\.[^.]*$//')
    stub="/tmp/cxx_stub_${base}_$$.c"

    if echo "$cpp_src" | grep -q "fast_float"; then
        cat > "$stub" << 'CSTUB'
#include <stdlib.h>
#include <errno.h>
double fast_float_strtod(const char *nptr, char **endptr) {
    return strtod(nptr, endptr);
}
CSTUB
    else
        # TODO: Unknown C++ file - only a warning stub is provided.
        # Add specific stubs here as new C++ dependencies are encountered.
        echo "Warning: i686-linux-gnu-g++ wrapper: unknown C++ file '$cpp_src', creating empty stub" >&2
        echo "/* C++ stub - no symbols */" > "$stub"
    fi

    # Use -o from args if provided, otherwise default to ${base}.o
    if [ -z "$output_file" ]; then
        output_file="${base}.o"
    fi

    # Replace the source file with our stub in the args
    filtered=(-c)
    for a in "${other_args[@]}"; do
        case "$a" in
            -c) ;; # already added
            -o) ;; # handled separately
            *) filtered+=("$a") ;;
        esac
    done

    i686-linux-gnu-gcc "${filtered[@]}" "$stub" -o "$output_file"
    ret=$?
    rm -f "$stub"
    exit $ret
else
    # Non-compile mode (linking etc.) - delegate to i686 gcc, strip -lstdc++
    filtered=()
    for arg in "$@"; do
        [ "$arg" != "-lstdc++" ] && filtered+=("$arg")
    done
    exec i686-linux-gnu-gcc "${filtered[@]}"
fi
WRAPPER_EOF
    chmod +x "$WRAPPER"
    echo "  Created $WRAPPER"
fi

# 2. Create libstdc++.so symlink for -lstdc++ to resolve
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

# 3. Create i386-linux-gnu sysroot symlinks for TCC/configure triplet detection
# Many configure scripts (including TCC) detect the target triplet by compiling
# a test program and checking for /usr/lib/<triplet>/crti.o. On cross-compilation
# environments, the i386 libraries live in /usr/i686-linux-gnu/lib/ but configure
# expects them at /usr/lib/i386-linux-gnu/.
I386_LIBDIR=/usr/lib/i386-linux-gnu
I686_LIBDIR=/usr/i686-linux-gnu/lib
I386_INCDIR=/usr/include/i386-linux-gnu
I686_INCDIR=/usr/i686-linux-gnu/include

if [ -d "$I686_LIBDIR" ]; then
    echo "Setting up i386-linux-gnu sysroot symlinks..."
    sudo mkdir -p "$I386_LIBDIR" 2>/dev/null || true
    # CRT startup files
    for f in crt1.o crti.o crtn.o Scrt1.o Mcrt1.o gcrt1.o grcrt1.o; do
        if [ -e "$I686_LIBDIR/$f" ] && [ ! -e "$I386_LIBDIR/$f" ]; then
            sudo ln -sf "$I686_LIBDIR/$f" "$I386_LIBDIR/$f" 2>/dev/null || true
        fi
    done
    # Key shared libraries
    for lib in libc.a libc.so libm.a libm.so libdl.a libdl.so libpthread.a libpthread.so; do
        if [ -e "$I686_LIBDIR/$lib" ] && [ ! -e "$I386_LIBDIR/$lib" ]; then
            sudo ln -sf "$I686_LIBDIR/$lib" "$I386_LIBDIR/$lib" 2>/dev/null || true
        fi
    done
    echo "  Symlinked CRT and library files to $I386_LIBDIR"
fi

if [ -d "$I686_INCDIR" ]; then
    sudo mkdir -p "$I386_INCDIR" 2>/dev/null || true
    for d in bits gnu sys asm; do
        if [ -d "$I686_INCDIR/$d" ] && [ ! -e "$I386_INCDIR/$d" ]; then
            sudo ln -sf "$I686_INCDIR/$d" "$I386_INCDIR/$d" 2>/dev/null || true
        fi
    done
    echo "  Symlinked include directories to $I386_INCDIR"
fi

# 4. Create uname wrapper for i686 configure script detection
# TCC's configure (and similar) uses `uname -m` to detect the target CPU.
# On x86_64 hosts cross-compiling for i686, uname returns x86_64 which causes
# TCC to compile the wrong codegen backend. This wrapper detects when a parent
# configure script is using ccc-i686 as the compiler and returns i686 instead.
UNAME_WRAPPER="$INSTALL_DIR/uname"
if [ ! -e "$UNAME_WRAPPER" ]; then
    echo "Creating uname wrapper at $UNAME_WRAPPER..."
    mkdir -p "$INSTALL_DIR"
    cat > "$UNAME_WRAPPER" << 'UNAME_EOF'
#!/bin/bash
# Smart uname wrapper for i686 cross-compilation detection.
# When the parent process is a configure script using ccc-i686 as CC,
# reports machine architecture as i686 for the -m flag.
if echo "$*" | grep -q "\-m"; then
    if [ -f "/proc/$PPID/cmdline" ]; then
        while IFS= read -r -d '' arg; do
            case "$arg" in
                --cc=*ccc-i686*|--cc=*i686*)
                    echo "i686"
                    exit 0
                    ;;
            esac
        done < "/proc/$PPID/cmdline"
    fi
    # Check grandparent (configure may use subshells)
    gppid=$(awk '{print $4}' "/proc/$PPID/stat" 2>/dev/null)
    if [ -n "$gppid" ] && [ "$gppid" -gt 1 ] 2>/dev/null; then
        if [ -f "/proc/$gppid/cmdline" ]; then
            while IFS= read -r -d '' arg; do
                case "$arg" in
                    --cc=*ccc-i686*|--cc=*i686*)
                        echo "i686"
                        exit 0
                        ;;
                esac
            done < "/proc/$gppid/cmdline"
        fi
    fi
fi
exec /usr/bin/uname "$@"
UNAME_EOF
    chmod +x "$UNAME_WRAPPER"
    echo "  Created $UNAME_WRAPPER"
else
    echo "uname wrapper already exists at $UNAME_WRAPPER, skipping."
fi

echo "i686 cross-compilation environment setup complete."
