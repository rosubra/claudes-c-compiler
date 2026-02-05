//! Native RISC-V 64-bit ELF linker.
//!
//! Reads ELF .o relocatable files and .a static archives, resolves symbols,
//! applies RISC-V relocations, generates PLT/GOT for dynamic symbols, and
//! emits a dynamically-linked ELF executable.
//!
//! This is the default linker (used when the `gcc_linker` feature is disabled).
//!
//! ## Module structure
//!
//! - `elf_read`: ELF object file parsing (delegates to shared linker_common)
//! - `relocations`: RISC-V relocation constants, instruction patching, shared
//!   types (GlobalSym, MergedSection), and utility functions used by both
//!   executable and shared library linking
//! - `link`: Main linking logic for executables and shared libraries

mod elf_read;
mod relocations;
mod link;

pub use link::link_builtin;
pub use link::link_shared;
