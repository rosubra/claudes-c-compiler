//! Shared linker infrastructure for all backends.
//!
//! This module extracts the duplicated linker code that was copied across x86,
//! ARM, RISC-V, and (partially) i686 backends. It provides:
//!
//! - **ELF64 object parser**: `parse_elf64_object()` replaces near-identical
//!   `parse_object()` functions in x86, ARM, and RISC-V linkers.
//! - **Shared library parser**: `parse_shared_library_symbols()` and `parse_soname()`
//!   for extracting dynamic symbols from .so files.
//! - **Archive loading**: `load_archive_members()` and `member_resolves_undefined()`
//!   for iterative archive resolution (the --start-group algorithm).
//! - **Section mapping**: `map_section_name()` for input-to-output section mapping.
//! - **DynStrTab**: Dynamic string table builder for dynamic linking.
//! - **GNU hash**: `build_gnu_hash()` for .gnu.hash section generation.
//! - **Program header writer**: `write_phdr64()` is already in elf.rs.
//!
//! Each backend linker still handles its own:
//! - Architecture-specific relocation application
//! - PLT/GOT layout (different instruction sequences per arch)
//! - ELF header emission (different e_machine, base addresses)
//! - Dynamic linking specifics (version tables, etc.)

use std::collections::HashMap;
use std::path::Path;

use crate::backend::elf::{
    ELF_MAGIC, ELFCLASS64, ELFDATA2LSB,
    ET_REL, ET_DYN,
    SHT_SYMTAB, SHT_RELA, SHT_DYNAMIC, SHT_NOBITS, SHT_DYNSYM,
    STB_LOCAL, STB_GLOBAL, STB_WEAK,
    STT_SECTION, STT_FILE,
    SHN_UNDEF,
    read_u16, read_u32, read_u64, read_i64, read_cstr,
    parse_archive_members,
};

// ── ELF64 object file types ──────────────────────────────────────────────
//
// These types are used by x86, ARM, and RISC-V linkers. The i686 linker uses
// its own ELF32 types since field widths differ (u32 vs u64).

/// Parsed ELF64 section header.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Elf64Section {
    pub name_idx: u32,
    pub name: String,
    pub sh_type: u32,
    pub flags: u64,
    pub addr: u64,
    pub offset: u64,
    pub size: u64,
    pub link: u32,
    pub info: u32,
    pub addralign: u64,
    pub entsize: u64,
}

/// Parsed ELF64 symbol.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Elf64Symbol {
    pub name_idx: u32,
    pub name: String,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
}

#[allow(dead_code)]
impl Elf64Symbol {
    pub fn binding(&self) -> u8 { self.info >> 4 }
    pub fn sym_type(&self) -> u8 { self.info & 0xf }
    pub fn visibility(&self) -> u8 { self.other & 0x3 }
    pub fn is_undefined(&self) -> bool { self.shndx == SHN_UNDEF }
    pub fn is_global(&self) -> bool { self.binding() == STB_GLOBAL }
    pub fn is_weak(&self) -> bool { self.binding() == STB_WEAK }
    pub fn is_local(&self) -> bool { self.binding() == STB_LOCAL }
}

/// Parsed ELF64 relocation with addend (RELA).
#[derive(Debug, Clone)]
pub struct Elf64Rela {
    pub offset: u64,
    pub sym_idx: u32,
    pub rela_type: u32,
    pub addend: i64,
}

/// Parsed ELF64 object file (.o).
#[derive(Debug)]
pub struct Elf64Object {
    pub sections: Vec<Elf64Section>,
    pub symbols: Vec<Elf64Symbol>,
    pub section_data: Vec<Vec<u8>>,
    /// Relocations indexed by the section they apply to.
    pub relocations: Vec<Vec<Elf64Rela>>,
    pub source_name: String,
}

/// Dynamic symbol from a shared library (.so).
#[derive(Debug, Clone)]
pub struct DynSymbol {
    pub name: String,
    pub info: u8,
    pub value: u64,
    pub size: u64,
}

#[allow(dead_code)]
impl DynSymbol {
    pub fn sym_type(&self) -> u8 { self.info & 0xf }
}

// ── ELF64 object parsing ─────────────────────────────────────────────────
//
// This single function replaces the near-identical parse_object() functions
// in x86/linker/elf.rs, arm/linker/elf.rs, and riscv/linker/elf_read.rs.
// The only parameter that differed was the expected e_machine value.

/// Parse an ELF64 relocatable object file (.o).
///
/// `expected_machine` is the ELF e_machine value to validate (e.g., EM_X86_64,
/// EM_AARCH64, EM_RISCV). Pass 0 to skip machine validation.
pub fn parse_elf64_object(data: &[u8], source_name: &str, expected_machine: u16) -> Result<Elf64Object, String> {
    if data.len() < 64 {
        return Err(format!("{}: file too small for ELF header", source_name));
    }
    if data[0..4] != ELF_MAGIC {
        return Err(format!("{}: not an ELF file", source_name));
    }
    if data[4] != ELFCLASS64 {
        return Err(format!("{}: not 64-bit ELF", source_name));
    }
    if data[5] != ELFDATA2LSB {
        return Err(format!("{}: not little-endian ELF", source_name));
    }

    let e_type = read_u16(data, 16);
    if e_type != ET_REL {
        return Err(format!("{}: not a relocatable object (type={})", source_name, e_type));
    }

    if expected_machine != 0 {
        let e_machine = read_u16(data, 18);
        if e_machine != expected_machine {
            return Err(format!("{}: wrong machine type (expected={}, got={})",
                source_name, expected_machine, e_machine));
        }
    }

    let e_shoff = read_u64(data, 40) as usize;
    let e_shentsize = read_u16(data, 58) as usize;
    let e_shnum = read_u16(data, 60) as usize;
    let e_shstrndx = read_u16(data, 62) as usize;

    if e_shoff == 0 || e_shnum == 0 {
        return Err(format!("{}: no section headers", source_name));
    }

    // Parse section headers
    let mut sections = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let off = e_shoff + i * e_shentsize;
        if off + e_shentsize > data.len() {
            return Err(format!("{}: section header {} out of bounds", source_name, i));
        }
        sections.push(Elf64Section {
            name_idx: read_u32(data, off),
            name: String::new(),
            sh_type: read_u32(data, off + 4),
            flags: read_u64(data, off + 8),
            addr: read_u64(data, off + 16),
            offset: read_u64(data, off + 24),
            size: read_u64(data, off + 32),
            link: read_u32(data, off + 40),
            info: read_u32(data, off + 44),
            addralign: read_u64(data, off + 48),
            entsize: read_u64(data, off + 56),
        });
    }

    // Read section name string table
    if e_shstrndx < sections.len() {
        let shstrtab = &sections[e_shstrndx];
        let strtab_off = shstrtab.offset as usize;
        let strtab_size = shstrtab.size as usize;
        if strtab_off + strtab_size <= data.len() {
            let strtab_data = &data[strtab_off..strtab_off + strtab_size];
            for sec in &mut sections {
                sec.name = read_cstr(strtab_data, sec.name_idx as usize);
            }
        }
    }

    // Read section data
    let mut section_data = Vec::with_capacity(e_shnum);
    for sec in &sections {
        if sec.sh_type == SHT_NOBITS || sec.size == 0 {
            section_data.push(Vec::new());
        } else {
            let start = sec.offset as usize;
            let end = start + sec.size as usize;
            if end > data.len() {
                return Err(format!("{}: section '{}' data out of bounds", source_name, sec.name));
            }
            section_data.push(data[start..end].to_vec());
        }
    }

    // Find symbol table and its string table
    let mut symbols = Vec::new();
    for i in 0..sections.len() {
        if sections[i].sh_type == SHT_SYMTAB {
            let strtab_idx = sections[i].link as usize;
            let strtab_data = if strtab_idx < section_data.len() {
                &section_data[strtab_idx]
            } else {
                continue;
            };
            let sym_data = &section_data[i];
            let sym_count = sym_data.len() / 24; // sizeof(Elf64_Sym) = 24
            for j in 0..sym_count {
                let off = j * 24;
                if off + 24 > sym_data.len() {
                    break;
                }
                let name_idx = read_u32(sym_data, off);
                symbols.push(Elf64Symbol {
                    name_idx,
                    name: read_cstr(strtab_data, name_idx as usize),
                    info: sym_data[off + 4],
                    other: sym_data[off + 5],
                    shndx: read_u16(sym_data, off + 6),
                    value: read_u64(sym_data, off + 8),
                    size: read_u64(sym_data, off + 16),
                });
            }
            break;
        }
    }

    // Parse relocations - index by the section they apply to
    let mut relocations = vec![Vec::new(); e_shnum];
    for i in 0..sections.len() {
        if sections[i].sh_type == SHT_RELA {
            let target_sec = sections[i].info as usize;
            let rela_data = &section_data[i];
            let rela_count = rela_data.len() / 24; // sizeof(Elf64_Rela) = 24
            let mut relas = Vec::with_capacity(rela_count);
            for j in 0..rela_count {
                let off = j * 24;
                if off + 24 > rela_data.len() {
                    break;
                }
                let r_info = read_u64(rela_data, off + 8);
                relas.push(Elf64Rela {
                    offset: read_u64(rela_data, off),
                    sym_idx: (r_info >> 32) as u32,
                    rela_type: (r_info & 0xffffffff) as u32,
                    addend: read_i64(rela_data, off + 16),
                });
            }
            if target_sec < relocations.len() {
                relocations[target_sec] = relas;
            }
        }
    }

    Ok(Elf64Object {
        sections,
        symbols,
        section_data,
        relocations,
        source_name: source_name.to_string(),
    })
}

// ── Shared library parsing ───────────────────────────────────────────────

/// Extract dynamic symbols from a shared library (.so) file.
///
/// Reads the .dynsym section to find exported symbols. Used by x86 and RISC-V
/// linkers for dynamic linking resolution.
pub fn parse_shared_library_symbols(data: &[u8], lib_name: &str) -> Result<Vec<DynSymbol>, String> {
    if data.len() < 64 {
        return Err(format!("{}: file too small for ELF header", lib_name));
    }
    if data[0..4] != ELF_MAGIC {
        return Err(format!("{}: not an ELF file", lib_name));
    }
    if data[4] != ELFCLASS64 || data[5] != ELFDATA2LSB {
        return Err(format!("{}: not 64-bit little-endian ELF", lib_name));
    }

    let e_type = read_u16(data, 16);
    if e_type != ET_DYN {
        return Err(format!("{}: not a shared library (type={})", lib_name, e_type));
    }

    let e_shoff = read_u64(data, 40) as usize;
    let e_shentsize = read_u16(data, 58) as usize;
    let e_shnum = read_u16(data, 60) as usize;

    if e_shoff == 0 || e_shnum == 0 {
        return Err(format!("{}: no section headers", lib_name));
    }

    // Parse section headers (minimal: just type/offset/size/link)
    let mut sections = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let off = e_shoff + i * e_shentsize;
        if off + e_shentsize > data.len() {
            break;
        }
        sections.push((
            read_u32(data, off + 4),  // sh_type
            read_u64(data, off + 24), // offset
            read_u64(data, off + 32), // size
            read_u32(data, off + 40), // link
        ));
    }

    // Find .dynsym and its string table
    let mut symbols = Vec::new();
    for i in 0..sections.len() {
        let (sh_type, offset, size, link) = sections[i];
        if sh_type == SHT_DYNSYM {
            let strtab_idx = link as usize;
            if strtab_idx >= sections.len() { continue; }
            let (_, str_off, str_size, _) = sections[strtab_idx];
            let str_off = str_off as usize;
            let str_size = str_size as usize;
            if str_off + str_size > data.len() { continue; }
            let strtab = &data[str_off..str_off + str_size];

            let sym_off = offset as usize;
            let sym_size = size as usize;
            if sym_off + sym_size > data.len() { continue; }
            let sym_data = &data[sym_off..sym_off + sym_size];
            let sym_count = sym_data.len() / 24;

            for j in 1..sym_count { // skip null symbol at index 0
                let off = j * 24;
                if off + 24 > sym_data.len() { break; }
                let name_idx = read_u32(sym_data, off) as usize;
                let info = sym_data[off + 4];
                let shndx = read_u16(sym_data, off + 6);
                let value = read_u64(sym_data, off + 8);
                let size = read_u64(sym_data, off + 16);

                if shndx == SHN_UNDEF { continue; }

                let name = read_cstr(strtab, name_idx);
                if name.is_empty() { continue; }

                symbols.push(DynSymbol { name, info, value, size });
            }
            break;
        }
    }

    Ok(symbols)
}

/// Get the SONAME from a shared library's .dynamic section.
pub fn parse_soname(data: &[u8]) -> Option<String> {
    if data.len() < 64 || data[0..4] != ELF_MAGIC {
        return None;
    }

    let e_shoff = read_u64(data, 40) as usize;
    let e_shentsize = read_u16(data, 58) as usize;
    let e_shnum = read_u16(data, 60) as usize;

    const DT_NULL: i64 = 0;
    const DT_SONAME: i64 = 14;

    for i in 0..e_shnum {
        let off = e_shoff + i * e_shentsize;
        if off + 64 > data.len() { break; }
        let sh_type = read_u32(data, off + 4);
        if sh_type == SHT_DYNAMIC {
            let dyn_off = read_u64(data, off + 24) as usize;
            let dyn_size = read_u64(data, off + 32) as usize;
            let link = read_u32(data, off + 40) as usize;

            let str_sec_off = e_shoff + link * e_shentsize;
            if str_sec_off + 64 > data.len() { return None; }
            let str_off = read_u64(data, str_sec_off + 24) as usize;
            let str_size = read_u64(data, str_sec_off + 32) as usize;
            if str_off + str_size > data.len() { return None; }
            let strtab = &data[str_off..str_off + str_size];

            let mut pos = dyn_off;
            while pos + 16 <= dyn_off + dyn_size && pos + 16 <= data.len() {
                let tag = read_i64(data, pos);
                let val = read_u64(data, pos + 8);
                if tag == DT_NULL { break; }
                if tag == DT_SONAME {
                    return Some(read_cstr(strtab, val as usize));
                }
                pos += 16;
            }
        }
    }
    None
}

// ── Section name mapping ─────────────────────────────────────────────────

/// Map an input section name to the standard output section name.
///
/// This is the shared implementation used by all linker backends. Input sections
/// like `.text.foo` are merged into `.text`, `.rodata.bar` into `.rodata`, etc.
/// RISC-V additionally maps `.sdata`/`.sbss` (via `map_section_name_riscv()`).
pub fn map_section_name(name: &str) -> &str {
    if name.starts_with(".text.") || name == ".text" { return ".text"; }
    if name.starts_with(".data.rel.ro") { return ".data.rel.ro"; }
    if name.starts_with(".data.") || name == ".data" { return ".data"; }
    if name.starts_with(".rodata.") || name == ".rodata" { return ".rodata"; }
    if name.starts_with(".bss.") || name == ".bss" { return ".bss"; }
    if name.starts_with(".init_array") { return ".init_array"; }
    if name.starts_with(".fini_array") { return ".fini_array"; }
    if name.starts_with(".tbss.") || name == ".tbss" { return ".tbss"; }
    if name.starts_with(".tdata.") || name == ".tdata" { return ".tdata"; }
    if name.starts_with(".gcc_except_table") { return ".gcc_except_table"; }
    if name.starts_with(".eh_frame") { return ".eh_frame"; }
    if name.starts_with(".note.") { return name; }
    name
}

// ── Dynamic string table ─────────────────────────────────────────────────

/// Dynamic string table builder.
///
/// Used by linkers that produce dynamically-linked executables (x86, i686, RISC-V).
/// Deduplicates strings and tracks offsets for .dynstr section emission.
pub struct DynStrTab {
    data: Vec<u8>,
    offsets: HashMap<String, usize>,
}

impl DynStrTab {
    pub fn new() -> Self {
        Self { data: vec![0], offsets: HashMap::new() }
    }

    pub fn add(&mut self, s: &str) -> usize {
        if s.is_empty() { return 0; }
        if let Some(&off) = self.offsets.get(s) { return off; }
        let off = self.data.len();
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);
        self.offsets.insert(s.to_string(), off);
        off
    }

    pub fn get_offset(&self, s: &str) -> usize {
        if s.is_empty() { 0 } else { self.offsets.get(s).copied().unwrap_or(0) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ── GNU hash table ───────────────────────────────────────────────────────

/// Compute the GNU hash of a symbol name.
pub fn gnu_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &b in name {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

/// Compute the SysV ELF hash of a symbol name.
pub fn sysv_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in name {
        h = (h << 4).wrapping_add(b as u32);
        let g = h & 0xf0000000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}

// ── Archive loading ──────────────────────────────────────────────────────

/// Check if an archive member defines any currently-undefined symbol.
///
/// Used by iterative archive resolution (the --start-group algorithm) where
/// members are pulled in only if they resolve at least one undefined reference.
// TODO: migrate x86/ARM archive loading to use this shared implementation
#[allow(dead_code)]
pub fn member_resolves_undefined_elf64(
    obj: &Elf64Object,
    undefined: &dyn Fn(&str) -> bool,
) -> bool {
    for sym in &obj.symbols {
        if sym.is_undefined() || sym.is_local() { continue; }
        if sym.sym_type() == STT_SECTION || sym.sym_type() == STT_FILE { continue; }
        if sym.name.is_empty() { continue; }
        if undefined(&sym.name) { return true; }
    }
    false
}

/// Load and resolve archive members that satisfy undefined symbols.
///
/// Parses the archive, then iteratively pulls in members that define currently-
/// undefined symbols until no more progress is made (group resolution).
///
/// Returns the list of parsed objects that were pulled in.
// TODO: migrate x86/ARM archive loading to use this shared implementation
#[allow(dead_code)]
pub fn load_archive_members_elf64(
    data: &[u8],
    archive_path: &str,
    expected_machine: u16,
    undefined: &dyn Fn(&str) -> bool,
) -> Result<Vec<Elf64Object>, String> {
    let members = parse_archive_members(data)?;
    let mut member_objects: Vec<Elf64Object> = Vec::new();
    for (name, offset, size) in &members {
        let member_data = &data[*offset..*offset + *size];
        if member_data.len() < 4 || member_data[0..4] != ELF_MAGIC { continue; }
        // Optionally check machine type
        if expected_machine != 0 && member_data.len() >= 20 {
            let e_machine = read_u16(member_data, 18);
            if e_machine != expected_machine { continue; }
        }
        let full_name = format!("{}({})", archive_path, name);
        if let Ok(obj) = parse_elf64_object(member_data, &full_name, expected_machine) {
            member_objects.push(obj);
        }
    }

    let mut pulled_in = Vec::new();
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < member_objects.len() {
            if member_resolves_undefined_elf64(&member_objects[i], undefined) {
                pulled_in.push(member_objects.remove(i));
                changed = true;
            } else {
                i += 1;
            }
        }
    }
    Ok(pulled_in)
}

// ── Library resolution helper ─────────────────────────────────────────────

/// Resolve a library name to a path by searching directories.
///
/// Handles both `-l:filename` (exact match) and `-lfoo` (lib prefix search).
/// When `prefer_static` is true, searches for `.a` before `.so`.
pub fn resolve_lib(name: &str, paths: &[String], prefer_static: bool) -> Option<String> {
    if let Some(exact) = name.strip_prefix(':') {
        for dir in paths {
            let p = format!("{}/{}", dir, exact);
            if Path::new(&p).exists() { return Some(p); }
        }
        return None;
    }
    if prefer_static {
        for dir in paths {
            let a = format!("{}/lib{}.a", dir, name);
            if Path::new(&a).exists() { return Some(a); }
            let so = format!("{}/lib{}.so", dir, name);
            if Path::new(&so).exists() { return Some(so); }
        }
    } else {
        for dir in paths {
            let so = format!("{}/lib{}.so", dir, name);
            if Path::new(&so).exists() { return Some(so); }
            let a = format!("{}/lib{}.a", dir, name);
            if Path::new(&a).exists() { return Some(a); }
        }
    }
    None
}
