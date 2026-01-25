//! Re-exports from frontend::sema::type_context for backward compatibility.
//!
//! TypeContext and related types now live in frontend/sema/type_context.rs,
//! which is the correct module boundary: sema creates and populates the
//! TypeContext, then transfers ownership to the lowerer. This re-export
//! lets existing lowering code use `super::type_context::TypeContext`
//! without changing every import.

pub use crate::frontend::sema::type_context::{TypeContext, TypeScopeFrame, FunctionTypedefInfo, extract_fptr_typedef_info};
