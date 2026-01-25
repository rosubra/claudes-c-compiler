pub mod definitions;
pub mod func_state;
pub mod type_context;
pub mod lowering;
pub mod expr;
pub mod expr_builtins;
pub mod expr_atomics;
pub mod expr_calls;
pub mod expr_assign;
pub mod stmt;
mod stmt_init;
mod stmt_return;
pub mod lvalue;
pub mod types;
pub mod structs;
pub mod complex;
pub mod global_init;
pub mod global_init_bytes;
pub mod global_init_compound;
pub mod const_eval;
pub mod expr_types;
mod pointer_analysis;
mod ref_collection;

pub use lowering::Lowerer;
// TypeContext and FunctionTypedefInfo are defined in frontend::sema::type_context.
// Re-exported here for backward compatibility with external imports.
pub use crate::frontend::sema::type_context::{TypeContext, FunctionTypedefInfo};
