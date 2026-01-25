pub mod sema;
pub mod builtins;
pub mod type_context;

pub use sema::SemanticAnalyzer;
pub use builtins::{resolve_builtin, builtin_to_libc_name, is_builtin, BuiltinInfo, BuiltinKind, BuiltinIntrinsic};
pub use type_context::{TypeContext, FunctionTypedefInfo};
