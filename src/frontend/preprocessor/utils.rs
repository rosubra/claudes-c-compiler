/// Shared utility functions for the preprocessor module.

/// Check if a character can start a C identifier.
pub fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Check if a character can continue a C identifier.
pub fn is_ident_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Skip past a string or character literal in a char slice, starting at position `i`.
/// Returns the position after the closing quote. Handles backslash escapes.
/// `quote` should be either `'"'` or `'\''`.
pub fn skip_literal(chars: &[char], start: usize, quote: char) -> usize {
    let len = chars.len();
    let mut i = start + 1; // skip opening quote
    while i < len {
        if chars[i] == '\\' && i + 1 < len {
            i += 2; // skip escape sequence
        } else if chars[i] == quote {
            return i + 1; // past closing quote
        } else {
            i += 1;
        }
    }
    i // unterminated literal - return end
}

/// Copy a string or character literal from chars into result, starting at position `i`.
/// Returns the position after the closing quote. Handles backslash escapes.
pub fn copy_literal(chars: &[char], start: usize, quote: char, result: &mut String) -> usize {
    let len = chars.len();
    result.push(chars[start]); // push opening quote
    let mut i = start + 1;
    while i < len {
        if chars[i] == '\\' && i + 1 < len {
            result.push(chars[i]);
            result.push(chars[i + 1]);
            i += 2;
        } else if chars[i] == quote {
            result.push(chars[i]);
            return i + 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    i
}
