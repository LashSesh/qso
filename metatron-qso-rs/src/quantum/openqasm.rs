//! OpenQASM 2.0 and a subset of OpenQASM 3 into [`Program`]s.
//!
//! One parser reads both versions; the `OPENQASM` statement selects the
//! version (a program without it is read as OpenQASM 3, where the header
//! is optional).
//!
//! Supported:
//!
//! * declarations `qreg`/`creg` and `qubit`/`bit` (single or arrays),
//!   several registers, indices and inclusive slices `r[a:b]`;
//! * `include "qelib1.inc"` / `"stdgates.inc"` (the standard gates are
//!   built in; other files are errors);
//! * `gate` definitions with parameters, used by later gates and the
//!   program; `U`, `CX` and `gphase`;
//! * the gate modifiers `ctrl @`, `ctrl(n) @`, `negctrl @`, `inv @` and
//!   `pow(k) @` for integer `k`;
//! * `measure q -> c`, `c = measure q`, `reset q`, `barrier` (ignored);
//! * conditions `if (c == v)`, `if (c[i] == v)`, `if (int[n](c) == v)`
//!   and `!=`, with a statement or a block, and `else`;
//! * register broadcasting and parameter expressions with `+ − * / ^ **`,
//!   `sin cos tan exp ln sqrt arcsin arccos arctan`, `pi π tau τ euler ℇ`.
//!
//! Not supported (an error names the feature): classical variables and
//! arithmetic, loops, subroutines, `defcal`/`cal`, timing, `opaque`,
//! hardware qubits, and non-integer `pow`.
//!
//! The semantics follow each version's specification exactly, including
//! global phases where they matter. In OpenQASM 3 the built-in
//! `U(θ, φ, λ)` is `e^{iθ/2}` times the standard U3 matrix (QSO's
//! [`GateKind::U`]); OpenQASM 2's `U` equals that matrix up to a global
//! phase, which OpenQASM 2 cannot observe. `gphase` and the phases of
//! `u2`/`u3` become relative phases under `ctrl @`. QSO keeps these phases while expanding gates and drops the
//! global phase of the whole program, which is unobservable.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::circuit::Instruction;
use super::gates::GateKind;
use super::program::{Condition, Operation, Program, Register};
use crate::error::{QsoError, Result};

/// Largest number of qubits a program may declare (simulation needs at
/// most [`MAX_QUBITS`](super::MAX_QUBITS)).
pub const MAX_PROGRAM_QUBITS: usize = super::program::MAX_PROGRAM_QUBITS;
/// Largest number of operations a program may expand to.
pub const MAX_PROGRAM_OPERATIONS: usize = 1 << 22;
/// Deepest nesting of blocks and gate calls.
const MAX_DEPTH: usize = 64;
/// Greatest height of an expression (nesting plus chained operators).
const MAX_EXPR_DEPTH: usize = 256;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(QsoError::Parse(msg.into()))
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64, Option<u64>), // value; the exact value of an integer literal
    Str(String),
    Punct(&'static str),
}

const PUNCT2: [&str; 14] = [
    "->", "==", "!=", "<=", ">=", "&&", "||", "**", "<<", ">>", "+=", "-=", "*=", "/=",
];
const PUNCT1: [&str; 25] = [
    ";", ",", "(", ")", "[", "]", "{", "}", "@", "=", "+", "-", "*", "/", "^", ":", "!", "<", ">",
    "~", "%", "#", "|", "&", ".",
];

fn lex(src: &str) -> Result<Vec<(Tok, usize)>> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            loop {
                if i + 1 >= chars.len() {
                    return err(format!("line {line}: unterminated comment"));
                }
                if chars[i] == '\n' {
                    line += 1;
                }
                if chars[i] == '*' && chars[i + 1] == '/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push((Tok::Ident(chars[start..i].iter().collect()), line));
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let start = i;
            let mut integer = true;
            while i < chars.len() {
                let d = chars[i];
                let prev = if i > start { chars[i - 1] } else { ' ' };
                if d.is_ascii_digit() || d == '_' {
                    i += 1;
                } else if d == '.' || d == 'e' || d == 'E' {
                    integer = false;
                    i += 1;
                } else if (d == '+' || d == '-') && (prev == 'e' || prev == 'E') {
                    i += 1;
                } else {
                    break;
                }
            }
            let text: String = chars[start..i].iter().filter(|&&d| d != '_').collect();
            let v: f64 = text
                .parse()
                .map_err(|_| QsoError::Parse(format!("line {line}: invalid number '{text}'")))?;
            // Integers keep their exact value (f64 rounds above 2⁵³).
            let exact = if integer {
                text.parse::<u64>().ok()
            } else {
                None
            };
            out.push((Tok::Num(v, exact), line));
            continue;
        }
        if c == '"' {
            let start = i + 1;
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\n' {
                    return err(format!("line {line}: unterminated string"));
                }
                i += 1;
            }
            if i >= chars.len() {
                return err(format!("line {line}: unterminated string"));
            }
            out.push((Tok::Str(chars[start..i].iter().collect()), line));
            i += 1;
            continue;
        }
        if c == '$' {
            return err(format!(
                "line {line}: hardware qubits ('$') are not supported"
            ));
        }
        let two: String = chars[i..chars.len().min(i + 2)].iter().collect();
        if let Some(p) = PUNCT2.iter().find(|&&p| p == two) {
            out.push((Tok::Punct(p), line));
            i += 2;
            continue;
        }
        let one = c.to_string();
        if let Some(p) = PUNCT1.iter().find(|&&p| p == one) {
            out.push((Tok::Punct(p), line));
            i += 1;
            continue;
        }
        return err(format!("line {line}: unexpected character '{c}'"));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Syntax
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Call(String, Box<Expr>),
}

impl Expr {
    fn eval(&self, env: &HashMap<String, f64>) -> Result<f64> {
        Ok(match self {
            Expr::Num(v) => *v,
            // Gate parameters shadow the constants: `euler` and `tau` are
            // ordinary names in OpenQASM 2.
            Expr::Var(name) => match (env.get(name), name.as_str()) {
                (Some(v), _) => *v,
                (None, "pi" | "π") => PI,
                (None, "tau" | "τ") => 2.0 * PI,
                (None, "euler" | "ℇ") => std::f64::consts::E,
                _ => return err(format!("unknown identifier '{name}'")),
            },
            Expr::Neg(e) => -e.eval(env)?,
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => a / b,
                    _ => a.powf(b),
                }
            }
            Expr::Call(f, a) => {
                let a = a.eval(env)?;
                match f.as_str() {
                    "sin" => a.sin(),
                    "cos" => a.cos(),
                    "tan" => a.tan(),
                    "exp" => a.exp(),
                    "ln" => a.ln(),
                    "sqrt" => a.sqrt(),
                    "arcsin" => a.asin(),
                    "arccos" => a.acos(),
                    _ => a.atan(),
                }
            }
        })
    }

    fn value(&self, env: &HashMap<String, f64>) -> Result<f64> {
        let v = self.eval(env)?;
        if !v.is_finite() {
            return err("expression is not a finite number");
        }
        Ok(v)
    }
}

#[derive(Clone, Debug)]
enum Index {
    One(Expr),
    Slice(Expr, Option<Expr>, Expr), // start, step, end (inclusive)
}

#[derive(Clone, Debug)]
struct Operand {
    name: String,
    index: Option<Index>,
    line: usize,
}

#[derive(Clone, Debug)]
enum Modifier {
    Ctrl(usize, bool), // count, positive
    Inv,
    Pow(Expr),
}

#[derive(Clone, Debug)]
struct Call {
    modifiers: Vec<Modifier>,
    name: String,
    params: Vec<Expr>,
    operands: Vec<Operand>,
    line: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cast {
    /// No cast: the register's unsigned value, or a bit.
    None,
    /// `bool(c)`: whether any bit is set.
    Bool,
    /// `int[n](c)` / `uint[n](c)`: `signed`, and the width if given.
    Int(bool, Option<u64>),
}

#[derive(Clone, Debug)]
struct Cond {
    operand: Operand,
    cast: Cast,
    equal: bool,
    value: u64,
    /// The value was written `true` or `false`.
    boolean: bool,
}

#[derive(Clone, Debug)]
enum Stmt {
    Version(f64),
    Include(String),
    Qubits(String, usize),
    Bits(String, usize, Option<Operand>), // optional `= measure q`
    Gate {
        name: String,
        params: Vec<String>,
        qargs: Vec<String>,
        body: Vec<Call>,
        line: usize,
    },
    Call(Call),
    Measure(Operand, Option<Operand>),
    Reset(Operand),
    Barrier,
    If(Cond, Vec<Stmt>, Vec<Stmt>),
}

const UNSUPPORTED: [&str; 28] = [
    "opaque",
    "def",
    "defcal",
    "cal",
    "defcalgrammar",
    "extern",
    "for",
    "while",
    "const",
    "let",
    "int",
    "uint",
    "float",
    "angle",
    "bool",
    "duration",
    "stretch",
    "box",
    "delay",
    "input",
    "output",
    "return",
    "break",
    "continue",
    "switch",
    "array",
    "complex",
    "end",
];

struct Parser {
    toks: Vec<(Tok, usize)>,
    pos: usize,
    depth: usize,
    /// Height of the expression being parsed: nesting plus the length of
    /// operator chains, which build left-deep trees.
    expr_depth: usize,
}

impl Parser {
    fn line(&self) -> usize {
        self.toks
            .get(self.pos)
            .or(self.toks.last())
            .map_or(0, |t| t.1)
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.0)
    }

    fn peek_at(&self, k: usize) -> Option<&Tok> {
        self.toks.get(self.pos + k).map(|t| &t.0)
    }

    fn fail<T>(&self, msg: impl std::fmt::Display) -> Result<T> {
        err(format!("line {}: {msg}", self.line()))
    }

    fn is(&self, p: &str) -> bool {
        matches!(self.peek(), Some(Tok::Punct(q)) if *q == p)
    }

    fn is_ident(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(n)) if n == name)
    }

    fn eat(&mut self, p: &str) -> bool {
        if self.is(p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, p: &str) -> Result<()> {
        if self.eat(p) {
            Ok(())
        } else {
            self.fail(format!("expected '{p}'"))
        }
    }

    fn ident(&mut self) -> Result<String> {
        match self.peek() {
            Some(Tok::Ident(n)) => {
                let n = n.clone();
                self.pos += 1;
                Ok(n)
            }
            _ => self.fail("expected a name"),
        }
    }

    fn integer(&mut self) -> Result<u64> {
        match self.peek() {
            Some(&Tok::Num(_, Some(n))) => {
                self.pos += 1;
                Ok(n)
            }
            _ => self.fail("expected a non-negative integer"),
        }
    }

    fn deeper<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.depth >= MAX_DEPTH {
            return self.fail(format!("nested deeper than {MAX_DEPTH} levels"));
        }
        self.depth += 1;
        let r = f(self);
        self.depth -= 1;
        r
    }

    /// Run `f` one level deeper in an expression. Evaluating, cloning and
    /// dropping an expression recurse once per level, so the height is
    /// bounded.
    fn deeper_expr<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.expr_depth >= MAX_EXPR_DEPTH {
            return self.fail(format!(
                "expression nested or chained deeper than {MAX_EXPR_DEPTH} levels"
            ));
        }
        self.expr_depth += 1;
        let r = f(self);
        self.expr_depth -= 1;
        r
    }

    /// A left-associative chain `first (op next)*` with the operators
    /// `ops`; each operator adds a level.
    fn chain(
        &mut self,
        ops: [&'static str; 2],
        next: fn(&mut Self) -> Result<Expr>,
    ) -> Result<Expr> {
        let start = self.expr_depth;
        let mut e = next(self)?;
        let r = loop {
            let op = if self.eat(ops[0]) {
                ops[0]
            } else if self.eat(ops[1]) {
                ops[1]
            } else {
                break Ok(e);
            };
            if self.expr_depth >= MAX_EXPR_DEPTH {
                break self.fail(format!(
                    "expression nested or chained deeper than {MAX_EXPR_DEPTH} levels"
                ));
            }
            self.expr_depth += 1;
            let rhs = match next(self) {
                Ok(x) => x,
                Err(x) => break Err(x),
            };
            let c = op.chars().next().expect("non-empty operator");
            e = Expr::Bin(c, Box::new(e), Box::new(rhs));
        };
        self.expr_depth = start;
        r
    }

    // Expressions: sum → product → unary → power → atom.
    fn expr(&mut self) -> Result<Expr> {
        self.deeper_expr(|p| p.chain(["+", "-"], Self::product))
    }

    fn product(&mut self) -> Result<Expr> {
        self.chain(["*", "/"], Self::unary)
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.deeper_expr(Self::unary)?)));
        }
        if self.eat("+") {
            return self.deeper_expr(Self::unary);
        }
        let base = self.atom()?;
        if self.eat("^") || self.eat("**") {
            let exp = self.deeper_expr(Self::unary)?;
            return Ok(Expr::Bin('^', Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Expr> {
        match self.peek().cloned() {
            Some(Tok::Num(v, _)) => {
                self.pos += 1;
                Ok(Expr::Num(v))
            }
            Some(Tok::Punct("(")) => {
                self.pos += 1;
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                const FUNCS: [&str; 9] = [
                    "sin", "cos", "tan", "exp", "ln", "sqrt", "arcsin", "arccos", "arctan",
                ];
                if FUNCS.contains(&name.as_str()) {
                    self.expect("(")?;
                    let a = self.expr()?;
                    self.expect(")")?;
                    return Ok(Expr::Call(name, Box::new(a)));
                }
                Ok(Expr::Var(name))
            }
            _ => self.fail("expected an expression"),
        }
    }

    fn expr_list(&mut self) -> Result<Vec<Expr>> {
        let mut v = Vec::new();
        self.expect("(")?;
        if self.eat(")") {
            return Ok(v);
        }
        loop {
            v.push(self.expr()?);
            if self.eat(")") {
                return Ok(v);
            }
            self.expect(",")?;
        }
    }

    fn operand(&mut self) -> Result<Operand> {
        let line = self.line();
        let name = self.ident()?;
        let index = if self.eat("[") {
            let a = self.expr()?;
            let idx = if self.eat(":") {
                let b = self.expr()?;
                if self.eat(":") {
                    let c = self.expr()?;
                    Index::Slice(a, Some(b), c)
                } else {
                    Index::Slice(a, None, b)
                }
            } else {
                Index::One(a)
            };
            self.expect("]")?;
            Some(idx)
        } else {
            None
        };
        Ok(Operand { name, index, line })
    }

    fn operands(&mut self) -> Result<Vec<Operand>> {
        let mut v = Vec::new();
        if self.is(";") {
            return Ok(v);
        }
        loop {
            v.push(self.operand()?);
            if !self.eat(",") {
                return Ok(v);
            }
        }
    }

    fn call(&mut self) -> Result<Call> {
        let line = self.line();
        let mut modifiers = Vec::new();
        loop {
            let is_mod = matches!(self.peek(), Some(Tok::Ident(n)) if ["ctrl", "negctrl", "inv", "pow"].contains(&n.as_str()))
                && (matches!(self.peek_at(1), Some(Tok::Punct("@")))
                    || matches!(self.peek_at(1), Some(Tok::Punct("("))));
            if !is_mod {
                break;
            }
            let m = self.ident()?;
            let modifier = match m.as_str() {
                "inv" => Modifier::Inv,
                "pow" => {
                    self.expect("(")?;
                    let e = self.expr()?;
                    self.expect(")")?;
                    Modifier::Pow(e)
                }
                _ => {
                    let n = if self.eat("(") {
                        let n = self.integer()?;
                        self.expect(")")?;
                        n as usize
                    } else {
                        1
                    };
                    if n == 0 || n > 64 {
                        return self.fail("control counts must lie in 1..=64");
                    }
                    Modifier::Ctrl(n, m == "ctrl")
                }
            };
            self.expect("@")?;
            modifiers.push(modifier);
        }
        let name = self.ident()?;
        let params = if self.is("(") {
            self.expr_list()?
        } else {
            Vec::new()
        };
        let operands = self.operands()?;
        Ok(Call {
            modifiers,
            name,
            params,
            operands,
            line,
        })
    }

    fn block(&mut self, version: u8) -> Result<Vec<Stmt>> {
        if self.eat("{") {
            let mut body = Vec::new();
            while !self.eat("}") {
                if self.peek().is_none() {
                    return self.fail("missing '}'");
                }
                body.push(self.deeper(|p| p.statement(version))?);
            }
            Ok(body)
        } else {
            Ok(vec![self.deeper(|p| p.statement(version))?])
        }
    }

    fn condition(&mut self) -> Result<Cond> {
        // Optional cast: int[n](x), uint[n](x), int(x), bool(x).
        let kind = match self.peek() {
            Some(Tok::Ident(n)) if ["int", "uint", "bool"].contains(&n.as_str()) => Some(n.clone()),
            _ => None,
        };
        let (operand, cast) = if let Some(kind) = kind {
            self.pos += 1;
            let width = if kind != "bool" && self.eat("[") {
                let w = self.integer()?;
                self.expect("]")?;
                Some(w)
            } else {
                None
            };
            self.expect("(")?;
            let o = self.operand()?;
            self.expect(")")?;
            let cast = if kind == "bool" {
                Cast::Bool
            } else {
                Cast::Int(kind == "int", width)
            };
            (o, cast)
        } else {
            (self.operand()?, Cast::None)
        };
        let (equal, value, boolean) = if self.eat("==") || self.eat("!=") {
            let equal = matches!(self.toks[self.pos - 1].0, Tok::Punct("=="));
            let (value, boolean) = self.cond_value()?;
            (equal, value, boolean)
        } else {
            (false, 0, false) // truthy: value != 0
        };
        if cast == Cast::Bool && !boolean && (equal || value != 0) {
            return self.fail("bool(...) compares with true or false");
        }
        Ok(Cond {
            operand,
            cast,
            equal,
            value,
            boolean,
        })
    }

    fn cond_value(&mut self) -> Result<(u64, bool)> {
        if self.is_ident("true") {
            self.pos += 1;
            return Ok((1, true));
        }
        if self.is_ident("false") {
            self.pos += 1;
            return Ok((0, true));
        }
        Ok((self.integer()?, false))
    }

    fn statement(&mut self, version: u8) -> Result<Stmt> {
        let line = self.line();
        let Some(tok) = self.peek().cloned() else {
            return self.fail("unexpected end of program");
        };
        let Tok::Ident(word) = tok else {
            return self.fail("expected a statement");
        };
        // OpenQASM 3's keywords are ordinary names in OpenQASM 2 (a gate
        // may be called `box`); `opaque` is OpenQASM 2's own.
        if (version == 3 && UNSUPPORTED.contains(&word.as_str())) || word == "opaque" {
            return self.fail(format!("OpenQASM feature '{word}' is not supported"));
        }
        match word.as_str() {
            "OPENQASM" => {
                self.pos += 1;
                let v = match self.peek().cloned() {
                    Some(Tok::Num(v, _)) => {
                        self.pos += 1;
                        v
                    }
                    _ => return self.fail("expected a version number"),
                };
                self.expect(";")?;
                Ok(Stmt::Version(v))
            }
            "include" => {
                self.pos += 1;
                let Some(Tok::Str(file)) = self.peek().cloned() else {
                    return self.fail("expected a file name");
                };
                self.pos += 1;
                self.expect(";")?;
                Ok(Stmt::Include(file))
            }
            "qreg" | "creg" => {
                self.pos += 1;
                let name = self.ident()?;
                self.expect("[")?;
                let size = self.integer()? as usize;
                self.expect("]")?;
                self.expect(";")?;
                Ok(if word == "qreg" {
                    Stmt::Qubits(name, size)
                } else {
                    Stmt::Bits(name, size, None)
                })
            }
            "qubit" | "bit" => {
                self.pos += 1;
                let size = if self.eat("[") {
                    let s = self.integer()? as usize;
                    self.expect("]")?;
                    s
                } else {
                    1
                };
                let name = self.ident()?;
                if word == "qubit" {
                    self.expect(";")?;
                    return Ok(Stmt::Qubits(name, size));
                }
                let init = if self.eat("=") {
                    if !self.is_ident("measure") {
                        return self
                            .fail("classical initialisers other than 'measure' are not supported");
                    }
                    self.pos += 1;
                    Some(self.operand()?)
                } else {
                    None
                };
                self.expect(";")?;
                Ok(Stmt::Bits(name, size, init))
            }
            "gate" => {
                self.pos += 1;
                let name = self.ident()?;
                let mut params = Vec::new();
                if self.eat("(") && !self.eat(")") {
                    loop {
                        params.push(self.ident()?);
                        if self.eat(")") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                let mut qargs = Vec::new();
                loop {
                    qargs.push(self.ident()?);
                    if !self.eat(",") {
                        break;
                    }
                }
                self.expect("{")?;
                let mut body = Vec::new();
                while !self.eat("}") {
                    if self.peek().is_none() {
                        return self.fail("missing '}' in gate definition");
                    }
                    if self.is_ident("barrier") {
                        self.pos += 1;
                        self.operands()?;
                        self.expect(";")?;
                        continue;
                    }
                    body.push(self.call()?);
                    self.expect(";")?;
                }
                Ok(Stmt::Gate {
                    name,
                    params,
                    qargs,
                    body,
                    line,
                })
            }
            "measure" => {
                self.pos += 1;
                let q = self.operand()?;
                let c = if self.eat("->") {
                    Some(self.operand()?)
                } else {
                    None
                };
                self.expect(";")?;
                Ok(Stmt::Measure(q, c))
            }
            "reset" => {
                self.pos += 1;
                let q = self.operand()?;
                self.expect(";")?;
                Ok(Stmt::Reset(q))
            }
            "barrier" => {
                self.pos += 1;
                self.operands()?;
                self.expect(";")?;
                Ok(Stmt::Barrier)
            }
            "if" => {
                self.pos += 1;
                self.expect("(")?;
                let cond = self.condition()?;
                self.expect(")")?;
                let then = self.block(version)?;
                let other = if self.is_ident("else") {
                    self.pos += 1;
                    self.block(version)?
                } else {
                    Vec::new()
                };
                Ok(Stmt::If(cond, then, other))
            }
            _ => {
                // `c = measure q;` or a gate call.
                let save = self.pos;
                if !matches!(
                    self.peek_at(1),
                    Some(Tok::Punct("(")) | Some(Tok::Punct("@"))
                ) {
                    if let Ok(target) = self.operand()
                        && self.eat("=")
                    {
                        if !self.is_ident("measure") {
                            return self.fail(
                                "classical assignments other than 'measure' are not supported",
                            );
                        }
                        self.pos += 1;
                        let q = self.operand()?;
                        self.expect(";")?;
                        return Ok(Stmt::Measure(q, Some(target)));
                    }
                    self.pos = save;
                }
                let call = self.call()?;
                self.expect(";")?;
                Ok(Stmt::Call(call))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

/// A gate primitive: an instruction or a (relative, under control) phase.
#[derive(Clone, Debug)]
enum Prim {
    Gate(Instruction),
    Phase(f64),
}

struct GateDef {
    params: Vec<String>,
    qargs: Vec<String>,
    body: Vec<Call>,
}

struct Lower {
    version: u8,
    program: Program,
    gates: HashMap<String, GateDef>,
    budget: usize,
}

fn inst(kind: GateKind, targets: &[usize], controls: &[usize], params: &[f64]) -> Prim {
    Prim::Gate(Instruction {
        kind,
        targets: targets.to_vec(),
        controls: controls.to_vec(),
        params: params.iter().map(|&v| v.into()).collect(),
        matrix: None,
    })
}

impl Lower {
    fn spend(&mut self, n: usize) -> Result<()> {
        if n > self.budget {
            return err(format!(
                "the program expands to more than {MAX_PROGRAM_OPERATIONS} operations"
            ));
        }
        self.budget -= n;
        Ok(())
    }

    /// The standard gates; `None` if `name` is not one.
    fn builtin(&self, name: &str, p: &[f64], q: &[usize]) -> Option<Result<Vec<Prim>>> {
        use GateKind::*;
        let v3 = self.version == 3;
        let arity = |nq: usize, np: usize| -> Result<()> {
            if q.len() != nq || p.len() != np {
                return err(format!(
                    "'{name}' expects {nq} qubit(s) and {np} parameter(s), got {} and {}",
                    q.len(),
                    p.len()
                ));
            }
            Ok(())
        };
        let r = (|| -> Result<Vec<Prim>> {
            Ok(match name {
                "U" => {
                    arity(1, 3)?;
                    let mut v = vec![inst(U, q, &[], p)];
                    if v3 {
                        v.push(Prim::Phase(p[0] / 2.0));
                    }
                    v
                }
                "CX" => {
                    arity(2, 0)?;
                    vec![inst(X, &q[1..], &q[..1], &[])]
                }
                "gphase" => {
                    arity(0, 1)?;
                    vec![Prim::Phase(p[0])]
                }
                "id" => {
                    // An identity that device rules can give a duration
                    // (a wait), as `Circuit::id`.
                    arity(1, 0)?;
                    let one = C64::new(1.0, 0.0);
                    let zero = C64::new(0.0, 0.0);
                    vec![Prim::Gate(Instruction {
                        kind: Unitary,
                        targets: q.to_vec(),
                        controls: Vec::new(),
                        params: Vec::new(),
                        matrix: Some(std::sync::Arc::new(vec![one, zero, zero, one])),
                    })]
                }
                "u0" => {
                    arity(1, 1)?;
                    vec![]
                }
                "x" | "y" | "z" | "h" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" => {
                    arity(1, 0)?;
                    let kind = match name {
                        "x" => X,
                        "y" => Y,
                        "z" => Z,
                        "h" => H,
                        "s" => S,
                        "sdg" => Sdg,
                        "t" => T,
                        "tdg" => Tdg,
                        "sx" => SX,
                        _ => SXdg,
                    };
                    vec![inst(kind, q, &[], &[])]
                }
                "rx" | "ry" | "rz" | "p" | "phase" | "u1" => {
                    arity(1, 1)?;
                    let kind = match name {
                        "rx" => RX,
                        "ry" => RY,
                        "rz" => RZ,
                        _ => P,
                    };
                    vec![inst(kind, q, &[], p)]
                }
                "u3" | "u" => {
                    arity(1, 3)?;
                    let mut v = vec![inst(U, q, &[], p)];
                    if v3 && name == "u3" {
                        v.push(Prim::Phase(-(p[1] + p[2]) / 2.0));
                    }
                    v
                }
                "u2" => {
                    arity(1, 2)?;
                    let mut v = vec![inst(U, q, &[], &[PI / 2.0, p[0], p[1]])];
                    if v3 {
                        v.push(Prim::Phase(-(p[0] + p[1]) / 2.0));
                    }
                    v
                }
                "cx" | "cy" | "cz" | "ch" => {
                    arity(2, 0)?;
                    let kind = match name {
                        "cy" => Y,
                        "cz" => Z,
                        "ch" => H,
                        _ => X,
                    };
                    vec![inst(kind, &q[1..], &q[..1], &[])]
                }
                "cp" | "cphase" | "cu1" | "crx" | "cry" | "crz" => {
                    arity(2, 1)?;
                    let kind = match name {
                        "crx" => RX,
                        "cry" => RY,
                        "crz" => RZ,
                        _ => P,
                    };
                    vec![inst(kind, &q[1..], &q[..1], p)]
                }
                "cu3" => {
                    arity(2, 3)?;
                    vec![inst(U, &q[1..], &q[..1], p)]
                }
                "cu" => {
                    // Controlled e^{iγ}·U3(θ, φ, λ).
                    arity(2, 4)?;
                    vec![
                        inst(P, &q[..1], &[], &p[3..4]),
                        inst(U, &q[1..], &q[..1], &p[..3]),
                    ]
                }
                "swap" => {
                    arity(2, 0)?;
                    vec![inst(Swap, q, &[], &[])]
                }
                "ccx" => {
                    arity(3, 0)?;
                    vec![inst(X, &q[2..], &q[..2], &[])]
                }
                "cswap" => {
                    arity(3, 0)?;
                    vec![inst(Swap, &q[1..], &q[..1], &[])]
                }
                "rxx" | "ryy" | "rzz" => {
                    arity(2, 1)?;
                    let kind = match name {
                        "rxx" => RXX,
                        "ryy" => RYY,
                        _ => RZZ,
                    };
                    vec![inst(kind, q, &[], p)]
                }
                _ => return err(String::new()),
            })
        })();
        match r {
            Err(QsoError::Parse(m)) if m.is_empty() => None,
            other => Some(other),
        }
    }

    fn is_known(&self, name: &str) -> bool {
        self.gates.contains_key(name) || self.builtin(name, &[], &[]).is_some()
    }

    /// Expand a gate with its modifiers on concrete qubits.
    fn apply(
        &mut self,
        call: &Call,
        params: &[f64],
        qubits: &[usize],
        depth: usize,
    ) -> Result<Vec<Prim>> {
        if depth > MAX_DEPTH {
            return err(format!("gate calls nested deeper than {MAX_DEPTH} levels"));
        }
        let n_ctrl: usize = call
            .modifiers
            .iter()
            .map(|m| if let Modifier::Ctrl(n, _) = m { *n } else { 0 })
            .sum();
        if qubits.len() < n_ctrl {
            return err(format!(
                "line {}: '{}' needs {n_ctrl} control qubit(s)",
                call.line, call.name
            ));
        }
        let mut seen = qubits.to_vec();
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return err(format!(
                "line {}: repeated qubit in '{}'",
                call.line, call.name
            ));
        }
        let (controls, base_qubits) = qubits.split_at(n_ctrl);
        let mut prims = self.base(&call.name, params, base_qubits, call.line, depth)?;
        // Control groups in modifier order; modifiers act innermost first.
        let mut groups = Vec::new();
        let mut at = 0;
        for m in &call.modifiers {
            if let Modifier::Ctrl(n, _) = m {
                groups.push(&controls[at..at + n]);
                at += n;
            } else {
                groups.push(&controls[..0]);
            }
        }
        for (m, group) in call.modifiers.iter().zip(groups).rev() {
            prims = match m {
                Modifier::Inv => invert(prims),
                Modifier::Pow(e) => {
                    let k = e.value(&HashMap::new())?;
                    if k.fract() != 0.0 || k.abs() > MAX_PROGRAM_OPERATIONS as f64 {
                        return err(format!(
                            "line {}: only integer powers are supported, got {k}",
                            call.line
                        ));
                    }
                    let base = if k < 0.0 { invert(prims) } else { prims };
                    let reps = k.abs() as usize;
                    self.spend(weight(&base).saturating_mul(reps).max(1))?;
                    let mut out = Vec::with_capacity(base.len() * reps);
                    for _ in 0..reps {
                        out.extend(base.iter().cloned());
                    }
                    out
                }
                Modifier::Ctrl(_, positive) => {
                    // Every gate stores the new controls.
                    self.spend(prims.len().saturating_mul(group.len()))?;
                    let mut out = Vec::with_capacity(prims.len() + 2 * group.len());
                    let flips: Vec<Prim> = if *positive {
                        Vec::new()
                    } else {
                        group
                            .iter()
                            .map(|&c| inst(GateKind::X, &[c], &[], &[]))
                            .collect()
                    };
                    out.extend(flips.iter().cloned());
                    for p in prims {
                        out.push(match p {
                            Prim::Gate(mut i) => {
                                i.controls.extend_from_slice(group);
                                Prim::Gate(i)
                            }
                            Prim::Phase(g) => {
                                let (last, rest) = group.split_last().expect("non-empty");
                                inst(GateKind::P, &[*last], rest, &[g])
                            }
                        });
                    }
                    out.extend(flips);
                    out
                }
            };
        }
        Ok(prims)
    }

    fn base(
        &mut self,
        name: &str,
        params: &[f64],
        qubits: &[usize],
        line: usize,
        depth: usize,
    ) -> Result<Vec<Prim>> {
        if let Some(def) = self.gates.get(name) {
            if def.params.len() != params.len() || def.qargs.len() != qubits.len() {
                return err(format!(
                    "line {line}: '{name}' expects {} qubit(s) and {} parameter(s), got {} and {}",
                    def.qargs.len(),
                    def.params.len(),
                    qubits.len(),
                    params.len()
                ));
            }
            let env: HashMap<String, f64> = def
                .params
                .iter()
                .cloned()
                .zip(params.iter().copied())
                .collect();
            let map: HashMap<String, usize> = def
                .qargs
                .iter()
                .cloned()
                .zip(qubits.iter().copied())
                .collect();
            let body = def.body.clone();
            let mut out = Vec::new();
            for c in &body {
                let ps = c
                    .params
                    .iter()
                    .map(|e| e.value(&env))
                    .collect::<Result<Vec<f64>>>()?;
                let qs = c
                    .operands
                    .iter()
                    .map(|o| {
                        if o.index.is_some() {
                            return err(format!(
                                "line {}: gate bodies use whole arguments",
                                o.line
                            ));
                        }
                        map.get(&o.name).copied().ok_or_else(|| {
                            QsoError::Parse(format!(
                                "line {}: unknown argument '{}'",
                                o.line, o.name
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?;
                let modifiers = c
                    .modifiers
                    .iter()
                    .map(|m| match m {
                        Modifier::Pow(e) => Ok(Modifier::Pow(Expr::Num(e.value(&env)?))),
                        other => Ok(other.clone()),
                    })
                    .collect::<Result<Vec<Modifier>>>()?;
                let inner = Call {
                    modifiers,
                    ..c.clone()
                };
                let prims = self.apply(&inner, &ps, &qs, depth + 1)?;
                // Every call costs, also one that expands to nothing, so
                // that doubling definitions of `id` cannot run forever.
                self.spend(weight(&prims).max(1))?;
                out.extend(prims);
            }
            return Ok(out);
        }
        match self.builtin(name, params, qubits) {
            Some(r) => r,
            None => err(format!("line {line}: unknown gate '{name}'")),
        }
    }

    fn resolve(&self, o: &Operand, quantum: bool) -> Result<Vec<usize>> {
        let regs: &[Register] = if quantum {
            self.program.qubit_registers()
        } else {
            self.program.bit_registers()
        };
        let r = regs.iter().find(|r| r.name == o.name).ok_or_else(|| {
            QsoError::Parse(format!(
                "line {}: unknown {} register '{}'",
                o.line,
                if quantum { "quantum" } else { "classical" },
                o.name
            ))
        })?;
        let int = |e: &Expr| -> Result<i64> {
            let v = e.value(&HashMap::new())?;
            if v.fract() != 0.0 || v.abs() > 1e15 {
                return err(format!("line {}: indices must be integers", o.line));
            }
            Ok(v as i64)
        };
        let check = |i: i64| -> Result<usize> {
            // Negative indices count from the end, as in OpenQASM 3.
            let k = if i < 0 { r.size as i64 + i } else { i };
            if k < 0 || k >= r.size as i64 {
                return err(format!(
                    "line {}: index {i} out of range for '{}'",
                    o.line, o.name
                ));
            }
            Ok(r.offset + k as usize)
        };
        match &o.index {
            None => Ok((r.offset..r.offset + r.size).collect()),
            Some(Index::One(e)) => Ok(vec![check(int(e)?)?]),
            Some(Index::Slice(a, step, b)) => {
                // Negative endpoints count from the end, each on its own.
                let from_end = |i: i64| if i < 0 { r.size as i64 + i } else { i };
                let (a, b) = (from_end(int(a)?), from_end(int(b)?));
                let s = match step {
                    Some(e) => int(e)?,
                    None => 1,
                };
                if s == 0 {
                    return err(format!("line {}: zero step in slice", o.line));
                }
                let mut v = Vec::new();
                let mut i = a;
                while (s > 0 && i <= b) || (s < 0 && i >= b) {
                    v.push(check(i)?);
                    if v.len() > r.size {
                        return err(format!("line {}: slice larger than its register", o.line));
                    }
                    i += s;
                }
                Ok(v)
            }
        }
    }

    fn lower(&mut self, stmts: &[Stmt], top: bool) -> Result<Vec<Operation>> {
        let mut ops = Vec::new();
        for s in stmts {
            match s {
                // The version line, when present, was read before the
                // other statements.
                Stmt::Version(_) => return err("'OPENQASM' must come first"),
                Stmt::Include(f) => {
                    if !matches!(f.as_str(), "qelib1.inc" | "stdgates.inc") {
                        return err(format!(
                            "cannot include '{f}' (only qelib1.inc and stdgates.inc)"
                        ));
                    }
                }
                Stmt::Qubits(name, size) => {
                    if !top {
                        return err("declarations must be at the top level");
                    }
                    if self.program.n_qubits().saturating_add(*size) > MAX_PROGRAM_QUBITS {
                        return err(format!(
                            "programs may declare at most {MAX_PROGRAM_QUBITS} qubits"
                        ));
                    }
                    self.program.add_qubit_register(name, *size)?;
                }
                Stmt::Bits(name, size, init) => {
                    if !top {
                        return err("declarations must be at the top level");
                    }
                    self.program.add_bit_register(name, *size)?;
                    if let Some(q) = init {
                        let target = Operand {
                            name: name.clone(),
                            index: None,
                            line: q.line,
                        };
                        ops.extend(self.measure(q, &target)?);
                    }
                }
                Stmt::Gate {
                    name,
                    params,
                    qargs,
                    body,
                    line,
                } => {
                    if !top {
                        return err(format!(
                            "line {line}: gates must be defined at the top level"
                        ));
                    }
                    if self.gates.contains_key(name) {
                        return err(format!("line {line}: gate '{name}' defined twice"));
                    }
                    let mut seen = qargs.clone();
                    seen.sort();
                    seen.dedup();
                    if seen.len() != qargs.len() {
                        return err(format!("line {line}: repeated argument in gate '{name}'"));
                    }
                    for c in body {
                        if !self.is_known(&c.name) {
                            return err(format!(
                                "line {}: unknown gate '{}' in the definition of '{name}'",
                                c.line, c.name
                            ));
                        }
                    }
                    self.gates.insert(
                        name.clone(),
                        GateDef {
                            params: params.clone(),
                            qargs: qargs.clone(),
                            body: body.clone(),
                        },
                    );
                }
                Stmt::Call(call) => {
                    let params = call
                        .params
                        .iter()
                        .map(|e| e.value(&HashMap::new()))
                        .collect::<Result<Vec<f64>>>()?;
                    let lists = call
                        .operands
                        .iter()
                        .map(|o| self.resolve(o, true))
                        .collect::<Result<Vec<Vec<usize>>>>()?;
                    let width = lists
                        .iter()
                        .map(Vec::len)
                        .filter(|&l| l != 1)
                        .max()
                        .unwrap_or(1);
                    for l in &lists {
                        if l.len() != 1 && l.len() != width {
                            return err(format!(
                                "line {}: registers of different sizes in '{}'",
                                call.line, call.name
                            ));
                        }
                    }
                    for k in 0..width {
                        let qs: Vec<usize> = lists
                            .iter()
                            .map(|l| if l.len() == 1 { l[0] } else { l[k] })
                            .collect();
                        let prims = self.apply(call, &params, &qs, 0)?;
                        self.spend(weight(&prims).max(1))?;
                        for p in prims {
                            if let Prim::Gate(i) = p {
                                ops.push(Operation::Gate(i));
                            }
                        }
                    }
                }
                Stmt::Measure(q, c) => {
                    let Some(c) = c else {
                        return err(format!(
                            "line {}: measurements need a classical target",
                            q.line
                        ));
                    };
                    ops.extend(self.measure(q, c)?);
                }
                Stmt::Reset(q) => {
                    let qs = self.resolve(q, true)?;
                    self.spend(qs.len())?;
                    ops.extend(qs.into_iter().map(|qubit| Operation::Reset { qubit }));
                }
                Stmt::Barrier => {}
                Stmt::If(cond, then, other) => {
                    let bits = self.resolve(&cond.operand, false)?;
                    let line = cond.operand.line;
                    if bits.is_empty() || bits.len() > 64 {
                        return err(format!("line {line}: conditions read 1 to 64 bits"));
                    }
                    // `bool(c) == b` is `c != 0` or `c == 0`.
                    let (equal, value) = match cond.cast {
                        Cast::Bool => (cond.equal != (cond.value == 1), 0),
                        _ => (cond.equal, cond.value),
                    };
                    if let Cast::Int(signed, width) = cond.cast {
                        if width.is_some_and(|w| w != bits.len() as u64) {
                            return err(format!(
                                "line {line}: casting {} bit(s) to an integer of another width",
                                bits.len()
                            ));
                        }
                        // A signed value with the top bit set is negative;
                        // only non-negative values can be written here.
                        if signed && width.is_some() && value >> (bits.len() - 1) != 0 {
                            return err(format!(
                                "line {line}: {value} is out of range for int[{}]",
                                bits.len()
                            ));
                        }
                    }
                    if cond.boolean && cond.cast != Cast::Bool && bits.len() != 1 {
                        return err(format!(
                            "line {line}: only single bits compare with true or false"
                        ));
                    }
                    if bits.len() < 64 && value >> bits.len() != 0 {
                        return err(format!(
                            "line {line}: {value} does not fit into {} bit(s)",
                            bits.len()
                        ));
                    }
                    let then_ops = self.lower(then, false)?;
                    let else_ops = self.lower(other, false)?;
                    if !else_ops.is_empty() && writes_any(&then_ops, &bits) {
                        return err(format!(
                            "line {}: an 'if' branch that measures into its own condition cannot have an 'else'",
                            cond.operand.line
                        ));
                    }
                    let condition = Condition {
                        bits: bits.clone(),
                        value,
                        equal,
                    };
                    if !then_ops.is_empty() {
                        ops.push(Operation::If {
                            condition: condition.clone(),
                            body: then_ops,
                        });
                    }
                    if !else_ops.is_empty() {
                        ops.push(Operation::If {
                            condition: Condition {
                                equal: !condition.equal,
                                ..condition
                            },
                            body: else_ops,
                        });
                    }
                }
            }
        }
        Ok(ops)
    }

    fn measure(&mut self, q: &Operand, c: &Operand) -> Result<Vec<Operation>> {
        let qs = self.resolve(q, true)?;
        let cs = self.resolve(c, false)?;
        if qs.len() != cs.len() {
            return err(format!(
                "line {}: measuring {} qubit(s) into {} bit(s)",
                q.line,
                qs.len(),
                cs.len()
            ));
        }
        self.spend(qs.len())?;
        Ok(qs
            .into_iter()
            .zip(cs)
            .map(|(qubit, bit)| Operation::Measure { qubit, bit })
            .collect())
    }
}

fn writes_any(ops: &[Operation], bits: &[usize]) -> bool {
    ops.iter().any(|o| match o {
        Operation::Measure { bit, .. } => bits.contains(bit),
        Operation::If { body, .. } => writes_any(body, bits),
        _ => false,
    })
}

/// Size of gate primitives for the operation budget: one per gate and
/// one per control qubit it stores.
fn weight(prims: &[Prim]) -> usize {
    prims
        .iter()
        .map(|p| match p {
            Prim::Gate(i) => 1 + i.controls.len(),
            Prim::Phase(_) => 1,
        })
        .sum()
}

fn invert(prims: Vec<Prim>) -> Vec<Prim> {
    prims
        .into_iter()
        .rev()
        .map(|p| match p {
            Prim::Gate(i) => Prim::Gate(i.inverse()),
            Prim::Phase(g) => Prim::Phase(-g),
        })
        .collect()
}

/// Parse an OpenQASM 2 or 3 program; returns the program and the
/// language version (2 or 3).
pub fn parse(src: &str) -> Result<(Program, u8)> {
    let toks = lex(src)?;
    let mut parser = Parser {
        toks,
        pos: 0,
        depth: 0,
        expr_depth: 0,
    };
    // The version decides the meaning of `U`, `u2` and `u3`.
    let mut version = 3u8;
    if parser.is_ident("OPENQASM") {
        let line = parser.line();
        if let Stmt::Version(v) = parser.statement(3)? {
            version = if v == 2.0 {
                2
            } else if (3.0..4.0).contains(&v) {
                3
            } else {
                return err(format!("line {line}: unsupported OpenQASM version {v}"));
            };
        }
    }
    let mut stmts = Vec::new();
    while parser.peek().is_some() {
        stmts.push(parser.statement(version)?);
    }
    let mut lower = Lower {
        version,
        program: Program::with_registers(Vec::new(), Vec::new())?,
        gates: HashMap::new(),
        budget: MAX_PROGRAM_OPERATIONS,
    };
    let ops = lower.lower(&stmts, true)?;
    let mut program = lower.program;
    for op in ops {
        program.push(op)?;
    }
    Ok((program, version))
}

impl Program {
    /// Parse an OpenQASM 2.0 or OpenQASM 3 program (see
    /// [`openqasm`](super::openqasm) for the supported subset).
    pub fn from_qasm(src: &str) -> Result<Program> {
        parse(src).map(|(p, _)| p)
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

use std::fmt::Write as _;

use super::circuit::Circuit;
use super::qasm::{emit_instruction, real, zyz};
use num_complex::Complex64 as C64;

const DEF_RXX: &str = "gate rxx(θ) a, b { h a; h b; cx a, b; rz(θ) b; cx a, b; h a; h b; }";
const DEF_RYY: &str =
    "gate ryy(θ) a, b { sdg a; h a; sdg b; h b; cx a, b; rz(θ) b; cx a, b; h a; s a; h b; s b; }";
const DEF_RZZ: &str = "gate rzz(θ) a, b { cx a, b; rz(θ) b; cx a, b; }";

fn uses(ops: &[Operation], kind: GateKind) -> bool {
    ops.iter().any(|o| match o {
        Operation::Gate(i) => i.kind == kind,
        Operation::If { body, .. } => uses(body, kind),
        _ => false,
    })
}

/// Words an exported register may not be called: keywords, built-in
/// gates and constants of either version.
const RESERVED: [&str; 26] = [
    "OPENQASM", "include", "qubit", "bit", "qreg", "creg", "gate", "measure", "reset", "barrier",
    "if", "else", "ctrl", "negctrl", "inv", "pow", "gphase", "U", "CX", "pi", "tau", "euler",
    "true", "false", "opaque", "sizeof",
];

/// Register names of an export: the program's, made unique and free of
/// reserved words by appending `_`.
struct Names {
    qubits: Vec<String>,
    bits: Vec<String>,
}

impl Program {
    /// Names for an export; OpenQASM 2 exports have one quantum register
    /// `q`.
    fn export_names(&self, version: u8) -> Names {
        let mut used = std::collections::HashSet::new();
        let mut fresh = |name: &str| {
            let mut n = name.to_string();
            while RESERVED.contains(&n.as_str())
                || UNSUPPORTED.contains(&n.as_str())
                || used.contains(&n)
            {
                n.push('_');
            }
            used.insert(n.clone());
            n
        };
        let qubits = if version == 2 {
            vec![fresh("q")]
        } else {
            self.qubit_registers()
                .iter()
                .map(|r| fresh(&r.name))
                .collect()
        };
        let bits = self
            .bit_registers()
            .iter()
            .map(|r| fresh(&r.name))
            .collect();
        Names { qubits, bits }
    }

    fn qubit_name(&self, names: &Names, q: usize) -> String {
        let (k, r) = self
            .qubit_registers()
            .iter()
            .enumerate()
            .find(|(_, r)| q >= r.offset && q < r.offset + r.size)
            .expect("qubit in a register");
        format!("{}[{}]", names.qubits[k], q - r.offset)
    }

    fn bit_name(&self, names: &Names, b: usize) -> String {
        let (k, r) = self
            .bit_registers()
            .iter()
            .enumerate()
            .find(|(_, r)| b >= r.offset && b < r.offset + r.size)
            .expect("bit in a register");
        format!("{}[{}]", names.bits[k], b - r.offset)
    }

    /// The index of the register whose bits are exactly `bits`, in order.
    fn register_of(&self, bits: &[usize]) -> Option<usize> {
        self.bit_registers().iter().position(|r| {
            r.size == bits.len() && bits.iter().enumerate().all(|(k, &b)| b == r.offset + k)
        })
    }

    fn gate3(&self, names: &Names, out: &mut String, inst: &Instruction) -> Result<()> {
        use GateKind::*;
        let a = inst.angles(&[]);
        let k = inst.controls.len();
        let ctrl = |k: usize| match k {
            0 => String::new(),
            1 => "ctrl @ ".to_string(),
            _ => format!("ctrl({k}) @ "),
        };
        let args = |p: &[f64]| {
            if p.is_empty() {
                String::new()
            } else {
                format!(
                    "({})",
                    p.iter().map(|&v| real(v)).collect::<Vec<_>>().join(", ")
                )
            }
        };
        let qubits: Vec<String> = inst
            .controls
            .iter()
            .chain(&inst.targets)
            .map(|&q| self.qubit_name(names, q))
            .collect();
        let (head, params): (String, Vec<f64>) = match (inst.kind, k) {
            (X, 1) => ("cx".into(), vec![]),
            (X, 2) => ("ccx".into(), vec![]),
            (Y, 1) => ("cy".into(), vec![]),
            (Z, 1) => ("cz".into(), vec![]),
            (H, 1) => ("ch".into(), vec![]),
            (RX, 1) => ("crx".into(), a.clone()),
            (RY, 1) => ("cry".into(), a.clone()),
            (RZ, 1) => ("crz".into(), a.clone()),
            (P, 1) => ("cp".into(), a.clone()),
            (Swap, 1) => ("cswap".into(), vec![]),
            (H | X | Y | Z | S | Sdg | T | Tdg | SX | Swap, _) => {
                let name = match inst.kind {
                    SX => "sx".to_string(),
                    other => other.name().to_string(),
                };
                (format!("{}{name}", ctrl(k)), vec![])
            }
            (SXdg, _) => (format!("{}inv @ sx", ctrl(k)), vec![]),
            (RX | RY | RZ | P | RXX | RYY | RZZ, _) => {
                (format!("{}{}", ctrl(k), inst.kind.name()), a.clone())
            }
            (U, 0) => ("U".into(), a.clone()),
            (U, _) => (format!("{}cu", ctrl(k - 1)), vec![a[0], a[1], a[2], 0.0]),
            (Unitary, 0) if inst.is_identity() => ("id".into(), vec![]),
            (Unitary, _) => {
                let m = inst.matrix.as_ref().expect("unitary carries a matrix");
                if inst.targets.len() != 1 {
                    return Err(QsoError::invalid(
                        "circuit",
                        "custom unitaries on more than one qubit have no OpenQASM representation",
                    ));
                }
                let m: [C64; 4] = [m[0], m[1], m[2], m[3]];
                let (theta, phi, lambda, alpha) = zyz(&m);
                if k == 0 {
                    ("U".into(), vec![theta, phi, lambda])
                } else {
                    (
                        format!("{}cu", ctrl(k - 1)),
                        vec![theta, phi, lambda, alpha],
                    )
                }
            }
        };
        let _ = writeln!(out, "{head}{} {};", args(&params), qubits.join(", "));
        Ok(())
    }

    fn ops3(
        &self,
        names: &Names,
        out: &mut String,
        ops: &[Operation],
        indent: usize,
    ) -> Result<()> {
        let pad = "  ".repeat(indent);
        for op in ops {
            match op {
                Operation::Gate(i) => {
                    out.push_str(&pad);
                    self.gate3(names, out, i)?;
                }
                Operation::Measure { qubit, bit } => {
                    let _ = writeln!(
                        out,
                        "{pad}{} = measure {};",
                        self.bit_name(names, *bit),
                        self.qubit_name(names, *qubit)
                    );
                }
                Operation::Reset { qubit } => {
                    let _ = writeln!(out, "{pad}reset {};", self.qubit_name(names, *qubit));
                }
                Operation::If { condition, body } => {
                    let lhs = match self.register_of(&condition.bits) {
                        Some(k) if condition.bits.len() == 1 => names.bits[k].clone(),
                        Some(k) => format!("uint[{}]({})", condition.bits.len(), names.bits[k]),
                        None if condition.bits.len() == 1 => {
                            self.bit_name(names, condition.bits[0])
                        }
                        None => {
                            return Err(QsoError::invalid(
                                "program",
                                "conditions must read one bit or a whole register",
                            ));
                        }
                    };
                    let op = if condition.equal { "==" } else { "!=" };
                    let _ = writeln!(out, "{pad}if ({lhs} {op} {}) {{", condition.value);
                    self.ops3(names, out, body, indent + 1)?;
                    let _ = writeln!(out, "{pad}}}");
                }
            }
        }
        Ok(())
    }

    /// The program as OpenQASM 3 (standard gates, with `ctrl @` for gates
    /// with more controls; `rxx`, `ryy`, `rzz` are defined in the header).
    /// Exact including relative phases of controlled gates.
    pub fn to_qasm3(&self) -> Result<String> {
        let mut out = String::from("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
        for (kind, def) in [
            (GateKind::RXX, DEF_RXX),
            (GateKind::RYY, DEF_RYY),
            (GateKind::RZZ, DEF_RZZ),
        ] {
            if uses(self.operations(), kind) {
                let _ = writeln!(out, "{def}");
            }
        }
        let names = self.export_names(3);
        for (r, name) in self.qubit_registers().iter().zip(&names.qubits) {
            let _ = writeln!(out, "qubit[{}] {name};", r.size);
        }
        for (r, name) in self.bit_registers().iter().zip(&names.bits) {
            let _ = writeln!(out, "bit[{}] {name};", r.size);
        }
        self.ops3(&names, &mut out, self.operations(), 0)?;
        Ok(out)
    }

    /// The program as OpenQASM 2.0: one quantum register `q`, the
    /// classical registers of the program (renamed with a trailing `_`
    /// where a name is taken or reserved). Conditions must compare a whole
    /// register for equality and may not be nested, as OpenQASM 2 requires.
    pub fn to_qasm2(&self) -> Result<String> {
        let n = self.n_qubits();
        let mut out = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
        let _ = writeln!(out, "qreg q[{n}];");
        let names = self.export_names(2);
        for (r, name) in self.bit_registers().iter().zip(&names.bits) {
            let _ = writeln!(out, "creg {name}[{}];", r.size);
        }
        for op in self.operations() {
            self.op2(&names, &mut out, op, None)?;
        }
        Ok(out)
    }

    fn op2(
        &self,
        names: &Names,
        out: &mut String,
        op: &Operation,
        guard: Option<&str>,
    ) -> Result<()> {
        let mut text = String::new();
        match op {
            Operation::Gate(i) => emit_instruction(&mut text, i, &i.angles(&[]), self.n_qubits())?,
            Operation::Measure { qubit, bit } => {
                let _ = writeln!(
                    text,
                    "measure q[{qubit}] -> {};",
                    self.bit_name(names, *bit)
                );
            }
            Operation::Reset { qubit } => {
                let _ = writeln!(text, "reset q[{qubit}];");
            }
            Operation::If { condition, body } => {
                let reg = self
                    .register_of(&condition.bits)
                    .filter(|_| condition.equal);
                let (Some(reg), None) = (reg, guard) else {
                    return Err(QsoError::invalid(
                        "program",
                        "OpenQASM 2 conditions compare a whole register for equality and do not nest",
                    ));
                };
                // Every statement of the body is guarded separately, which
                // is only the same if the body does not change the register.
                if body.len() > 1 && writes_any(body, &condition.bits) {
                    return Err(QsoError::invalid(
                        "program",
                        "a conditional block that measures into its own condition has no OpenQASM 2 form",
                    ));
                }
                let g = format!("if({}=={}) ", names.bits[reg], condition.value);
                for o in body {
                    self.op2(names, out, o, Some(&g))?;
                }
                return Ok(());
            }
        }
        match guard {
            Some(g) => {
                for line in text.lines() {
                    let _ = writeln!(out, "{g}{line}");
                }
            }
            None => out.push_str(&text),
        }
        Ok(())
    }
}

/// Export `circuit` (with parameters `theta` bound) as OpenQASM 3; with
/// `measure_all` every qubit is measured into the bit of the same index.
pub fn to_qasm3(circuit: &Circuit, theta: &[f64], measure_all: bool) -> Result<String> {
    Program::from_circuit(circuit, theta, measure_all)?.to_qasm3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CMatrix;
    use crate::quantum::StateVector;
    use crate::rng::Rng;

    fn unitary(c: &Circuit) -> CMatrix {
        let n = c.n_qubits();
        let mut u = CMatrix::zeros(1 << n, 1 << n);
        for col in 0..1 << n {
            let mut psi = StateVector::basis(n, col).unwrap();
            psi.apply_circuit(c, &[]).unwrap();
            for (row, a) in psi.amplitudes().iter().enumerate() {
                u[(row, col)] = *a;
            }
        }
        u
    }

    fn program_unitary(src: &str) -> CMatrix {
        unitary(&Program::from_qasm(src).unwrap().to_circuit().unwrap())
    }

    /// `a` and `b` (bodies on qubits `q[0]`, `q[1]`, …) are the same
    /// unitary, phases included.
    fn assert_same(a: &str, b: &str, n: usize) {
        let head = |v: &str| format!("OPENQASM {v}; include \"stdgates.inc\"; qubit[{n}] q;\n");
        let ua = program_unitary(&format!("{}{a}", head("3.0")));
        let ub = program_unitary(&format!("{}{b}", head("3.0")));
        let diff = (&ua - &ub).iter().map(|z| z.norm()).fold(0.0, f64::max);
        assert!(diff < 1e-12, "{a}  vs  {b}: {diff}");
    }

    #[test]
    fn modifiers_and_global_phases_follow_the_specification() {
        // U carries e^{iθ/2} in OpenQASM 3: visible under ctrl.
        assert_same(
            "ctrl @ U(0.3, 0.2, 0.1) q[0], q[1];",
            "cu(0.3, 0.2, 0.1, 0.15) q[0], q[1];",
            2,
        );
        // u3 = e^{-i(φ+λ)/2}·U3.
        assert_same(
            "ctrl @ u3(0.3, 0.2, 0.1) q[0], q[1];",
            "cu(0.3, 0.2, 0.1, -0.15) q[0], q[1];",
            2,
        );
        assert_same("ctrl @ gphase(0.7) q[0];", "p(0.7) q[0];", 1);
        assert_same(
            "ctrl(2) @ gphase(0.7) q[0], q[1];",
            "cp(0.7) q[0], q[1];",
            2,
        );
        assert_same(
            "negctrl @ x q[0], q[1];",
            "x q[0]; cx q[0], q[1]; x q[0];",
            2,
        );
        assert_same("inv @ rx(0.3) q[0];", "rx(-0.3) q[0];", 1);
        assert_same("pow(3) @ t q[0];", "t q[0]; t q[0]; t q[0];", 1);
        assert_same("pow(-2) @ s q[0];", "sdg q[0]; sdg q[0];", 1);
        assert_same("ctrl @ h q[0], q[1];", "ch q[0], q[1];", 2);
        assert_same(
            "ctrl @ ctrl @ x q[0], q[1], q[2];",
            "ccx q[0], q[1], q[2];",
            3,
        );
        assert_same("ctrl @ inv @ s q[1], q[0];", "cp(-pi/2) q[1], q[0];", 2);
        // A gate with an internal global phase: e^{it/2}·RZ(t) = P(t).
        assert_same(
            "gate g(t) a { rz(t) a; gphase(t/2); } ctrl @ g(0.4) q[0], q[1];",
            "cp(0.4) q[0], q[1];",
            2,
        );
        // sx = pow(0.5) @ x = e^{iπ/4}·RX(π/2) = U(π/2, −π/2, π/2) in
        // OpenQASM 3, exactly.
        assert_same(
            "ctrl @ sx q[0], q[1];",
            "ctrl @ U(pi/2, -pi/2, pi/2) q[0], q[1];",
            2,
        );
        assert_same(
            "ctrl @ x q[0], q[1];",
            "ctrl @ U(pi, 0, pi) q[0], q[1]; ctrl @ gphase(-pi/2) q[0];",
            2,
        );
        // OpenQASM 2: U has no extra phase (cu3 is controlled U3).
        let v2 = program_unitary("OPENQASM 2.0; qreg q[2]; cu3(0.3,0.2,0.1) q[0],q[1];");
        let v3 = program_unitary("OPENQASM 3.0; qubit[2] q; cu(0.3, 0.2, 0.1, 0) q[0], q[1];");
        assert!((&v2 - &v3).iter().all(|z| z.norm() < 1e-12));
    }

    fn random_circuit(n: usize, gates: usize, seed: u64) -> Circuit {
        use GateKind::*;
        let mut rng = Rng::seed_from_u64(seed);
        let mut c = Circuit::new(n);
        for _ in 0..gates {
            let mut qs: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut qs);
            let k = rng.below(3) as usize; // controls
            let a = rng.uniform_range(-3.0, 3.0);
            let kinds = [
                H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, RX, RY, RZ, P, U, Swap, RXX, RYY, RZZ,
                Unitary,
            ];
            let kind = kinds[rng.below(kinds.len() as u64) as usize];
            let n_t = kind.num_targets().unwrap_or(1);
            if n_t + k > n {
                continue;
            }
            let targets = qs[..n_t].to_vec();
            let controls = qs[n_t..n_t + k].to_vec();
            let (params, matrix) = match kind {
                RX | RY | RZ | P | RXX | RYY | RZZ => (vec![a.into()], None),
                U => (vec![a.into(), (a * 0.7).into(), (-a * 0.3).into()], None),
                Unitary => {
                    let (t, p, l, g) = (a, 0.4 * a, -0.9, 0.3);
                    let m = vec![
                        C64::from_polar((t / 2.0).cos(), g),
                        -C64::from_polar((t / 2.0).sin(), g + l),
                        C64::from_polar((t / 2.0).sin(), g + p),
                        C64::from_polar((t / 2.0).cos(), g + p + l),
                    ];
                    (vec![], Some(std::sync::Arc::new(m)))
                }
                _ => (vec![], None),
            };
            c.try_push(Instruction {
                kind,
                targets,
                controls,
                params,
                matrix,
            })
            .unwrap();
        }
        c
    }

    #[test]
    fn circuits_round_trip_through_openqasm3_exactly() {
        for seed in 0..20 {
            let c = random_circuit(4, 30, seed);
            let text = to_qasm3(&c, &[], false).unwrap();
            let back = Program::from_qasm(&text).unwrap().to_circuit().unwrap();
            let (u, v) = (unitary(&c), unitary(&back));
            // Equal up to one global phase (uncontrolled U and custom
            // unitaries are exported without theirs).
            let k = (0..16).find(|&i| u[(i, 0)].norm() > 1e-6).unwrap();
            let phase = v[(k, 0)] / u[(k, 0)];
            let diff = (&u * phase - &v)
                .iter()
                .map(|z| z.norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-10, "seed {seed}: {diff}\n{text}");
            // And through OpenQASM 2 (qelib1 decompositions) where it has
            // a form (no controlled rxx, ryy, rzz).
            if let Ok(text2) = crate::quantum::to_qasm2(&c, &[], false) {
                let back2 = Program::from_qasm(&text2).unwrap().to_circuit().unwrap();
                let w = unitary(&back2);
                let phase = w[(k, 0)] / u[(k, 0)];
                let diff = (&u * phase - &w)
                    .iter()
                    .map(|z| z.norm())
                    .fold(0.0, f64::max);
                assert!(diff < 1e-10, "seed {seed} (OpenQASM 2): {diff}");
            }
        }
        // Multi-qubit custom unitaries have no OpenQASM form.
        let mut c = Circuit::new(2);
        let mut m = vec![C64::new(0.0, 0.0); 16];
        for i in 0..4 {
            m[i * 5] = C64::new(1.0, 0.0);
        }
        c.unitary(m, &[0, 1], &[]).unwrap();
        assert!(to_qasm3(&c, &[], false).is_err());
    }

    #[test]
    fn dynamic_programs_sample_and_branch() {
        // 13 measured coin flips: too many branches for the exact
        // distribution, so sampling runs shot by shot.
        let mut src = String::from("OPENQASM 3.0; include \"stdgates.inc\"; qubit q; bit[13] c;\n");
        for k in 0..13 {
            src.push_str(&format!("reset q; h q; c[{k}] = measure q;\n"));
        }
        let p = Program::from_qasm(&src).unwrap();
        assert!(p.probabilities().is_err());
        let counts = p.sample(4000, 1).unwrap();
        assert_eq!(counts.values().sum::<usize>(), 4000);
        let ones: usize = counts.iter().map(|(r, n)| (r & 1) as usize * n).sum();
        assert!((ones as f64 / 4000.0 - 0.5).abs() < 0.05);
        // Few branches: sampling follows the exact distribution.
        let bell = Program::from_qasm(
            "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;",
        )
        .unwrap();
        let d = bell.probabilities().unwrap();
        assert_eq!(d.keys().copied().collect::<Vec<_>>(), vec![0, 3]);
        let s = bell.sample(1000, 2).unwrap();
        assert_eq!(s.keys().copied().collect::<Vec<_>>(), vec![0, 3]);
        // Measuring a qubit twice, resets and else branches.
        let p = Program::from_qasm(
            "OPENQASM 3.0; include \"stdgates.inc\"; qubit q; bit[2] c;
             x q; c[0] = measure q; reset q; c[1] = measure q;
             if (c[0] == 1) { x q; } else { h q; }
             c[1] = measure q;",
        )
        .unwrap();
        assert_eq!(
            p.probabilities().unwrap().into_iter().collect::<Vec<_>>(),
            vec![(3, 1.0)]
        );
    }

    #[test]
    fn malformed_and_unsupported_programs_are_errors() {
        let head = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c;\n";
        for body in [
            "pow(0.5) @ x q[0];",
            "gate g a { x a; } gate g a { y a; }",
            "gate g a { g a; }",
            "gate g a { foo a; }",
            "nope q[0];",
            "cx q[0];",
            "cx q[0], q[0];",
            "rx q[0];",
            "c[0] = measure q;",
            "if (c == 4) x q[0];",
            "if (c[0] == 1) { c[0] = measure q[0]; x q[1]; } else { x q[0]; }",
            "if (c[0] == 1) { qubit r; }",
            "include \"other.inc\";",
            "x q[2];",
            "x q[0:5];",
            "x q[0:0:1];",
            "for uint i in [0:1] { x q[i]; }",
            "ctrl(0) @ x q[0], q[1];",
            "ctrl @ x q[0];",
            "c = 3;",
            "x q[0]",
            "/* unterminated",
            "rx(1/0) q[0];",
        ] {
            assert!(
                Program::from_qasm(&format!("{head}{body}")).is_err(),
                "{body}"
            );
        }
        assert!(Program::from_qasm("OPENQASM 4.0; qubit q;").is_err());
        assert!(Program::from_qasm("OPENQASM 3.0; qubit[100000] q;").is_err());
        // Negative indices count from the end; slices are inclusive.
        let p = Program::from_qasm(&format!("{head}x q[-1]; h q[0:1];")).unwrap();
        assert_eq!(p.len(), 3);
        // Exponential gate nesting is cut off by the operation budget.
        let mut src = String::from(
            "OPENQASM 3.0; include \"stdgates.inc\"; qubit q;\ngate g0 a { x a; x a; }\n",
        );
        for k in 1..40 {
            src.push_str(&format!("gate g{k} a {{ g{} a; g{} a; }}\n", k - 1, k - 1));
        }
        src.push_str("g39 q;\n");
        assert!(Program::from_qasm(&src).is_err());
    }

    /// Regressions of the review of the OpenQASM reader.
    #[test]
    fn review_regressions() {
        use std::collections::BTreeMap;
        let dist = |src: &str| Program::from_qasm(src).unwrap().probabilities().unwrap();
        let head = "OPENQASM 3.0; include \"stdgates.inc\";";
        // Long operator chains are errors, not stack overflows; moderate
        // ones still parse.
        let long = vec!["1"; 50_000].join("+");
        assert!(Program::from_qasm(&format!("{head} qubit q; rx({long}) q;")).is_err());
        let short = vec!["0.01"; 200].join("+");
        assert!(Program::from_qasm(&format!("{head} qubit q; rx({short}) q;")).is_ok());
        // Gates that expand to nothing still cost: 40 doubling levels of
        // `id` hit the budget instead of running for days.
        let mut src = String::from("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1];\n");
        src.push_str("gate g0 a { id a; id a; }\n");
        for k in 1..40 {
            src.push_str(&format!("gate g{k} a {{ g{} a; g{} a; }}\n", k - 1, k - 1));
        }
        src.push_str("g39 q[0];");
        let start = std::time::Instant::now();
        assert!(Program::from_qasm(&src).is_err());
        assert!(start.elapsed().as_secs() < 30);
        // Controls count towards the budget.
        let qs: Vec<String> = (0..64).map(|k| format!("q[{k}]")).collect();
        let src = format!(
            "{head} qubit[64] q; ctrl(63) @ pow(2097000) @ x {};",
            qs.join(", ")
        );
        assert!(Program::from_qasm(&src).is_err());
        // bool(c) is c != 0, compared with true or false.
        let cond = |c: &str| {
            dist(&format!(
                "{head} qubit[3] q; bit[3] c; x q[1]; c[0] = measure q[0]; c[1] = measure q[1]; if ({c}) x q[2]; c[2] = measure q[2];"
            ))
        };
        let taken: BTreeMap<u64, f64> = [(0b110, 1.0)].into_iter().collect();
        let not_taken: BTreeMap<u64, f64> = [(0b010, 1.0)].into_iter().collect();
        assert_eq!(cond("bool(c)"), taken);
        assert_eq!(cond("bool(c) == true"), taken);
        assert_eq!(cond("bool(c) != false"), taken);
        assert_eq!(cond("bool(c) == false"), not_taken);
        assert_eq!(cond("c[1] == true"), taken);
        assert_eq!(cond("uint[3](c) == 2"), taken);
        for bad in [
            "bool(c) == 1",
            "c == true",
            "int[2](c) == 1",
            "uint[4](c) == 2",
            "int[3](c) == 5",
        ] {
            assert!(
                Program::from_qasm(&format!("{head} qubit q; bit[3] c; if ({bad}) x q;")).is_err(),
                "{bad}"
            );
        }
        // Negative slice ends count from the end: q[0:-1] is every qubit.
        let all: BTreeMap<u64, f64> = [(0b111, 1.0)].into_iter().collect();
        assert_eq!(
            dist(&format!(
                "{head} qubit[3] q; bit[3] c; x q[0:-1]; c = measure q;"
            )),
            all
        );
        // Gate parameters shadow the constants.
        let p = dist(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; gate g(euler) a { ry(euler) a; } g(0) q[0]; measure q[0] -> c[0];",
        );
        assert_eq!(p.into_iter().collect::<Vec<_>>(), vec![(0, 1.0)]);
        // Integers above 2⁵³ are exact.
        let big = Program::from_qasm(&format!(
            "{head} qubit q; bit[63] c; if (uint[63](c) == 9007199254740993) x q;"
        ))
        .unwrap();
        let Operation::If { condition, .. } = &big.operations()[0] else {
            panic!("a condition");
        };
        assert_eq!(condition.value, 9_007_199_254_740_993);
        let again = Program::from_qasm(&big.to_qasm3().unwrap()).unwrap();
        assert_eq!(again.operations(), big.operations());
        // Versions: 2.0 and 3.x only, and only on the first line.
        for v in ["2.5", "20", "4", "1.0"] {
            assert!(
                Program::from_qasm(&format!("OPENQASM {v}; qubit q;")).is_err(),
                "{v}"
            );
        }
        assert!(Program::from_qasm("OPENQASM 3.1; qubit q;").is_ok());
        assert!(Program::from_qasm("OPENQASM 3.0; qubit q; OPENQASM 2.0;").is_err());
        // OpenQASM 3 keywords are names in OpenQASM 2; exports rename
        // registers that collide with `q` or with keywords.
        let v2 = "OPENQASM 2.0; include \"qelib1.inc\"; gate box a { x a; } qreg a[2]; creg q[1]; creg output[1]; box a[0]; measure a[0] -> q[0]; measure a[1] -> output[0];";
        let p = Program::from_qasm(v2).unwrap();
        let d = p.probabilities().unwrap();
        for text in [p.to_qasm2().unwrap(), p.to_qasm3().unwrap()] {
            let back = Program::from_qasm(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
            assert_eq!(back.probabilities().unwrap(), d, "{text}");
        }
    }
}
