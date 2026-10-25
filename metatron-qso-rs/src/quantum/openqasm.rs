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
//! `U(θ, φ, λ)` carries the phase `e^{iθ/2}` relative to OpenQASM 2's `U`,
//! and `gphase` and the phases of `u2`/`u3` become relative phases under
//! `ctrl @`. QSO keeps these phases while expanding gates and drops the
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
/// Deepest nesting of expressions, blocks and gate calls.
const MAX_DEPTH: usize = 64;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(QsoError::Parse(msg.into()))
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64, bool), // value, written as an integer
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
            out.push((Tok::Num(v, integer), line));
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
            Expr::Var(name) => match name.as_str() {
                "pi" | "π" => PI,
                "tau" | "τ" => 2.0 * PI,
                "euler" | "ℇ" => std::f64::consts::E,
                _ => *env
                    .get(name)
                    .ok_or_else(|| QsoError::Parse(format!("unknown identifier '{name}'")))?,
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

#[derive(Clone, Debug)]
struct Cond {
    operand: Operand,
    equal: bool,
    value: u64,
}

#[derive(Clone, Debug)]
enum Stmt {
    Version(String),
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
            Some(&Tok::Num(v, true)) if v >= 0.0 && v <= u64::MAX as f64 => {
                self.pos += 1;
                Ok(v as u64)
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

    // Expressions: sum → product → unary → power → atom.
    fn expr(&mut self) -> Result<Expr> {
        self.deeper(|p| {
            let mut e = p.product()?;
            loop {
                if p.eat("+") {
                    e = Expr::Bin('+', Box::new(e), Box::new(p.product()?));
                } else if p.eat("-") {
                    e = Expr::Bin('-', Box::new(e), Box::new(p.product()?));
                } else {
                    return Ok(e);
                }
            }
        })
    }

    fn product(&mut self) -> Result<Expr> {
        let mut e = self.unary()?;
        loop {
            if self.eat("*") {
                e = Expr::Bin('*', Box::new(e), Box::new(self.unary()?));
            } else if self.eat("/") {
                e = Expr::Bin('/', Box::new(e), Box::new(self.unary()?));
            } else {
                return Ok(e);
            }
        }
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.deeper(Self::unary)?)));
        }
        if self.eat("+") {
            return self.deeper(Self::unary);
        }
        let base = self.atom()?;
        if self.eat("^") || self.eat("**") {
            let exp = self.deeper(Self::unary)?;
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
        let cast = matches!(self.peek(), Some(Tok::Ident(n)) if ["int", "uint", "bool"].contains(&n.as_str()));
        let operand = if cast {
            self.pos += 1;
            if self.eat("[") {
                self.integer()?;
                self.expect("]")?;
            }
            self.expect("(")?;
            let o = self.operand()?;
            self.expect(")")?;
            o
        } else {
            self.operand()?
        };
        let (equal, value) = if self.eat("==") {
            (true, self.cond_value()?)
        } else if self.eat("!=") {
            (false, self.cond_value()?)
        } else {
            (false, 0) // truthy: value != 0
        };
        Ok(Cond {
            operand,
            equal,
            value,
        })
    }

    fn cond_value(&mut self) -> Result<u64> {
        if self.is_ident("true") {
            self.pos += 1;
            return Ok(1);
        }
        if self.is_ident("false") {
            self.pos += 1;
            return Ok(0);
        }
        self.integer()
    }

    fn statement(&mut self, version: u8) -> Result<Stmt> {
        let line = self.line();
        let Some(tok) = self.peek().cloned() else {
            return self.fail("unexpected end of program");
        };
        let Tok::Ident(word) = tok else {
            return self.fail("expected a statement");
        };
        if UNSUPPORTED.contains(&word.as_str()) {
            return self.fail(format!("OpenQASM feature '{word}' is not supported"));
        }
        match word.as_str() {
            "OPENQASM" => {
                self.pos += 1;
                let v = match self.peek().cloned() {
                    Some(Tok::Num(v, _)) => {
                        self.pos += 1;
                        format!("{v}")
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
                    arity(1, 0)?;
                    vec![]
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
                    self.spend(base.len().saturating_mul(reps))?;
                    let mut out = Vec::with_capacity(base.len() * reps);
                    for _ in 0..reps {
                        out.extend(base.iter().cloned());
                    }
                    out
                }
                Modifier::Ctrl(_, positive) => {
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
                self.spend(prims.len())?;
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
                let (a, b) = (int(a)?, int(b)?);
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
                Stmt::Version(_) => {
                    if !top {
                        return err("'OPENQASM' must come first");
                    }
                }
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
                        self.spend(prims.len())?;
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
                    if bits.len() > 64 {
                        return err("conditions read at most 64 bits");
                    }
                    if bits.len() < 64 && cond.value >> bits.len() != 0 {
                        return err(format!(
                            "line {}: {} does not fit into {} bit(s)",
                            cond.operand.line,
                            cond.value,
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
                        value: cond.value,
                        equal: cond.equal,
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
    };
    // The version decides the meaning of `U`, `u2` and `u3`.
    let mut version = 3u8;
    if parser.is_ident("OPENQASM") {
        let line = parser.line();
        if let Stmt::Version(v) = parser.statement(3)? {
            version = match v.chars().next() {
                Some('2') => 2,
                Some('3') => 3,
                _ => return err(format!("line {line}: unsupported OpenQASM version {v}")),
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

impl Program {
    fn qubit_name(&self, q: usize) -> String {
        let r = self
            .qubit_registers()
            .iter()
            .find(|r| q >= r.offset && q < r.offset + r.size)
            .expect("qubit in a register");
        format!("{}[{}]", r.name, q - r.offset)
    }

    fn bit_name(&self, b: usize) -> String {
        let r = self
            .bit_registers()
            .iter()
            .find(|r| b >= r.offset && b < r.offset + r.size)
            .expect("bit in a register");
        format!("{}[{}]", r.name, b - r.offset)
    }

    /// The register whose bits are exactly `bits`, in order.
    fn register_of(&self, bits: &[usize]) -> Option<&Register> {
        self.bit_registers().iter().find(|r| {
            r.size == bits.len() && bits.iter().enumerate().all(|(k, &b)| b == r.offset + k)
        })
    }

    fn gate3(&self, out: &mut String, inst: &Instruction) -> Result<()> {
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
            .map(|&q| self.qubit_name(q))
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

    fn ops3(&self, out: &mut String, ops: &[Operation], indent: usize) -> Result<()> {
        let pad = "  ".repeat(indent);
        for op in ops {
            match op {
                Operation::Gate(i) => {
                    out.push_str(&pad);
                    self.gate3(out, i)?;
                }
                Operation::Measure { qubit, bit } => {
                    let _ = writeln!(
                        out,
                        "{pad}{} = measure {};",
                        self.bit_name(*bit),
                        self.qubit_name(*qubit)
                    );
                }
                Operation::Reset { qubit } => {
                    let _ = writeln!(out, "{pad}reset {};", self.qubit_name(*qubit));
                }
                Operation::If { condition, body } => {
                    let lhs = match self.register_of(&condition.bits) {
                        Some(r) if r.size == 1 => r.name.clone(),
                        Some(r) => format!("int[{}]({})", r.size, r.name),
                        None if condition.bits.len() == 1 => self.bit_name(condition.bits[0]),
                        None => {
                            return Err(QsoError::invalid(
                                "program",
                                "conditions must read one bit or a whole register",
                            ));
                        }
                    };
                    let op = if condition.equal { "==" } else { "!=" };
                    let _ = writeln!(out, "{pad}if ({lhs} {op} {}) {{", condition.value);
                    self.ops3(out, body, indent + 1)?;
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
        for r in self.qubit_registers() {
            let _ = writeln!(out, "qubit[{}] {};", r.size, r.name);
        }
        for r in self.bit_registers() {
            let _ = writeln!(out, "bit[{}] {};", r.size, r.name);
        }
        self.ops3(&mut out, self.operations(), 0)?;
        Ok(out)
    }

    /// The program as OpenQASM 2.0: one quantum register `q`, the
    /// classical registers of the program. Conditions must compare a whole
    /// register for equality and may not be nested, as OpenQASM 2 requires.
    pub fn to_qasm2(&self) -> Result<String> {
        let n = self.n_qubits();
        let mut out = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
        let _ = writeln!(out, "qreg q[{n}];");
        for r in self.bit_registers() {
            let _ = writeln!(out, "creg {}[{}];", r.name, r.size);
        }
        for op in self.operations() {
            self.op2(&mut out, op, None)?;
        }
        Ok(out)
    }

    fn op2(&self, out: &mut String, op: &Operation, guard: Option<&str>) -> Result<()> {
        let mut text = String::new();
        match op {
            Operation::Gate(i) => emit_instruction(&mut text, i, &i.angles(&[]), self.n_qubits())?,
            Operation::Measure { qubit, bit } => {
                let _ = writeln!(text, "measure q[{qubit}] -> {};", self.bit_name(*bit));
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
                let g = format!("if({}=={}) ", reg.name, condition.value);
                for o in body {
                    self.op2(out, o, Some(&g))?;
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
}
