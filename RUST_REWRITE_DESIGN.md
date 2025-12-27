# RIES Rust Rewrite Design

## Current C Architecture Analysis

### Key Data Structures (from ries.c)

```c
// Expression node stored in binary tree (64 bytes)
typedef struct expr {
  ries_val val;           // floating-point value
  ries_dif der;           // derivative (LHS only)
  ries_tgs tags;          // type tags (INT, RAT, ALG, etc.)
  struct expr *left, *up, *right;  // tree pointers
  s16 elen;               // expression length
  symbol sym[MAX_ELEN+1]; // postfix expression (21 symbols max)
} expr;

// Metastack for expression evaluation
typedef struct metastack {
  ries_val uv[MS_UV_MAX];   // undo values
  ries_dif udv[MS_UV_MAX];  // undo derivatives
  ries_tgs utg[MS_UV_MAX];  // undo tags
  ries_val s[MS_STK_MAX];   // value stack
  ries_dif ds[MS_STK_MAX];  // derivative stack
  ries_tgs tg[MS_STK_MAX];  // tag stack
  // ... more fields
} metastack;
```

### Algorithm Summary

1. **Bidirectional Search**: Generate LHS expressions (containing x) and RHS expressions (constants only), store in sorted binary tree by value
2. **Expression Generation**: Recursive enumeration of valid postfix expressions using "forms" (patterns like "aabacbc")
3. **Matching**: Scan tree to find LHS-RHS pairs where `|val_l - val_r| / |deriv_l|` is small
4. **Newton-Raphson**: Refine matches to get precise x values

### Current Bottlenecks

- Single-threaded (mentioned in comments as future work)
- Double precision only (15-16 digits)
- Memory-bound at high search levels
- Binary tree has poor cache locality

---

## Rust Rewrite Architecture

### Crate Dependencies

```toml
[dependencies]
rug = "1.24"              # Arbitrary precision (MPFR/GMP bindings)
rayon = "1.10"            # Data parallelism
hashbrown = "0.14"        # Fast hash maps
parking_lot = "0.12"      # Better mutexes
crossbeam = "0.8"         # Lock-free data structures
typed-arena = "2.0"       # Arena allocation
num-traits = "0.2"        # Generic numeric traits
clap = "4.5"              # CLI parsing

[features]
default = ["f64"]
f64 = []                  # Standard double precision
f128 = ["dep:f128"]       # 128-bit float (if available)
mpfr = ["rug"]            # Arbitrary precision
simd = ["wide"]           # SIMD batch evaluation
```

### Core Types

```rust
use rug::Float;
use std::sync::Arc;

/// Precision-generic value type
pub trait RiesFloat: Clone + Send + Sync + PartialOrd {
    fn from_f64(v: f64) -> Self;
    fn to_f64(&self) -> f64;
    fn precision_bits() -> u32;

    // Math operations
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn sqrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn sin_pi(&self) -> Self;  // sin(π * x)
    fn cos_pi(&self) -> Self;
    // ... etc
}

impl RiesFloat for f64 { /* ... */ }
impl RiesFloat for rug::Float { /* ... */ }

/// Symbol representing an operation or constant
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Symbol {
    // Constants (seft 'a')
    One = b'1', Two = b'2', Three = b'3', /* ... */ Nine = b'9',
    Pi = b'p', E = b'e', Phi = b'f', X = b'x',

    // Unary operators (seft 'b')
    Neg = b'n', Recip = b'r', Sqrt = b'q', Square = b's',
    Ln = b'l', Exp = b'E', SinPi = b'S', CosPi = b'C', TanPi = b'T',
    LambertW = b'W',

    // Binary operators (seft 'c')
    Add = b'+', Sub = b'-', Mul = b'*', Div = b'/',
    Pow = b'^', Root = b'v', Log = b'L', Atan2 = b'A',

    // Custom user-defined
    Custom(u8),
}

/// Number type classification
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NumType {
    Transcendental = 0,
    Liouvillian = 1,
    Elementary = 2,
    Algebraic = 3,
    Constructible = 4,
    Rational = 5,
    Integer = 6,
}

/// A symbolic expression in postfix notation
#[derive(Clone)]
pub struct Expression {
    symbols: smallvec::SmallVec<[Symbol; 24]>,  // Inline small expressions
    complexity: u16,
}

/// Evaluated expression with cached value
pub struct EvaluatedExpr<V: RiesFloat> {
    expr: Expression,
    value: V,
    derivative: V,      // d(expr)/dx
    num_type: NumType,
}
```

### Parallel Expression Generation

```rust
use rayon::prelude::*;

/// Form pattern for expression generation (e.g., "aabacbc")
pub struct Form {
    pattern: Vec<Seft>,  // Stack effect types
    min_complexity: u16,
    max_complexity: u16,
}

/// Generate all expressions matching a form, in parallel
pub fn generate_expressions<V: RiesFloat>(
    form: &Form,
    symbols: &SymbolTable,
    complexity_limit: u16,
) -> Vec<EvaluatedExpr<V>> {
    // Split work by first symbol choice
    let first_symbols: Vec<Symbol> = symbols.get_seft_a().collect();

    first_symbols.par_iter()
        .flat_map(|&first_sym| {
            let mut results = Vec::new();
            generate_recursive(
                form, symbols, complexity_limit,
                &[first_sym], 0,
                &mut results,
            );
            results
        })
        .collect()
}
```

### Lock-Free Expression Database

```rust
use crossbeam::skiplist::SkipMap;
use ordered_float::OrderedFloat;

/// Thread-safe expression database sorted by value
pub struct ExpressionDatabase<V: RiesFloat> {
    // LHS expressions (contain x) sorted by value
    lhs: SkipMap<OrderedFloat<f64>, Arc<EvaluatedExpr<V>>>,
    // RHS expressions (constants) sorted by value
    rhs: SkipMap<OrderedFloat<f64>, Arc<EvaluatedExpr<V>>>,
}

impl<V: RiesFloat> ExpressionDatabase<V> {
    /// Insert expression, checking for matches in parallel
    pub fn insert_and_match(
        &self,
        expr: EvaluatedExpr<V>,
        is_lhs: bool,
        match_threshold: f64,
    ) -> Vec<Match<V>> {
        let value = expr.value.to_f64();
        let expr = Arc::new(expr);

        // Insert into appropriate side
        let (self_tree, other_tree) = if is_lhs {
            self.lhs.insert(OrderedFloat(value), expr.clone());
            (&self.lhs, &self.rhs)
        } else {
            self.rhs.insert(OrderedFloat(value), expr.clone());
            (&self.rhs, &self.lhs)
        };

        // Search for matches in the other tree
        let range_start = OrderedFloat(value - match_threshold);
        let range_end = OrderedFloat(value + match_threshold);

        other_tree.range(range_start..=range_end)
            .filter_map(|entry| {
                let other = entry.value();
                self.check_match(&expr, other, is_lhs)
            })
            .collect()
    }
}
```

### SIMD Batch Evaluation (Optional)

```rust
#[cfg(feature = "simd")]
use wide::f64x4;

/// Evaluate 4 expressions simultaneously using SIMD
pub fn eval_batch_simd(
    exprs: &[Expression; 4],
    x_values: f64x4,
) -> [f64x4; 4] {
    // Vectorized stack machine
    let mut stacks: [Vec<f64x4>; 4] = Default::default();

    // Process symbols in lockstep where possible
    // Fall back to scalar for divergent branches
    // ...
}
```

### Memory-Efficient Expression Storage

```rust
use typed_arena::Arena;

/// Arena-allocated expression storage for cache efficiency
pub struct ExpressionArena {
    // Expressions stored contiguously
    arena: Arena<EvaluatedExpr<f64>>,
    // Indices for tree structure (more cache-friendly than pointers)
    tree_indices: Vec<(u32, u32, u32)>,  // (left, parent, right)
}
```

### High-Precision Mode

```rust
use rug::{Float, float::Round};

/// Arbitrary precision configuration
pub struct HighPrecisionConfig {
    precision_bits: u32,  // e.g., 256 for ~77 decimal digits
}

impl RiesFloat for Float {
    fn precision_bits() -> u32 { 256 }  // Configurable

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn sin_pi(&self) -> Self {
        let pi = Float::with_val(Self::precision_bits(), rug::float::Constant::Pi);
        (self.clone() * pi).sin()
    }
    // ... etc
}

/// Run search with arbitrary precision
pub fn search_high_precision(
    target: &str,  // Parse from string for full precision
    config: &SearchConfig,
) -> Vec<Match<Float>> {
    let target = Float::parse(target).unwrap()
        .complete(config.precision_bits);

    // Same algorithm, different numeric type
    search_generic::<Float>(target, config)
}
```

### CLI Interface

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "ries-rs")]
#[command(about = "Find algebraic equations given their solution")]
pub struct Cli {
    /// Target value to find equations for
    pub target: String,

    /// Search level (each increment = ~10x more equations)
    #[arg(short = 'l', default_value = "2")]
    pub level: f32,

    /// Use high precision (specify bits, e.g., 256)
    #[arg(long, value_name = "BITS")]
    pub precision: Option<u32>,

    /// Number of threads (0 = auto)
    #[arg(short = 'j', default_value = "0")]
    pub threads: usize,

    /// Restrict to algebraic solutions
    #[arg(short = 'a')]
    pub algebraic: bool,

    /// Symbols to exclude
    #[arg(short = 'N')]
    pub exclude_symbols: Option<String>,

    // ... more options
}
```

---

## Performance Projections

### Expected Speedups

| Optimization | Estimated Speedup |
|--------------|-------------------|
| Parallelization (8 cores) | 5-7x |
| Better cache locality | 1.5-2x |
| Lock-free data structures | 1.2-1.5x |
| SIMD evaluation (4-wide) | 2-3x (eval only) |
| **Combined** | **~15-30x** |

### Precision Improvements

| Mode | Decimal Digits | Memory Overhead |
|------|----------------|-----------------|
| f64 (current) | ~15 | 1x |
| f128 | ~33 | 2x |
| MPFR 128-bit | ~38 | 3x |
| MPFR 256-bit | ~77 | 5x |
| MPFR 512-bit | ~154 | 9x |

---

## Implementation Phases

### Phase 1: Core Port (2-3 weeks)
- [ ] Expression representation and evaluation
- [ ] Symbol table and configuration
- [ ] Basic search algorithm (single-threaded)
- [ ] CLI with basic options
- [ ] Test suite matching C version output

### Phase 2: Parallelization (1-2 weeks)
- [ ] Parallel expression generation
- [ ] Lock-free expression database
- [ ] Work-stealing match checking
- [ ] Benchmark and tune

### Phase 3: High Precision (1 week)
- [ ] Generic numeric trait implementation
- [ ] MPFR integration via `rug`
- [ ] Precision configuration
- [ ] String parsing for arbitrary precision input

### Phase 4: Optimizations (1-2 weeks)
- [ ] Arena allocation
- [ ] SIMD batch evaluation
- [ ] Profile-guided optimization
- [ ] Memory mapping for huge searches

### Phase 5: Feature Parity (ongoing)
- [ ] All original options (-s, -x, -F formats, etc.)
- [ ] Profile file support
- [ ] User-defined constants and functions
- [ ] Output format compatibility

---

## Quick Start Prototype

```rust
// src/main.rs - Minimal working example

use rug::Float;

fn main() {
    let target: f64 = std::env::args()
        .nth(1)
        .expect("Usage: ries-rs <number>")
        .parse()
        .expect("Invalid number");

    println!("Searching for equations with solution near {}", target);

    // Simple brute-force search for demonstration
    for a in 1..=9 {
        for b in 1..=9 {
            // x = a + b
            let x = (a + b) as f64;
            if (x - target).abs() < 0.01 {
                println!("x = {} + {} (x = {}, error = {})", a, b, x, x - target);
            }
            // x = a * b
            let x = (a * b) as f64;
            if (x - target).abs() < 0.01 {
                println!("x = {} * {} (x = {}, error = {})", a, b, x, x - target);
            }
            // ... more operations
        }
    }
}
```

---

## Conclusion

A Rust rewrite offers significant advantages:

1. **~15-30x speedup** through parallelization and better memory layout
2. **Arbitrary precision** via MPFR, enabling 100+ digit accuracy
3. **Memory safety** eliminating buffer overflows and undefined behavior
4. **Modern tooling** (cargo, clippy, rustfmt, criterion benchmarks)
5. **Easy cross-compilation** to Windows, Linux, macOS, WASM

The modular design allows incremental development while maintaining compatibility with the original C version's output format.
