//! # spike-lmo
//!
//! **SNN/LLM Fusion Framework** — the core engine for SpikeLMo.
//!
//! This crate provides the mathematical and architectural building blocks for
//! fusing Spikenaut's spiking neural networks with large language models.
//!
//! ## Key modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`fusion`] | `SnnLlmFusion` — the SNN ↔ LLM gating layer |
//! | [`tensor`] | Candle-free `Tensor` type + ops |
//! | [`training`] | `FusionTrainer` — E-prop / OTTT learning loop |
//! | [`transformer`] | Transformer building blocks |
//! | [`error`] | `SpikeLmoError` unified error type |

pub mod tensor;
pub mod transformer;
pub mod fusion;
pub mod training;
pub mod error;

// ── Convenient top-level re-exports ──────────────────────────────────────────

pub use fusion::{SnnLlmFusion, FusionConfig, GateMode};
pub use tensor::Tensor;
pub use training::{FusionTrainer, TrainingConfig};
pub use error::SpikeLmoError;
