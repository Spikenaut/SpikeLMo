use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpikeLMoError {
    #[error("tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("dimension mismatch for matmul: [{m}×{k1}] × [{k2}×{n}]")]
    MatmulDim { m: usize, k1: usize, k2: usize, n: usize },

    #[error("index {index} out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds { axis: usize, index: usize, size: usize },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, SpikeLMoError>;
