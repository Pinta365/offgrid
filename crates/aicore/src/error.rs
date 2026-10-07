use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("failed to load model: {0}")]
    ModelLoad(PathBuf),
    #[error("failed to create inference context")]
    ContextCreate,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("prompt needs {needed} tokens but the context holds {n_ctx}")]
    ContextOverflow { needed: usize, n_ctx: usize },
    #[error("chat template failed: {0}")]
    ChatTemplate(String),
    #[error("tokenization failed")]
    Tokenize,
    #[error("invalid grammar")]
    Grammar,
    #[error("llama_decode failed with code {0}")]
    Decode(i32),
    #[error("model does not produce embeddings")]
    NoEmbeddings,
}
