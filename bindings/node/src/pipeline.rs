extern crate tokenizers as tk;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tk::tokenizer::pipeline::PipelineTokenizer as Pipeline;

fn err<E: std::fmt::Display>(e: E) -> Error {
  Error::from_reason(format!("{e}"))
}

#[napi]
pub struct PipelineTokenizer(Pipeline);

#[napi]
impl PipelineTokenizer {
  #[napi(factory)]
  pub fn from_file(path: String) -> Result<Self> {
    let tok = tk::Tokenizer::from_file(path).map_err(err)?;
    Ok(Self(Pipeline::try_from(&tok).map_err(err)?))
  }

  /// `Uint32Array`, not `Vec<u32>`: a JS `Array` costs one napi value per token, which
  /// on token-dense input is 13× the encode itself (gpt2 chinese 31 vs 616 MB/s).
  #[napi]
  pub fn encode(&self, text: String, add_special_tokens: Option<bool>) -> Result<Uint32Array> {
    let ids = self
      .0
      .encode(&text, add_special_tokens.unwrap_or(true))
      .map_err(err)?;
    Ok(Uint32Array::new(ids.iter().map(|t| t.id).collect()))
  }

  /// Drops the two remaining per-call costs of `encode`: the JS string → UTF-8 copy
  /// (as fast as the tokenizer itself, so it halves throughput) and the fresh
  /// ArrayBuffer (388 ns of a 789 ns call). Returns how many ids were written.
  #[napi]
  pub fn encode_bytes_into(
    &self,
    text: &[u8],
    mut out: Uint32Array,
    add_special_tokens: Option<bool>,
  ) -> Result<u32> {
    let text = std::str::from_utf8(text).map_err(err)?;
    let ids = self
      .0
      .encode(text, add_special_tokens.unwrap_or(true))
      .map_err(err)?;
    // SAFETY: JS is blocked for this synchronous call, so nothing else aliases `out`.
    let dst = unsafe { out.as_mut() };
    if ids.len() > dst.len() {
      return Err(err(format!(
        "need {} ids, buffer holds {}",
        ids.len(),
        dst.len()
      )));
    }
    for (d, t) in dst.iter_mut().zip(&ids) {
      *d = t.id;
    }
    Ok(ids.len() as u32)
  }
}
