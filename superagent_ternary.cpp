#include "superagent_ternary.h"

namespace superagent {

static inline float fast_gelu(float x) {
  return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

static inline float ternary_quant_val(float x, float threshold) {
  if (x > threshold) {
    return 1.0f;
  }
  if (x < -threshold) {
    return -1.0f;
  }
  return 0.0f;
}

void TernaryLinear::set_weight_ternary(const std::vector<float> &w, float threshold) {
  if (w.size() != weight.size()) {
    return;
  }
  for (size_t i = 0; i < w.size(); ++i) {
    const float val = w[i];
    if (val > threshold) {
      weight[i] = 1;
    } else if (val < -threshold) {
      weight[i] = -1;
    } else {
      weight[i] = 0;
    }
  }
}

void TernaryLinear::forward(const Tensor3 &x, Tensor3 &out) const {
  out = Tensor3(x.B, x.S, out_f);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      for (int o = 0; o < out_f; ++o) {
        float acc = bias.empty() ? 0.0f : bias[static_cast<size_t>(o)];
        const size_t w_base = static_cast<size_t>(o) * in_f;
        for (int i = 0; i < in_f; ++i) {
          const int8_t w = weight[w_base + static_cast<size_t>(i)];
          const float x_val = quantize_act ? ternary_quant_val(x.at(b, s, i), threshold) : x.at(b, s, i);
          if (w == 1) {
            acc += x_val;
          } else if (w == -1) {
            acc -= x_val;
          }
        }
        out.at(b, s, o) = acc;
      }
    }
  }
}

void IntRMSNorm::forward(const Tensor3 &x, Tensor3 &out) const {
  out = Tensor3(x.B, x.S, x.H);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      float mean_sq = 0.0f;
      for (int h = 0; h < x.H; ++h) {
        const float v = x.at(b, s, h);
        mean_sq += v * v;
      }
      mean_sq /= static_cast<float>(x.H);
      const float inv = 1.0f / std::sqrt(mean_sq + eps);
      for (int h = 0; h < x.H; ++h) {
        out.at(b, s, h) = x.at(b, s, h) * inv * weight[static_cast<size_t>(h)];
      }
    }
  }
}

void IntSoftmax::forward(const std::vector<float> &x, int rows, int cols, std::vector<float> &out) const {
  out.assign(static_cast<size_t>(rows) * cols, 0.0f);
  for (int r = 0; r < rows; ++r) {
    float max_v = -1e9f;
    for (int c = 0; c < cols; ++c) {
      max_v = std::max(max_v, x[static_cast<size_t>(r) * cols + c]);
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float v = x[static_cast<size_t>(r) * cols + c] - max_v;
      v = std::max(clamp_min, std::min(clamp_max, v));
      const float e = std::exp(v);
      out[static_cast<size_t>(r) * cols + c] = e;
      sum += e;
    }
    const float inv = 1.0f / (sum + 1e-8f);
    for (int c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r) * cols + c] *= inv;
    }
  }
}

static void mlstm_scan(const Tensor3 &q, const Tensor3 &k, const Tensor3 &v,
                       const Tensor3 &i_gate, const Tensor3 &f_gate,
                       std::vector<float> &C, std::vector<float> &n,
                       Tensor3 &h_out) {
  h_out = Tensor3(q.B, q.S, q.H);
  for (int b = 0; b < q.B; ++b) {
    for (int t = 0; t < q.S; ++t) {
      const float it = i_gate.at(b, t, 0);
      const float ft = f_gate.at(b, t, 0);
      // Update C and n
      for (int i = 0; i < q.H; ++i) {
        const float vt = v.at(b, t, i);
        const float kt = k.at(b, t, i);
        const size_t n_idx = (static_cast<size_t>(b) * q.H + i);
        n[n_idx] = ft * n[n_idx] + it * kt;
        for (int j = 0; j < q.H; ++j) {
          const size_t c_idx = (static_cast<size_t>(b) * q.H + i) * q.H + j;
          C[c_idx] = ft * C[c_idx] + it * vt * k.at(b, t, j);
        }
      }
      // num = C * q
      std::vector<float> num(static_cast<size_t>(q.H), 0.0f);
      for (int i = 0; i < q.H; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < q.H; ++j) {
          const size_t c_idx = (static_cast<size_t>(b) * q.H + i) * q.H + j;
          acc += C[c_idx] * q.at(b, t, j);
        }
        num[static_cast<size_t>(i)] = acc;
      }
      float den = 0.0f;
      for (int i = 0; i < q.H; ++i) {
        const size_t n_idx = (static_cast<size_t>(b) * q.H + i);
        den += n[n_idx] * q.at(b, t, i);
      }
      den = std::max(std::abs(den), 1.0f);
      for (int i = 0; i < q.H; ++i) {
        h_out.at(b, t, i) = num[static_cast<size_t>(i)] / den;
      }
    }
  }
}

void TernarymLSTMCell::forward(const Tensor3 &x, Tensor3 &h_out, std::vector<float> &C, std::vector<float> &n) const {
  Tensor3 qkv;
  W_qkv.forward(x, qkv);
  Tensor3 q(x.B, x.S, hidden_dim);
  Tensor3 k(x.B, x.S, hidden_dim);
  Tensor3 v(x.B, x.S, hidden_dim);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      for (int h = 0; h < hidden_dim; ++h) {
        q.at(b, s, h) = qkv.at(b, s, h);
        k.at(b, s, h) = qkv.at(b, s, h + hidden_dim);
        v.at(b, s, h) = qkv.at(b, s, h + 2 * hidden_dim);
      }
    }
  }
  Tensor3 gates;
  W_if.forward(x, gates);
  Tensor3 i_gate(x.B, x.S, 1);
  Tensor3 f_gate(x.B, x.S, 1);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      const float g0 = gates.at(b, s, 0);
      const float g1 = gates.at(b, s, 1);
      const float i = std::exp(-std::log1p(std::exp(-g0)));
      const float f = std::exp(-std::log1p(std::exp(-g1)));
      i_gate.at(b, s, 0) = i;
      f_gate.at(b, s, 0) = f;
    }
  }
  if (C.empty()) {
    C.assign(static_cast<size_t>(x.B) * hidden_dim * hidden_dim, 0.0f);
  }
  if (n.empty()) {
    n.assign(static_cast<size_t>(x.B) * hidden_dim, 0.0f);
  }
  mlstm_scan(q, k, v, i_gate, f_gate, C, n, h_out);
}

void TernaryLightningIndexer::forward(const Tensor3 &meta, std::vector<float> &scores) const {
  Tensor3 q;
  Tensor3 k;
  W_qI.forward(meta, q);
  W_kI.forward(meta, k);
  const int B = meta.B;
  const int N = meta.S;
  scores.assign(static_cast<size_t>(B) * N * N, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int h = 0; h < heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < h_dim; ++d) {
            const int q_idx = (h * h_dim + d);
            dot += q.at(b, i, q_idx) * k.at(b, j, d);
          }
          if (dot < 0.0f) {
            dot = 0.0f;
          }
          acc += dot * w[static_cast<size_t>(h)];
        }
        scores[(static_cast<size_t>(b) * N + i) * N + j] = acc;
      }
    }
  }
}

void TernaryDeepSeekSparseAttention::forward(const Tensor3 &x, Tensor3 &out_x) const {
  const int B = x.B;
  const int S = x.S;
  const int pad = (block_size - (S % block_size)) % block_size;
  const int S_p = S + pad;
  const int N = S_p / block_size;

  Tensor3 x_p(B, S_p, x.H);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < x.H; ++h) {
        x_p.at(b, s, h) = (s < S) ? x.at(b, s, h) : 0.0f;
      }
    }
  }

  Tensor3 meta(B, N, x.H);
  for (int b = 0; b < B; ++b) {
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < x.H; ++h) {
        float acc = 0.0f;
        for (int s = 0; s < block_size; ++s) {
          acc += x_p.at(b, n * block_size + s, h);
        }
        meta.at(b, n, h) = acc / static_cast<float>(block_size);
      }
    }
  }

  std::vector<float> scores;
  indexer.forward(meta, scores);

  std::vector<int> topk_idx(static_cast<size_t>(B) * N * topk, 0);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < N; ++i) {
      std::vector<std::pair<float, int>> row;
      row.reserve(N);
      for (int j = 0; j < N; ++j) {
        if (j > i) {
          row.emplace_back(-1e9f, j);
        } else {
          row.emplace_back(scores[(static_cast<size_t>(b) * N + i) * N + j], j);
        }
      }
      std::partial_sort(row.begin(), row.begin() + std::min(topk, N), row.end(),
                        [](const auto &a, const auto &b) { return a.first > b.first; });
      for (int k = 0; k < std::min(topk, N); ++k) {
        topk_idx[(static_cast<size_t>(b) * N + i) * topk + k] = row[static_cast<size_t>(k)].second;
      }
    }
  }

  Tensor3 qkv_out;
  qkv.forward(x_p, qkv_out);
  const int head_dim = H / heads;
  Tensor3 q(B, S_p, H);
  Tensor3 k(B, S_p, H);
  Tensor3 v(B, S_p, H);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < H; ++h) {
        q.at(b, s, h) = qkv_out.at(b, s, h);
        k.at(b, s, h) = qkv_out.at(b, s, h + H);
        v.at(b, s, h) = qkv_out.at(b, s, h + 2 * H);
      }
    }
  }

  Tensor3 out_full(B, S_p, H);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < H; ++h) {
        out_full.at(b, s, h) = 0.0f;
      }
    }
    const int local_window = std::max(0, local_window_blocks);
    for (int s = 0; s < S_p; ++s) {
      const int block_i = s / block_size;
      std::vector<int> allowed_blocks;
      allowed_blocks.reserve(topk + 2 * local_window + 1);
      for (int k_i = 0; k_i < std::min(topk, N); ++k_i) {
        allowed_blocks.push_back(topk_idx[(static_cast<size_t>(b) * N + block_i) * topk + k_i]);
      }
      for (int j = std::max(0, block_i - local_window); j <= std::min(N - 1, block_i + local_window); ++j) {
        allowed_blocks.push_back(j);
      }
      std::sort(allowed_blocks.begin(), allowed_blocks.end());
      allowed_blocks.erase(std::unique(allowed_blocks.begin(), allowed_blocks.end()), allowed_blocks.end());

      for (int h = 0; h < heads; ++h) {
        std::vector<float> attn_scores(static_cast<size_t>(S_p), -1e9f);
        for (int s2 = 0; s2 <= s; ++s2) {
          const int block_j = s2 / block_size;
          if (!std::binary_search(allowed_blocks.begin(), allowed_blocks.end(), block_j)) {
            continue;
          }
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            dot += q.at(b, s, h * head_dim + d) * k.at(b, s2, h * head_dim + d);
          }
          attn_scores[static_cast<size_t>(s2)] = dot / std::sqrt(static_cast<float>(head_dim));
        }
        std::vector<float> probs;
        softmax.forward(attn_scores, 1, S_p, probs);
        for (int s2 = 0; s2 <= s; ++s2) {
          const float p = probs[static_cast<size_t>(s2)];
          if (p == 0.0f) {
            continue;
          }
          for (int d = 0; d < head_dim; ++d) {
            out_full.at(b, s, h * head_dim + d) += p * v.at(b, s2, h * head_dim + d);
          }
        }
      }
    }
  }

  Tensor3 projected;
  out.forward(out_full, projected);
  out_x = Tensor3(B, S, H);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < H; ++h) {
        out_x.at(b, s, h) = projected.at(b, s, h);
      }
    }
  }
}

void TernaryFeedForward::forward(const Tensor3 &x, Tensor3 &out) const {
  Tensor3 hidden;
  fc.forward(x, hidden);
  for (int b = 0; b < hidden.B; ++b) {
    for (int s = 0; s < hidden.S; ++s) {
      for (int h = 0; h < hidden.H; ++h) {
        hidden.at(b, s, h) = fast_gelu(hidden.at(b, s, h));
      }
    }
  }
  out = hidden;
}

void LatentMemory::write(const Tensor3 &latents) {
  Tensor3 gist(latents.B, 1, latents.H);
  for (int b = 0; b < latents.B; ++b) {
    for (int h = 0; h < latents.H; ++h) {
      float acc = 0.0f;
      for (int s = 0; s < latents.S; ++s) {
        acc += latents.at(b, s, h);
      }
      gist.at(b, 0, h) = acc / static_cast<float>(latents.S);
    }
  }
  for (int b = 0; b < gist.B; ++b) {
    for (int h = 0; h < dim; ++h) {
      bank[static_cast<size_t>(ptr) * dim + h] = gist.at(b, 0, h);
    }
    ptr = (ptr + 1) % capacity;
    if (ptr == 0) {
      full = true;
    }
  }
}

void LatentMemory::read_and_inject(const Tensor3 &x, Tensor3 &out) const {
  const int mem_size = full ? capacity : ptr;
  if (mem_size == 0) {
    out = x;
    return;
  }
  out = Tensor3(x.B, x.S, x.H);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      std::vector<float> weights(static_cast<size_t>(mem_size), 0.0f);
      float max_v = -1e9f;
      for (int m = 0; m < mem_size; ++m) {
        float dot = 0.0f;
        for (int h = 0; h < x.H; ++h) {
          dot += x.at(b, s, h) * bank[static_cast<size_t>(m) * dim + h];
        }
        dot /= std::sqrt(static_cast<float>(x.H));
        weights[static_cast<size_t>(m)] = dot;
        max_v = std::max(max_v, dot);
      }
      float sum = 0.0f;
      for (int m = 0; m < mem_size; ++m) {
        weights[static_cast<size_t>(m)] = std::exp(weights[static_cast<size_t>(m)] - max_v);
        sum += weights[static_cast<size_t>(m)];
      }
      const float inv = 1.0f / (sum + 1e-8f);
      for (int m = 0; m < mem_size; ++m) {
        weights[static_cast<size_t>(m)] *= inv;
      }
      std::vector<float> retrieved(static_cast<size_t>(x.H), 0.0f);
      for (int h = 0; h < x.H; ++h) {
        float acc = 0.0f;
        for (int m = 0; m < mem_size; ++m) {
          acc += weights[static_cast<size_t>(m)] * bank[static_cast<size_t>(m) * dim + h];
        }
        retrieved[static_cast<size_t>(h)] = acc;
      }
      for (int h = 0; h < x.H; ++h) {
        float gate = gate_bias[static_cast<size_t>(h)];
        const size_t w_base = static_cast<size_t>(h) * dim * 2;
        for (int d = 0; d < x.H; ++d) {
          gate += gate_weight[w_base + static_cast<size_t>(d)] * x.at(b, s, d);
          gate += gate_weight[w_base + static_cast<size_t>(dim) + d] * retrieved[static_cast<size_t>(d)];
        }
        gate = 1.0f / (1.0f + std::exp(-gate));
        out.at(b, s, h) = x.at(b, s, h) + gate * retrieved[static_cast<size_t>(h)];
      }
    }
  }
}

SuperAgent::SuperAgent(const TernaryConfig &config)
    : cfg(config),
      byte_embed(static_cast<size_t>(cfg.vocab_size) * cfg.byte_embed_dim, 0.0f),
      encoder(cfg.byte_embed_dim, cfg.enc_mlstm_dim, cfg.threshold_w),
      enc_proj(cfg.enc_mlstm_dim * cfg.compression_rate, cfg.hidden_dim, true, true, cfg.threshold_w),
      memory(cfg.hidden_dim, cfg.memory_capacity),
      ctx_proj(cfg.hidden_dim, cfg.byte_embed_dim, true, true, cfg.threshold_w),
      decoder(cfg.byte_embed_dim * 2, cfg.dec_mlstm_dim, cfg.threshold_w),
      head(cfg.dec_mlstm_dim, cfg.vocab_size, true, true, cfg.threshold_w),
      global_bos(static_cast<size_t>(cfg.hidden_dim), 0.0f) {
  ln1.reserve(cfg.physical_depth);
  ln2.reserve(cfg.physical_depth);
  attn.reserve(cfg.physical_depth);
  ff.reserve(cfg.physical_depth);
  for (int i = 0; i < cfg.physical_depth; ++i) {
    ln1.emplace_back(cfg.hidden_dim);
    ln2.emplace_back(cfg.hidden_dim);
    attn.emplace_back(cfg);
    ff.emplace_back(cfg.hidden_dim, cfg.threshold_w);
  }
}

void SuperAgent::forward(const std::vector<uint8_t> &byte_input, int B, int S, Tensor3 &logits, Tensor3 &x_glob, bool update_memory) {
  const int r = cfg.compression_rate;
  Tensor3 x_emb(B, S, cfg.byte_embed_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      const uint8_t idx = byte_input[static_cast<size_t>(b) * S + s];
      const size_t base = static_cast<size_t>(idx) * cfg.byte_embed_dim;
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        x_emb.at(b, s, h) = byte_embed[base + static_cast<size_t>(h)];
      }
    }
  }

  Tensor3 enc_feats;
  std::vector<float> C_enc;
  std::vector<float> n_enc;
  encoder.forward(x_emb, enc_feats, C_enc, n_enc);

  const int pad = (r - (S % r)) % r;
  const int S_p = S + pad;
  Tensor3 enc_p(B, S_p, cfg.enc_mlstm_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < cfg.enc_mlstm_dim; ++h) {
        enc_p.at(b, s, h) = (s < S) ? enc_feats.at(b, s, h) : 0.0f;
      }
    }
  }

  Tensor3 latents(B, S_p / r, cfg.enc_mlstm_dim * r);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; s += r) {
      const int out_s = s / r;
      for (int h = 0; h < cfg.enc_mlstm_dim; ++h) {
        for (int k = 0; k < r; ++k) {
          latents.at(b, out_s, h * r + k) = enc_p.at(b, s + k, h);
        }
      }
    }
  }

  Tensor3 latents_proj;
  enc_proj.forward(latents, latents_proj);

  Tensor3 injected;
  memory.read_and_inject(latents_proj, injected);
  x_glob = injected;
  for (int loop = 0; loop < cfg.logical_loops; ++loop) {
    for (int i = 0; i < cfg.physical_depth; ++i) {
      Tensor3 norm1;
      ln1[static_cast<size_t>(i)].forward(x_glob, norm1);
      Tensor3 attn_out;
      attn[static_cast<size_t>(i)].forward(norm1, attn_out);
      for (int b = 0; b < x_glob.B; ++b) {
        for (int s = 0; s < x_glob.S; ++s) {
          for (int h = 0; h < x_glob.H; ++h) {
            x_glob.at(b, s, h) += attn_out.at(b, s, h);
          }
        }
      }
      Tensor3 norm2;
      ln2[static_cast<size_t>(i)].forward(x_glob, norm2);
      Tensor3 ff_out;
      ff[static_cast<size_t>(i)].forward(norm2, ff_out);
      for (int b = 0; b < x_glob.B; ++b) {
        for (int s = 0; s < x_glob.S; ++s) {
          for (int h = 0; h < x_glob.H; ++h) {
            x_glob.at(b, s, h) += ff_out.at(b, s, h);
          }
        }
      }
    }
  }
  if (update_memory) {
    memory.write(x_glob);
  }

  Tensor3 ctx_proj_out;
  ctx_proj.forward(x_glob, ctx_proj_out);
  std::vector<float> bos_proj(static_cast<size_t>(cfg.byte_embed_dim), 0.0f);
  for (int h = 0; h < cfg.byte_embed_dim; ++h) {
    float acc = ctx_proj.bias.empty() ? 0.0f : ctx_proj.bias[static_cast<size_t>(h)];
    const size_t w_base = static_cast<size_t>(h) * ctx_proj.in_f;
    for (int i = 0; i < ctx_proj.in_f; ++i) {
      const int8_t w = ctx_proj.weight[w_base + static_cast<size_t>(i)];
      if (w == 1) {
        acc += global_bos[static_cast<size_t>(i)];
      } else if (w == -1) {
        acc -= global_bos[static_cast<size_t>(i)];
      }
    }
    bos_proj[static_cast<size_t>(h)] = acc;
  }
  Tensor3 ctx_byte(B, S_p, cfg.byte_embed_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < x_glob.S; ++s) {
      const bool use_bos = (s == 0);
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        const float v = use_bos ? bos_proj[static_cast<size_t>(h)] : ctx_proj_out.at(b, s - 1, h);
        for (int k = 0; k < r; ++k) {
          ctx_byte.at(b, s * r + k, h) = v;
        }
      }
    }
  }

  Tensor3 x_emb_p(B, S_p, cfg.byte_embed_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        x_emb_p.at(b, s, h) = (s < S) ? x_emb.at(b, s, h) : 0.0f;
      }
    }
  }

  Tensor3 dec_in(B, S_p, cfg.byte_embed_dim * 2);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        dec_in.at(b, s, h) = x_emb_p.at(b, s, h);
        dec_in.at(b, s, h + cfg.byte_embed_dim) = ctx_byte.at(b, s, h);
      }
    }
  }

  Tensor3 dec_out;
  std::vector<float> C_dec;
  std::vector<float> n_dec;
  decoder.forward(dec_in, dec_out, C_dec, n_dec);

  Tensor3 logits_full;
  head.forward(dec_out, logits_full);
  logits = Tensor3(B, S, cfg.vocab_size);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < cfg.vocab_size; ++h) {
        logits.at(b, s, h) = logits_full.at(b, s, h);
      }
    }
  }
}

}  // namespace superagent
