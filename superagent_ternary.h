#ifndef SUPERAGENT_TERNARY_H
#define SUPERAGENT_TERNARY_H

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace superagent {

// Q15 fixed-point scale for mLSTM C/n states (value = q15 / kQ15Scale).
constexpr int32_t kQ15Scale = 32768;

struct TernaryConfig {
  int vocab_size = 256;
  int compression_rate = 4;
  int byte_embed_dim = 128;

  int enc_mlstm_dim = 128;
  int dec_mlstm_dim = 128;

  int hidden_dim = 256;
  int physical_depth = 6;
  int logical_loops = 2;
  int num_heads = 4;

  int memory_capacity = 512;

  int block_size = 16;
  int top_k_blocks = 4;
  int local_window_blocks = 1;
  int index_num_heads = 2;
  int index_head_dim = 16;

  float threshold_w = 0.5f;
};

struct Tensor3 {
  int B = 0;
  int S = 0;
  int H = 0;
  std::vector<float> data;

  Tensor3() = default;
  Tensor3(int b, int s, int h) : B(b), S(s), H(h), data(static_cast<size_t>(b) * s * h, 0.0f) {}

  inline float &at(int b, int s, int h) {
    return data[(static_cast<size_t>(b) * S + s) * H + h];
  }
  inline float at(int b, int s, int h) const {
    return data[(static_cast<size_t>(b) * S + s) * H + h];
  }
};

struct TernaryLinear {
  int in_f = 0;
  int out_f = 0;
  bool quantize_act = true;
  float threshold = 0.5f;
  std::vector<uint8_t> weight_packed;  // 4 ternary weights per byte
  std::vector<float> bias;

  TernaryLinear() = default;
  TernaryLinear(int in_features, int out_features, bool use_bias = true, bool quantize_act_in = true, float threshold_in = 0.5f)
      : in_f(in_features), out_f(out_features), quantize_act(quantize_act_in), threshold(threshold_in),
        weight_packed((static_cast<size_t>(out_features) * in_features + 3) / 4, 0),
        bias(use_bias ? static_cast<size_t>(out_features) : 0, 0.0f) {}

  void set_weight_ternary(const std::vector<float> &w, float threshold);
  void forward(const Tensor3 &x, Tensor3 &out) const;
};

struct IntRMSNorm {
  int dim = 0;
  float eps = 1e-5f;
  std::vector<float> weight;

  explicit IntRMSNorm(int d) : dim(d), weight(static_cast<size_t>(d), 1.0f) {}
  void forward(const Tensor3 &x, Tensor3 &out) const;
};

struct IntSoftmax {
  float clamp_min = -8.0f;
  float clamp_max = 8.0f;
  void forward(const std::vector<float> &x, int rows, int cols, std::vector<float> &out) const;
};

struct TernarymLSTMCell {
  int input_dim = 0;
  int hidden_dim = 0;
  TernaryLinear W_qkv;
  TernaryLinear W_if;

  TernarymLSTMCell(int in_dim, int hid_dim, float threshold = 0.5f)
      : input_dim(in_dim), hidden_dim(hid_dim),
        W_qkv(in_dim, hid_dim * 3, true, true, threshold),
        W_if(in_dim, 2, true, true, threshold) {}

  void forward(const Tensor3 &x, Tensor3 &h_out, std::vector<int16_t> &C, std::vector<int16_t> &n) const;
};

struct TernaryLightningIndexer {
  int heads = 0;
  int h_dim = 0;
  TernaryLinear W_qI;
  TernaryLinear W_kI;
  std::vector<float> w;

  TernaryLightningIndexer(int hidden_dim, int heads_in, int h_dim_in, float threshold = 0.5f)
      : heads(heads_in), h_dim(h_dim_in),
        W_qI(hidden_dim, heads_in * h_dim_in, false, true, threshold),
        W_kI(hidden_dim, h_dim_in, false, true, threshold),
        w(static_cast<size_t>(heads_in), 1.0f / std::max(1, heads_in)) {}

  void forward(const Tensor3 &meta, std::vector<float> &scores) const;
};

struct TernaryDeepSeekSparseAttention {
  int H = 0;
  int heads = 0;
  int block_size = 0;
  int topk = 0;
  int local_window_blocks = 0;
  TernaryLightningIndexer indexer;
  TernaryLinear qkv;
  TernaryLinear out;
  IntSoftmax softmax;

  explicit TernaryDeepSeekSparseAttention(const TernaryConfig &cfg)
      : H(cfg.hidden_dim), heads(cfg.num_heads), block_size(cfg.block_size), topk(cfg.top_k_blocks),
        local_window_blocks(cfg.local_window_blocks),
        indexer(cfg.hidden_dim, cfg.index_num_heads, cfg.index_head_dim, cfg.threshold_w),
        qkv(cfg.hidden_dim, 3 * cfg.hidden_dim, false, true, cfg.threshold_w),
        out(cfg.hidden_dim, cfg.hidden_dim, false, true, cfg.threshold_w) {}

  void forward(const Tensor3 &x, Tensor3 &out_x) const;
};

struct TernaryFeedForward {
  int dim = 0;
  TernaryLinear fc;

  explicit TernaryFeedForward(int d, float threshold = 0.5f)
      : dim(d), fc(d, d, true, true, threshold) {}

  void forward(const Tensor3 &x, Tensor3 &out) const;
};

struct LatentMemory {
  int capacity = 0;
  int dim = 0;
  std::vector<int16_t> bank;
  int ptr = 0;
  bool full = false;
  std::vector<float> gate_weight;
  std::vector<float> gate_bias;

  LatentMemory(int d, int cap)
      : capacity(cap), dim(d),
        bank(static_cast<size_t>(cap) * d, 0.0f),
        gate_weight(static_cast<size_t>(d) * d * 2, 0.0f),
        gate_bias(static_cast<size_t>(d), 0.0f) {}

  void write(const Tensor3 &latents);
  void read_and_inject(const Tensor3 &x, Tensor3 &out) const;
};

struct SuperAgent {
  TernaryConfig cfg;
  std::vector<float> byte_embed;
  TernarymLSTMCell encoder;
  TernaryLinear enc_proj;
  LatentMemory memory;
  std::vector<IntRMSNorm> ln1;
  std::vector<TernaryDeepSeekSparseAttention> attn;
  std::vector<IntRMSNorm> ln2;
  std::vector<TernaryFeedForward> ff;
  TernaryLinear ctx_proj;
  TernarymLSTMCell decoder;
  TernaryLinear head;
  std::vector<float> global_bos;

  explicit SuperAgent(const TernaryConfig &config);
  void forward(const std::vector<uint8_t> &byte_input, int B, int S, Tensor3 &logits, Tensor3 &x_glob, bool update_memory);
};

}  // namespace superagent

#endif  // SUPERAGENT_TERNARY_H
