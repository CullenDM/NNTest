#include "superagent_ternary.h"

#include <array>
#include <limits>

namespace superagent {

namespace {

constexpr float kSigmoidLutMin = -8.0f;
constexpr float kSigmoidLutMax = 8.0f;
constexpr float kExpLutMin = -8.0f;
constexpr float kExpLutMax = 0.0f;
constexpr int kLutSize = 256;

struct TernaryLut {
  std::array<uint16_t, kLutSize> sigmoid_q15{};
  std::array<uint16_t, kLutSize> exp_q15{};

  TernaryLut() {
    for (int i = 0; i < kLutSize; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(kLutSize - 1);
      const float sigmoid_x = kSigmoidLutMin + t * (kSigmoidLutMax - kSigmoidLutMin);
      const float sigmoid_v = 1.0f / (1.0f + std::exp(-sigmoid_x));
      sigmoid_q15[static_cast<size_t>(i)] =
          static_cast<uint16_t>(std::min<float>(kQ15Scale, std::lround(sigmoid_v * kQ15Scale)));

      const float exp_x = kExpLutMin + t * (kExpLutMax - kExpLutMin);
      const float exp_v = std::exp(exp_x);
      exp_q15[static_cast<size_t>(i)] =
          static_cast<uint16_t>(std::min<float>(kQ15Scale, std::lround(exp_v * kQ15Scale)));
    }
  }
};

inline int lut_index(float x, float min_v, float max_v) {
  const float clamped = std::max(min_v, std::min(max_v, x));
  const float scaled = (clamped - min_v) / (max_v - min_v);
  const int idx = static_cast<int>(std::lround(scaled * static_cast<float>(kLutSize - 1)));
  return std::max(0, std::min(kLutSize - 1, idx));
}

inline uint16_t lut_sigmoid_q15(float x) {
  static const TernaryLut lut;
  return lut.sigmoid_q15[static_cast<size_t>(lut_index(x, kSigmoidLutMin, kSigmoidLutMax))];
}

inline uint16_t lut_exp_q15(float x) {
  static const TernaryLut lut;
  return lut.exp_q15[static_cast<size_t>(lut_index(x, kExpLutMin, kExpLutMax))];
}

}  // namespace

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

static inline uint8_t encode_ternary(int8_t value) {
  switch (value) {
    case 1:
      return 0x1;
    case -1:
      return 0x2;
    default:
      return 0x0;
  }
}

static inline int8_t decode_ternary(uint8_t bits) {
  switch (bits & 0x3u) {
    case 0x1:
      return 1;
    case 0x2:
      return -1;
    default:
      return 0;
  }
}

static std::vector<uint8_t> pack_ternary_weights(const std::vector<int8_t> &weights) {
  std::vector<uint8_t> packed((weights.size() + 3) / 4, 0);
  for (size_t i = 0; i < weights.size(); ++i) {
    const size_t byte_index = i / 4;
    const size_t shift = (i % 4) * 2;
    packed[byte_index] |= static_cast<uint8_t>(encode_ternary(weights[i]) << shift);
  }
  return packed;
}

static inline int8_t get_packed_weight(const std::vector<uint8_t> &packed, size_t idx) {
  const size_t byte_index = idx / 4;
  const size_t shift = (idx % 4) * 2;
  const uint8_t bits = static_cast<uint8_t>((packed[byte_index] >> shift) & 0x3u);
  return decode_ternary(bits);
}

void TernaryLinear::set_weight_ternary(const std::vector<float> &w, float threshold) {
  const size_t weight_count = static_cast<size_t>(in_f) * out_f;
  if (w.size() != weight_count) {
    return;
  }
  std::vector<int8_t> ternary(weight_count, 0);
  for (size_t i = 0; i < w.size(); ++i) {
    const float val = w[i];
    if (val > threshold) {
      ternary[i] = 1;
    } else if (val < -threshold) {
      ternary[i] = -1;
    } else {
      ternary[i] = 0;
    }
  }
  weight_packed = pack_ternary_weights(ternary);
}

void TernaryLinear::forward(const Tensor3 &x, Tensor3 &out) const {
  out.resize(x.B, x.S, out_f);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      for (int o = 0; o < out_f; ++o) {
        float acc = bias.empty() ? 0.0f : bias[static_cast<size_t>(o)];
        const size_t w_base = static_cast<size_t>(o) * in_f;
        for (int i = 0; i < in_f; ++i) {
          const int8_t w = get_packed_weight(weight_packed, w_base + static_cast<size_t>(i));
          const float x_val = ternary_quant_val(x.at(b, s, i), threshold);
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
  out.resize(x.B, x.S, x.H);
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
      const float e = static_cast<float>(lut_exp_q15(v)) / static_cast<float>(kQ15Scale);
      out[static_cast<size_t>(r) * cols + c] = e;
      sum += e;
    }
    const float inv = 1.0f / (sum + 1e-8f);
    for (int c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r) * cols + c] *= inv;
    }
  }
}

static inline int16_t clamp_q15(int32_t v) {
  const int32_t min_v = std::numeric_limits<int16_t>::min();
  const int32_t max_v = std::numeric_limits<int16_t>::max();
  return static_cast<int16_t>(std::max(min_v, std::min(max_v, v)));
}

static inline int32_t float_to_q15(float v) {
  const float scaled = v * static_cast<float>(kQ15Scale);
  const long rounded = std::lround(scaled);
  const long clamped = std::max(static_cast<long>(std::numeric_limits<int16_t>::min()),
                                std::min(static_cast<long>(std::numeric_limits<int16_t>::max()), rounded));
  return static_cast<int32_t>(clamped);
}

static inline float q15_to_float(int16_t v) {
  return static_cast<float>(v) / static_cast<float>(kQ15Scale);
}

static void mlstm_scan(const Tensor3 &q, const Tensor3 &k, const Tensor3 &v,
                       const Tensor3 &i_gate, const Tensor3 &f_gate,
                       std::vector<int16_t> &C, std::vector<int16_t> &n,
                       Tensor3 &h_out) {
  h_out.resize(q.B, q.S, q.H);
  for (int b = 0; b < q.B; ++b) {
    for (int t = 0; t < q.S; ++t) {
      const float it = i_gate.at(b, t, 0);
      const float ft = f_gate.at(b, t, 0);
      const int32_t it_q15 = float_to_q15(it);
      const int32_t ft_q15 = float_to_q15(ft);
      // Q15 fixed-point update: q15_out = (ft_q15 * q15_prev + it_q15 * q15_input) / kQ15Scale.
      // We accumulate in int32 and clamp back to int16 storage.
      for (int i = 0; i < q.H; ++i) {
        const float vt = v.at(b, t, i);
        const float kt = k.at(b, t, i);
        const int32_t vt_q15 = float_to_q15(vt);
        const int32_t kt_q15 = float_to_q15(kt);
        const size_t n_idx = (static_cast<size_t>(b) * q.H + i);
        const int32_t n_old = n[n_idx];
        const int32_t n_acc = static_cast<int32_t>(
            (static_cast<int64_t>(ft_q15) * n_old + static_cast<int64_t>(it_q15) * kt_q15) / kQ15Scale);
        n[n_idx] = clamp_q15(n_acc);
        for (int j = 0; j < q.H; ++j) {
          const size_t c_idx = (static_cast<size_t>(b) * q.H + i) * q.H + j;
          const int32_t k_j_q15 = float_to_q15(k.at(b, t, j));
          const int32_t c_old = C[c_idx];
          const int64_t term1 = static_cast<int64_t>(ft_q15) * c_old;
          const int64_t term2 = static_cast<int64_t>(it_q15) * vt_q15 * k_j_q15;
          const int32_t c_acc = static_cast<int32_t>((term1 + term2 / kQ15Scale) / kQ15Scale);
          C[c_idx] = clamp_q15(c_acc);
        }
      }
      // num = C * q
      std::vector<float> num(static_cast<size_t>(q.H), 0.0f);
      for (int i = 0; i < q.H; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < q.H; ++j) {
          const size_t c_idx = (static_cast<size_t>(b) * q.H + i) * q.H + j;
          const float c_val = static_cast<float>(C[c_idx]) / static_cast<float>(kQ15Scale);
          acc += c_val * q.at(b, t, j);
        }
        num[static_cast<size_t>(i)] = acc;
      }
      float den = 0.0f;
      for (int i = 0; i < q.H; ++i) {
        const size_t n_idx = (static_cast<size_t>(b) * q.H + i);
        const float n_val = static_cast<float>(n[n_idx]) / static_cast<float>(kQ15Scale);
        den += n_val * q.at(b, t, i);
      }
      den = std::max(std::abs(den), 1.0f);
      for (int i = 0; i < q.H; ++i) {
        h_out.at(b, t, i) = num[static_cast<size_t>(i)] / den;
      }
    }
  }
}

static inline size_t idx2(int s, int h, int H) {
  return static_cast<size_t>(s) * H + h;
}

static void ternary_linear_forward_streaming(const TernaryLinear &layer, const std::vector<float> &x, int S,
                                             std::vector<float> &out) {
  out.assign(static_cast<size_t>(S) * layer.out_f, 0.0f);
  for (int s = 0; s < S; ++s) {
    const size_t x_base = static_cast<size_t>(s) * layer.in_f;
    for (int o = 0; o < layer.out_f; ++o) {
      float acc = layer.bias.empty() ? 0.0f : layer.bias[static_cast<size_t>(o)];
      const size_t w_base = static_cast<size_t>(o) * layer.in_f;
      for (int i = 0; i < layer.in_f; ++i) {
        const int8_t w = get_packed_weight(layer.weight_packed, w_base + static_cast<size_t>(i));
        const float x_val = ternary_quant_val(x[x_base + static_cast<size_t>(i)], layer.threshold);
        if (w == 1) {
          acc += x_val;
        } else if (w == -1) {
          acc -= x_val;
        }
      }
      out[static_cast<size_t>(s) * layer.out_f + o] = acc;
    }
  }
}

static void rmsnorm_forward_streaming(const IntRMSNorm &norm, const std::vector<float> &x, int S,
                                      std::vector<float> &out) {
  out.assign(static_cast<size_t>(S) * norm.dim, 0.0f);
  for (int s = 0; s < S; ++s) {
    float mean_sq = 0.0f;
    for (int h = 0; h < norm.dim; ++h) {
      const float v = x[idx2(s, h, norm.dim)];
      mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(norm.dim);
    const float inv = 1.0f / std::sqrt(mean_sq + norm.eps);
    for (int h = 0; h < norm.dim; ++h) {
      out[idx2(s, h, norm.dim)] = x[idx2(s, h, norm.dim)] * inv * norm.weight[static_cast<size_t>(h)];
    }
  }
}

static void mlstm_scan_streaming(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
                                 const std::vector<float> &i_gate, const std::vector<float> &f_gate,
                                 int S, int H, std::vector<int16_t> &C, std::vector<int16_t> &n,
                                 std::vector<int16_t> &work, std::vector<float> &h_out) {
  h_out.assign(static_cast<size_t>(S) * H, 0.0f);
  if (work.size() < static_cast<size_t>(H)) {
    work.resize(static_cast<size_t>(H), 0);
  }
  std::vector<float> num(static_cast<size_t>(H), 0.0f);
  for (int t = 0; t < S; ++t) {
    const float it = i_gate[static_cast<size_t>(t)];
    const float ft = f_gate[static_cast<size_t>(t)];
    const int32_t it_q15 = float_to_q15(it);
    const int32_t ft_q15 = float_to_q15(ft);
    for (int j = 0; j < H; ++j) {
      work[static_cast<size_t>(j)] = static_cast<int16_t>(float_to_q15(k[idx2(t, j, H)]));
    }
    for (int i = 0; i < H; ++i) {
      const float vt = v[idx2(t, i, H)];
      const int32_t vt_q15 = float_to_q15(vt);
      const int32_t kt_q15 = work[static_cast<size_t>(i)];
      const int32_t n_old = n[static_cast<size_t>(i)];
      const int32_t n_acc = static_cast<int32_t>(
          (static_cast<int64_t>(ft_q15) * n_old + static_cast<int64_t>(it_q15) * kt_q15) / kQ15Scale);
      n[static_cast<size_t>(i)] = clamp_q15(n_acc);
      for (int j = 0; j < H; ++j) {
        const size_t c_idx = static_cast<size_t>(i) * H + j;
        const int32_t k_j_q15 = work[static_cast<size_t>(j)];
        const int32_t c_old = C[c_idx];
        const int64_t term1 = static_cast<int64_t>(ft_q15) * c_old;
        const int64_t term2 = static_cast<int64_t>(it_q15) * vt_q15 * k_j_q15;
        const int32_t c_acc = static_cast<int32_t>((term1 + term2 / kQ15Scale) / kQ15Scale);
        C[c_idx] = clamp_q15(c_acc);
      }
    }
    std::fill(num.begin(), num.end(), 0.0f);
    for (int i = 0; i < H; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < H; ++j) {
        const size_t c_idx = static_cast<size_t>(i) * H + j;
        const float c_val = static_cast<float>(C[c_idx]) / static_cast<float>(kQ15Scale);
        acc += c_val * q[idx2(t, j, H)];
      }
      num[static_cast<size_t>(i)] = acc;
    }
    float den = 0.0f;
    for (int i = 0; i < H; ++i) {
      const float n_val = static_cast<float>(n[static_cast<size_t>(i)]) / static_cast<float>(kQ15Scale);
      den += n_val * q[idx2(t, i, H)];
    }
    den = std::max(std::abs(den), 1.0f);
    for (int i = 0; i < H; ++i) {
      h_out[idx2(t, i, H)] = num[static_cast<size_t>(i)] / den;
    }
  }
}

static void mlstm_cell_forward_streaming(const TernarymLSTMCell &cell, const std::vector<float> &x, int S,
                                         std::vector<float> &h_out, std::vector<int16_t> &C, std::vector<int16_t> &n,
                                         std::vector<int16_t> &work) {
  std::vector<float> qkv;
  ternary_linear_forward_streaming(cell.W_qkv, x, S, qkv);
  const int H = cell.hidden_dim;
  std::vector<float> q(static_cast<size_t>(S) * H, 0.0f);
  std::vector<float> k(static_cast<size_t>(S) * H, 0.0f);
  std::vector<float> v(static_cast<size_t>(S) * H, 0.0f);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      const size_t base = static_cast<size_t>(s) * H;
      q[base + h] = qkv[static_cast<size_t>(s) * (3 * H) + h];
      k[base + h] = qkv[static_cast<size_t>(s) * (3 * H) + h + H];
      v[base + h] = qkv[static_cast<size_t>(s) * (3 * H) + h + 2 * H];
    }
  }
  std::vector<float> gates;
  ternary_linear_forward_streaming(cell.W_if, x, S, gates);
  std::vector<float> i_gate(static_cast<size_t>(S), 0.0f);
  std::vector<float> f_gate(static_cast<size_t>(S), 0.0f);
  for (int s = 0; s < S; ++s) {
    const float g0 = gates[static_cast<size_t>(s) * 2];
    const float g1 = gates[static_cast<size_t>(s) * 2 + 1];
    i_gate[static_cast<size_t>(s)] = static_cast<float>(lut_sigmoid_q15(g0)) / static_cast<float>(kQ15Scale);
    f_gate[static_cast<size_t>(s)] = static_cast<float>(lut_sigmoid_q15(g1)) / static_cast<float>(kQ15Scale);
  }
  C.assign(static_cast<size_t>(H) * H, 0);
  n.assign(static_cast<size_t>(H), 0);
  mlstm_scan_streaming(q, k, v, i_gate, f_gate, S, H, C, n, work, h_out);
}

static void read_and_inject_streaming(const LatentMemory &memory, const std::vector<float> &x, int S,
                                      std::vector<float> &out) {
  const int mem_size = memory.full ? memory.capacity : memory.ptr;
  if (mem_size == 0) {
    out = x;
    return;
  }
  out.assign(static_cast<size_t>(S) * memory.dim, 0.0f);
  std::vector<float> weights(static_cast<size_t>(mem_size), 0.0f);
  std::vector<float> retrieved(static_cast<size_t>(memory.dim), 0.0f);
  for (int s = 0; s < S; ++s) {
    float max_v = -1e9f;
    for (int m = 0; m < mem_size; ++m) {
      float dot = 0.0f;
      for (int h = 0; h < memory.dim; ++h) {
        const float mem_val = q15_to_float(memory.bank[static_cast<size_t>(m) * memory.dim + h]);
        dot += x[idx2(s, h, memory.dim)] * mem_val;
      }
      dot /= std::sqrt(static_cast<float>(memory.dim));
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
    std::fill(retrieved.begin(), retrieved.end(), 0.0f);
    for (int h = 0; h < memory.dim; ++h) {
      float acc = 0.0f;
      for (int m = 0; m < mem_size; ++m) {
        const float mem_val = q15_to_float(memory.bank[static_cast<size_t>(m) * memory.dim + h]);
        acc += weights[static_cast<size_t>(m)] * mem_val;
      }
      retrieved[static_cast<size_t>(h)] = acc;
    }
    for (int h = 0; h < memory.dim; ++h) {
      float gate = memory.gate_bias[static_cast<size_t>(h)];
      const size_t w_base = static_cast<size_t>(h) * memory.dim * 2;
      for (int d = 0; d < memory.dim; ++d) {
        gate += memory.gate_weight[w_base + static_cast<size_t>(d)] * x[idx2(s, d, memory.dim)];
        gate += memory.gate_weight[w_base + static_cast<size_t>(memory.dim) + d] * retrieved[static_cast<size_t>(d)];
      }
      gate = static_cast<float>(lut_sigmoid_q15(gate)) / static_cast<float>(kQ15Scale);
      out[idx2(s, h, memory.dim)] = x[idx2(s, h, memory.dim)] + gate * retrieved[static_cast<size_t>(h)];
    }
  }
}

static void write_streaming(LatentMemory &memory, const std::vector<float> &latents, int S) {
  std::vector<float> gist(static_cast<size_t>(memory.dim), 0.0f);
  for (int h = 0; h < memory.dim; ++h) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
      acc += latents[idx2(s, h, memory.dim)];
    }
    gist[static_cast<size_t>(h)] = acc / static_cast<float>(S);
  }
  for (int h = 0; h < memory.dim; ++h) {
    const int32_t q15 = float_to_q15(gist[static_cast<size_t>(h)]);
    memory.bank[static_cast<size_t>(memory.ptr) * memory.dim + h] = clamp_q15(q15);
  }
  memory.ptr = (memory.ptr + 1) % memory.capacity;
  if (memory.ptr == 0) {
    memory.full = true;
  }
}

static void attention_forward_streaming(const TernaryDeepSeekSparseAttention &attn_layer, const std::vector<float> &x, int S,
                                        std::vector<float> &out_x) {
  const int H = attn_layer.H;
  const int pad = (attn_layer.block_size - (S % attn_layer.block_size)) % attn_layer.block_size;
  const int S_p = S + pad;
  const int N = S_p / attn_layer.block_size;

  std::vector<float> x_p(static_cast<size_t>(S_p) * H, 0.0f);
  for (int s = 0; s < S_p; ++s) {
    for (int h = 0; h < H; ++h) {
      x_p[idx2(s, h, H)] = (s < S) ? x[idx2(s, h, H)] : 0.0f;
    }
  }

  std::vector<float> meta(static_cast<size_t>(N) * H, 0.0f);
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      float acc = 0.0f;
      for (int s = 0; s < attn_layer.block_size; ++s) {
        acc += x_p[idx2(n * attn_layer.block_size + s, h, H)];
      }
      meta[idx2(n, h, H)] = acc / static_cast<float>(attn_layer.block_size);
    }
  }

  std::vector<float> q_meta;
  std::vector<float> k_meta;
  ternary_linear_forward_streaming(attn_layer.indexer.W_qI, meta, N, q_meta);
  ternary_linear_forward_streaming(attn_layer.indexer.W_kI, meta, N, k_meta);

  std::vector<float> scores(static_cast<size_t>(N) * N, 0.0f);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int h = 0; h < attn_layer.indexer.heads; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < attn_layer.indexer.h_dim; ++d) {
          const int q_idx = (h * attn_layer.indexer.h_dim + d);
          dot += q_meta[idx2(i, q_idx, attn_layer.indexer.heads * attn_layer.indexer.h_dim)] *
                 k_meta[idx2(j, d, attn_layer.indexer.h_dim)];
        }
        if (dot < 0.0f) {
          dot = 0.0f;
        }
        acc += dot * attn_layer.indexer.w[static_cast<size_t>(h)];
      }
      scores[idx2(i, j, N)] = acc;
    }
  }

  std::vector<int> topk_idx(static_cast<size_t>(N) * attn_layer.topk, 0);
  for (int i = 0; i < N; ++i) {
    std::vector<std::pair<float, int>> row;
    row.reserve(static_cast<size_t>(N));
    for (int j = 0; j < N; ++j) {
      if (j > i) {
        row.emplace_back(-1e9f, j);
      } else {
        row.emplace_back(scores[idx2(i, j, N)], j);
      }
    }
    std::partial_sort(row.begin(), row.begin() + std::min(attn_layer.topk, N), row.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    for (int k = 0; k < std::min(attn_layer.topk, N); ++k) {
      topk_idx[static_cast<size_t>(i) * attn_layer.topk + k] = row[static_cast<size_t>(k)].second;
    }
  }

  std::vector<float> qkv_out;
  ternary_linear_forward_streaming(attn_layer.qkv, x_p, S_p, qkv_out);
  const int head_dim = H / attn_layer.heads;
  std::vector<float> q(static_cast<size_t>(S_p) * H, 0.0f);
  std::vector<float> k(static_cast<size_t>(S_p) * H, 0.0f);
  std::vector<float> v(static_cast<size_t>(S_p) * H, 0.0f);
  for (int s = 0; s < S_p; ++s) {
    for (int h = 0; h < H; ++h) {
      q[idx2(s, h, H)] = qkv_out[static_cast<size_t>(s) * (3 * H) + h];
      k[idx2(s, h, H)] = qkv_out[static_cast<size_t>(s) * (3 * H) + h + H];
      v[idx2(s, h, H)] = qkv_out[static_cast<size_t>(s) * (3 * H) + h + 2 * H];
    }
  }

  std::vector<float> out_full(static_cast<size_t>(S_p) * H, 0.0f);
  const int local_window = std::max(0, attn_layer.local_window_blocks);
  std::vector<int> allowed_blocks;
  std::vector<float> attn_scores(static_cast<size_t>(S_p), -1e9f);
  std::vector<float> probs;
  for (int s = 0; s < S_p; ++s) {
    const int block_i = s / attn_layer.block_size;
    allowed_blocks.clear();
    allowed_blocks.reserve(static_cast<size_t>(attn_layer.topk + 2 * local_window + 1));
    for (int k_i = 0; k_i < std::min(attn_layer.topk, N); ++k_i) {
      allowed_blocks.push_back(topk_idx[static_cast<size_t>(block_i) * attn_layer.topk + k_i]);
    }
    for (int j = std::max(0, block_i - local_window); j <= std::min(N - 1, block_i + local_window); ++j) {
      allowed_blocks.push_back(j);
    }
    std::sort(allowed_blocks.begin(), allowed_blocks.end());
    allowed_blocks.erase(std::unique(allowed_blocks.begin(), allowed_blocks.end()), allowed_blocks.end());

    for (int h = 0; h < attn_layer.heads; ++h) {
      std::fill(attn_scores.begin(), attn_scores.end(), -1e9f);
      for (int s2 = 0; s2 <= s; ++s2) {
        const int block_j = s2 / attn_layer.block_size;
        if (!std::binary_search(allowed_blocks.begin(), allowed_blocks.end(), block_j)) {
          continue;
        }
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += q[idx2(s, h * head_dim + d, H)] * k[idx2(s2, h * head_dim + d, H)];
        }
        attn_scores[static_cast<size_t>(s2)] = dot / std::sqrt(static_cast<float>(head_dim));
      }
      attn_layer.softmax.forward(attn_scores, 1, S_p, probs);
      for (int s2 = 0; s2 <= s; ++s2) {
        const float p = probs[static_cast<size_t>(s2)];
        if (p == 0.0f) {
          continue;
        }
        for (int d = 0; d < head_dim; ++d) {
          out_full[idx2(s, h * head_dim + d, H)] += p * v[idx2(s2, h * head_dim + d, H)];
        }
      }
    }
  }

  std::vector<float> projected;
  ternary_linear_forward_streaming(attn_layer.out, out_full, S_p, projected);
  out_x.assign(static_cast<size_t>(S) * H, 0.0f);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < H; ++h) {
      out_x[idx2(s, h, H)] = projected[idx2(s, h, H)];
    }
  }
}

static void feed_forward_streaming(const TernaryFeedForward &ff_layer, const std::vector<float> &x, int S,
                                   std::vector<float> &out) {
  ternary_linear_forward_streaming(ff_layer.fc, x, S, out);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < ff_layer.dim; ++h) {
      out[idx2(s, h, ff_layer.dim)] = fast_gelu(out[idx2(s, h, ff_layer.dim)]);
    }
  }
}

void TernarymLSTMCell::forward(const Tensor3 &x, Tensor3 &h_out, std::vector<int16_t> &C, std::vector<int16_t> &n) const {
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
      const float i = static_cast<float>(lut_sigmoid_q15(g0)) / static_cast<float>(kQ15Scale);
      const float f = static_cast<float>(lut_sigmoid_q15(g1)) / static_cast<float>(kQ15Scale);
      i_gate.at(b, s, 0) = i;
      f_gate.at(b, s, 0) = f;
    }
  }
  if (C.empty()) {
    C.assign(static_cast<size_t>(x.B) * hidden_dim * hidden_dim, 0);
  }
  if (n.empty()) {
    n.assign(static_cast<size_t>(x.B) * hidden_dim, 0);
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
  out_x.resize(B, S, H);
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
      const int32_t q15 = float_to_q15(gist.at(b, 0, h));
      bank[static_cast<size_t>(ptr) * dim + h] = clamp_q15(q15);
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
  out.resize(x.B, x.S, x.H);
  for (int b = 0; b < x.B; ++b) {
    for (int s = 0; s < x.S; ++s) {
      std::vector<float> weights(static_cast<size_t>(mem_size), 0.0f);
      float max_v = -1e9f;
      for (int m = 0; m < mem_size; ++m) {
        float dot = 0.0f;
        for (int h = 0; h < x.H; ++h) {
          const float mem_val = q15_to_float(bank[static_cast<size_t>(m) * dim + h]);
          dot += x.at(b, s, h) * mem_val;
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
          const float mem_val = q15_to_float(bank[static_cast<size_t>(m) * dim + h]);
          acc += weights[static_cast<size_t>(m)] * mem_val;
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
        gate = static_cast<float>(lut_sigmoid_q15(gate)) / static_cast<float>(kQ15Scale);
        out.at(b, s, h) = x.at(b, s, h) + gate * retrieved[static_cast<size_t>(h)];
      }
    }
  }
}

SuperAgent::SuperAgent(const TernaryConfig &config)
    : cfg(config),
      byte_embed(static_cast<size_t>(cfg.vocab_size) * cfg.byte_embed_dim, 0.0f),
      work(static_cast<size_t>(std::max({cfg.byte_embed_dim, cfg.enc_mlstm_dim, cfg.dec_mlstm_dim, cfg.hidden_dim})), 0),
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
  if (B == 1) {
    std::vector<float> logits_flat;
    std::vector<float> x_glob_flat;
    forward_streaming(byte_input, S, logits_flat, x_glob_flat, update_memory);
    const int r = cfg.compression_rate;
    const int S_p = S + (r - (S % r)) % r;
    const int x_glob_s = S_p / r;
    logits.resize(1, S, cfg.vocab_size);
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < cfg.vocab_size; ++h) {
        logits.at(0, s, h) = logits_flat[idx2(s, h, cfg.vocab_size)];
      }
    }
    x_glob.resize(1, x_glob_s, cfg.hidden_dim);
    for (int s = 0; s < x_glob_s; ++s) {
      for (int h = 0; h < cfg.hidden_dim; ++h) {
        x_glob.at(0, s, h) = x_glob_flat[idx2(s, h, cfg.hidden_dim)];
      }
    }
    return;
  }
  const int r = cfg.compression_rate;
  Tensor3 &x_emb = buf_x_emb;
  x_emb.resize(B, S, cfg.byte_embed_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      const uint8_t idx = byte_input[static_cast<size_t>(b) * S + s];
      const size_t base = static_cast<size_t>(idx) * cfg.byte_embed_dim;
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        x_emb.at(b, s, h) = byte_embed[base + static_cast<size_t>(h)];
      }
    }
  }

  Tensor3 &enc_feats = buf_enc_feats;
  enc_C.assign(static_cast<size_t>(B) * cfg.enc_mlstm_dim * cfg.enc_mlstm_dim, 0);
  enc_n.assign(static_cast<size_t>(B) * cfg.enc_mlstm_dim, 0);
  encoder.forward(x_emb, enc_feats, enc_C, enc_n);

  const int pad = (r - (S % r)) % r;
  const int S_p = S + pad;
  Tensor3 &enc_p = buf_enc_p;
  enc_p.resize(B, S_p, cfg.enc_mlstm_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < cfg.enc_mlstm_dim; ++h) {
        enc_p.at(b, s, h) = (s < S) ? enc_feats.at(b, s, h) : 0.0f;
      }
    }
  }

  Tensor3 &latents = buf_latents;
  latents.resize(B, S_p / r, cfg.enc_mlstm_dim * r);
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

  Tensor3 &latents_proj = buf_latents_proj;
  enc_proj.forward(latents, latents_proj);

  memory.read_and_inject(latents_proj, x_glob);
  for (int loop = 0; loop < cfg.logical_loops; ++loop) {
    for (int i = 0; i < cfg.physical_depth; ++i) {
      Tensor3 &norm1 = buf_norm1;
      ln1[static_cast<size_t>(i)].forward(x_glob, norm1);
      Tensor3 &attn_out = buf_attn_out;
      attn[static_cast<size_t>(i)].forward(norm1, attn_out);
      for (int b = 0; b < x_glob.B; ++b) {
        for (int s = 0; s < x_glob.S; ++s) {
          for (int h = 0; h < x_glob.H; ++h) {
            x_glob.at(b, s, h) += attn_out.at(b, s, h);
          }
        }
      }
      Tensor3 &norm2 = buf_norm2;
      ln2[static_cast<size_t>(i)].forward(x_glob, norm2);
      Tensor3 &ff_out = buf_ff_out;
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

  Tensor3 &ctx_proj_out = buf_ctx_proj_out;
  ctx_proj.forward(x_glob, ctx_proj_out);
  bos_proj.assign(static_cast<size_t>(cfg.byte_embed_dim), 0.0f);
  for (int h = 0; h < cfg.byte_embed_dim; ++h) {
    float acc = ctx_proj.bias.empty() ? 0.0f : ctx_proj.bias[static_cast<size_t>(h)];
    const size_t w_base = static_cast<size_t>(h) * ctx_proj.in_f;
    for (int i = 0; i < ctx_proj.in_f; ++i) {
      const int8_t w = get_packed_weight(ctx_proj.weight_packed, w_base + static_cast<size_t>(i));
      if (w == 1) {
        acc += global_bos[static_cast<size_t>(i)];
      } else if (w == -1) {
        acc -= global_bos[static_cast<size_t>(i)];
      }
    }
    bos_proj[static_cast<size_t>(h)] = acc;
  }
  Tensor3 &ctx_byte = buf_ctx_byte;
  ctx_byte.resize(B, S_p, cfg.byte_embed_dim);
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

  Tensor3 &x_emb_p = buf_x_emb_p;
  x_emb_p.resize(B, S_p, cfg.byte_embed_dim);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        x_emb_p.at(b, s, h) = (s < S) ? x_emb.at(b, s, h) : 0.0f;
      }
    }
  }

  Tensor3 &dec_in = buf_dec_in;
  dec_in.resize(B, S_p, cfg.byte_embed_dim * 2);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S_p; ++s) {
      for (int h = 0; h < cfg.byte_embed_dim; ++h) {
        dec_in.at(b, s, h) = x_emb_p.at(b, s, h);
        dec_in.at(b, s, h + cfg.byte_embed_dim) = ctx_byte.at(b, s, h);
      }
    }
  }

  Tensor3 &dec_out = buf_dec_out;
  dec_C.assign(static_cast<size_t>(B) * cfg.dec_mlstm_dim * cfg.dec_mlstm_dim, 0);
  dec_n.assign(static_cast<size_t>(B) * cfg.dec_mlstm_dim, 0);
  decoder.forward(dec_in, dec_out, dec_C, dec_n);

  Tensor3 &logits_full = buf_logits_full;
  head.forward(dec_out, logits_full);
  logits.resize(B, S, cfg.vocab_size);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      for (int h = 0; h < cfg.vocab_size; ++h) {
        logits.at(b, s, h) = logits_full.at(b, s, h);
      }
    }
  }
}

void SuperAgent::forward_streaming(const std::vector<uint8_t> &byte_input, int S, std::vector<float> &logits,
                                   std::vector<float> &x_glob, bool update_memory) {
  const int r = cfg.compression_rate;
  const int pad = (r - (S % r)) % r;
  const int S_p = S + pad;
  const int latent_S = S_p / r;

  std::vector<float> x_emb(static_cast<size_t>(S) * cfg.byte_embed_dim, 0.0f);
  for (int s = 0; s < S; ++s) {
    const uint8_t idx = byte_input[static_cast<size_t>(s)];
    const size_t base = static_cast<size_t>(idx) * cfg.byte_embed_dim;
    for (int h = 0; h < cfg.byte_embed_dim; ++h) {
      x_emb[idx2(s, h, cfg.byte_embed_dim)] = byte_embed[base + static_cast<size_t>(h)];
    }
  }

  std::vector<float> enc_feats;
  mlstm_cell_forward_streaming(encoder, x_emb, S, enc_feats, enc_C, enc_n, work);

  std::vector<float> enc_p(static_cast<size_t>(S_p) * cfg.enc_mlstm_dim, 0.0f);
  for (int s = 0; s < S_p; ++s) {
    for (int h = 0; h < cfg.enc_mlstm_dim; ++h) {
      enc_p[idx2(s, h, cfg.enc_mlstm_dim)] = (s < S) ? enc_feats[idx2(s, h, cfg.enc_mlstm_dim)] : 0.0f;
    }
  }

  std::vector<float> latents(static_cast<size_t>(latent_S) * cfg.enc_mlstm_dim * r, 0.0f);
  for (int s = 0; s < S_p; s += r) {
    const int out_s = s / r;
    for (int h = 0; h < cfg.enc_mlstm_dim; ++h) {
      for (int k = 0; k < r; ++k) {
        latents[static_cast<size_t>(out_s) * cfg.enc_mlstm_dim * r + h * r + k] =
            enc_p[idx2(s + k, h, cfg.enc_mlstm_dim)];
      }
    }
  }

  std::vector<float> latents_proj;
  ternary_linear_forward_streaming(enc_proj, latents, latent_S, latents_proj);

  read_and_inject_streaming(memory, latents_proj, latent_S, x_glob);

  std::vector<float> norm1;
  std::vector<float> attn_out;
  std::vector<float> norm2;
  std::vector<float> ff_out;
  for (int loop = 0; loop < cfg.logical_loops; ++loop) {
    for (int i = 0; i < cfg.physical_depth; ++i) {
      rmsnorm_forward_streaming(ln1[static_cast<size_t>(i)], x_glob, latent_S, norm1);
      attention_forward_streaming(attn[static_cast<size_t>(i)], norm1, latent_S, attn_out);
      for (int s = 0; s < latent_S; ++s) {
        for (int h = 0; h < cfg.hidden_dim; ++h) {
          x_glob[idx2(s, h, cfg.hidden_dim)] += attn_out[idx2(s, h, cfg.hidden_dim)];
        }
      }
      rmsnorm_forward_streaming(ln2[static_cast<size_t>(i)], x_glob, latent_S, norm2);
      feed_forward_streaming(ff[static_cast<size_t>(i)], norm2, latent_S, ff_out);
      for (int s = 0; s < latent_S; ++s) {
        for (int h = 0; h < cfg.hidden_dim; ++h) {
          x_glob[idx2(s, h, cfg.hidden_dim)] += ff_out[idx2(s, h, cfg.hidden_dim)];
        }
      }
    }
  }
  if (update_memory) {
    write_streaming(memory, x_glob, latent_S);
  }

  std::vector<float> ctx_proj_out;
  ternary_linear_forward_streaming(ctx_proj, x_glob, latent_S, ctx_proj_out);
  bos_proj.assign(static_cast<size_t>(cfg.byte_embed_dim), 0.0f);
  for (int h = 0; h < cfg.byte_embed_dim; ++h) {
    float acc = ctx_proj.bias.empty() ? 0.0f : ctx_proj.bias[static_cast<size_t>(h)];
    const size_t w_base = static_cast<size_t>(h) * ctx_proj.in_f;
    for (int i = 0; i < ctx_proj.in_f; ++i) {
      const int8_t w = get_packed_weight(ctx_proj.weight_packed, w_base + static_cast<size_t>(i));
      if (w == 1) {
        acc += global_bos[static_cast<size_t>(i)];
      } else if (w == -1) {
        acc -= global_bos[static_cast<size_t>(i)];
      }
    }
    bos_proj[static_cast<size_t>(h)] = acc;
  }
  std::vector<float> ctx_byte(static_cast<size_t>(S_p) * cfg.byte_embed_dim, 0.0f);
  for (int s = 0; s < latent_S; ++s) {
    const bool use_bos = (s == 0);
    for (int h = 0; h < cfg.byte_embed_dim; ++h) {
      const float v = use_bos ? bos_proj[static_cast<size_t>(h)] : ctx_proj_out[idx2(s - 1, h, cfg.byte_embed_dim)];
      for (int k = 0; k < r; ++k) {
        ctx_byte[idx2(s * r + k, h, cfg.byte_embed_dim)] = v;
      }
    }
  }

  std::vector<float> x_emb_p(static_cast<size_t>(S_p) * cfg.byte_embed_dim, 0.0f);
  for (int s = 0; s < S_p; ++s) {
    for (int h = 0; h < cfg.byte_embed_dim; ++h) {
      x_emb_p[idx2(s, h, cfg.byte_embed_dim)] = (s < S) ? x_emb[idx2(s, h, cfg.byte_embed_dim)] : 0.0f;
    }
  }

  std::vector<float> dec_in(static_cast<size_t>(S_p) * cfg.byte_embed_dim * 2, 0.0f);
  for (int s = 0; s < S_p; ++s) {
    for (int h = 0; h < cfg.byte_embed_dim; ++h) {
      dec_in[idx2(s, h, cfg.byte_embed_dim * 2)] = x_emb_p[idx2(s, h, cfg.byte_embed_dim)];
      dec_in[idx2(s, h + cfg.byte_embed_dim, cfg.byte_embed_dim * 2)] = ctx_byte[idx2(s, h, cfg.byte_embed_dim)];
    }
  }

  std::vector<float> dec_out;
  mlstm_cell_forward_streaming(decoder, dec_in, S_p, dec_out, dec_C, dec_n, work);

  std::vector<float> logits_full;
  ternary_linear_forward_streaming(head, dec_out, S_p, logits_full);
  logits.assign(static_cast<size_t>(S) * cfg.vocab_size, 0.0f);
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < cfg.vocab_size; ++h) {
      logits[idx2(s, h, cfg.vocab_size)] = logits_full[idx2(s, h, cfg.vocab_size)];
    }
  }
}

}  // namespace superagent
