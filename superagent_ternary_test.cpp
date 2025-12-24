#include "superagent_ternary.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

using superagent::SuperAgent;
using superagent::TernaryConfig;
using superagent::TernaryLinear;
using superagent::Tensor3;

static void fill_ternary(TernaryLinear &layer) {
  for (size_t i = 0; i < layer.weight_count; ++i) {
    const int mod = static_cast<int>(i % 3);
    if (mod == 0) {
      layer.set_weight_value(i, 1);
    } else if (mod == 1) {
      layer.set_weight_value(i, 0);
    } else {
      layer.set_weight_value(i, -1);
    }
  }
  for (size_t i = 0; i < layer.bias.size(); ++i) {
    layer.bias[i] = 0.1f;
  }
}

static void fill_agent_weights(SuperAgent &agent) {
  for (size_t i = 0; i < agent.byte_embed.size(); ++i) {
    agent.byte_embed[i] = static_cast<float>((i % 5) - 2) * 0.1f;
  }
  for (size_t i = 0; i < agent.global_bos.size(); ++i) {
    agent.global_bos[i] = 0.05f;
  }
  fill_ternary(agent.encoder.W_qkv);
  fill_ternary(agent.encoder.W_if);
  fill_ternary(agent.enc_proj);
  fill_ternary(agent.ctx_proj);
  fill_ternary(agent.decoder.W_qkv);
  fill_ternary(agent.decoder.W_if);
  fill_ternary(agent.head);

  for (size_t i = 0; i < agent.memory.gate_weight.size(); ++i) {
    agent.memory.gate_weight[i] = 0.01f;
  }
  for (size_t i = 0; i < agent.memory.gate_bias.size(); ++i) {
    agent.memory.gate_bias[i] = 0.0f;
  }

  for (size_t i = 0; i < agent.ln1.size(); ++i) {
    std::fill(agent.ln1[i].weight.begin(), agent.ln1[i].weight.end(), 1.0f);
    std::fill(agent.ln2[i].weight.begin(), agent.ln2[i].weight.end(), 1.0f);
    fill_ternary(agent.attn[i].qkv);
    fill_ternary(agent.attn[i].out);
    fill_ternary(agent.attn[i].indexer.W_qI);
    fill_ternary(agent.attn[i].indexer.W_kI);
    fill_ternary(agent.ff[i].fc);
  }
}

static void print_model_summary(const SuperAgent &agent, std::ostream &os) {
  const TernaryConfig &cfg = agent.cfg;
  os << "Model summary (from config/weights):\n";
  os << "  vocab_size=" << cfg.vocab_size << "\n";
  os << "  byte_embed_dim=" << cfg.byte_embed_dim << "\n";
  os << "  enc_mlstm_dim=" << cfg.enc_mlstm_dim << "\n";
  os << "  dec_mlstm_dim=" << cfg.dec_mlstm_dim << "\n";
  os << "  hidden_dim=" << cfg.hidden_dim << "\n";
  os << "  physical_depth=" << cfg.physical_depth << "\n";
  os << "  logical_loops=" << cfg.logical_loops << "\n";
  os << "  num_heads=" << cfg.num_heads << "\n";
  os << "  block_size=" << cfg.block_size << "\n";
  os << "  top_k_blocks=" << cfg.top_k_blocks << "\n";
  os << "  local_window_blocks=" << cfg.local_window_blocks << "\n";
  os << "  index_num_heads=" << cfg.index_num_heads << "\n";
  os << "  index_head_dim=" << cfg.index_head_dim << "\n";
  os << "  memory_capacity=" << cfg.memory_capacity << "\n";
  os << "  byte_embed params=" << agent.byte_embed.size() << "\n";
  os << "  encoder qkv params=" << agent.encoder.W_qkv.weight_count << "\n";
  os << "  encoder if params=" << agent.encoder.W_if.weight_count << "\n";
  os << "  enc_proj params=" << agent.enc_proj.weight_count << "\n";
  os << "  ctx_proj params=" << agent.ctx_proj.weight_count << "\n";
  os << "  decoder qkv params=" << agent.decoder.W_qkv.weight_count << "\n";
  os << "  decoder if params=" << agent.decoder.W_if.weight_count << "\n";
  os << "  head params=" << agent.head.weight_count << "\n";
  os << "  memory gate params=" << agent.memory.gate_weight.size() << "\n";
  os << "  attention blocks=" << agent.attn.size() << "\n";
  os << "  ff blocks=" << agent.ff.size() << "\n";
}

struct ParamStats {
  size_t ternary_weights = 0;
  size_t float_params = 0;
};

static void add_linear_params(const TernaryLinear &layer, ParamStats &stats) {
  stats.ternary_weights += layer.weight_count;
  stats.float_params += layer.bias.size();
}

static ParamStats count_parameters(const SuperAgent &agent) {
  ParamStats stats;
  stats.float_params += agent.byte_embed.size();
  stats.float_params += agent.global_bos.size();
  stats.float_params += agent.memory.gate_weight.size();
  stats.float_params += agent.memory.gate_bias.size();

  add_linear_params(agent.encoder.W_qkv, stats);
  add_linear_params(agent.encoder.W_if, stats);
  add_linear_params(agent.enc_proj, stats);
  add_linear_params(agent.ctx_proj, stats);
  add_linear_params(agent.decoder.W_qkv, stats);
  add_linear_params(agent.decoder.W_if, stats);
  add_linear_params(agent.head, stats);

  for (size_t i = 0; i < agent.attn.size(); ++i) {
    add_linear_params(agent.attn[i].qkv, stats);
    add_linear_params(agent.attn[i].out, stats);
    add_linear_params(agent.attn[i].indexer.W_qI, stats);
    add_linear_params(agent.attn[i].indexer.W_kI, stats);
    add_linear_params(agent.ff[i].fc, stats);
    stats.float_params += agent.ln1[i].weight.size();
    stats.float_params += agent.ln2[i].weight.size();
  }

  return stats;
}

static size_t tensor_elems(int b, int s, int h) {
  return static_cast<size_t>(b) * static_cast<size_t>(s) * static_cast<size_t>(h);
}

static size_t estimate_forward_activation_elems(const TernaryConfig &cfg, int B, int S) {
  const int r = cfg.compression_rate;
  const int pad = (r - (S % r)) % r;
  const int S_p = S + pad;
  const int latents_s = S_p / r;

  size_t elems = 0;
  elems += tensor_elems(B, S, cfg.byte_embed_dim);              // x_emb
  elems += tensor_elems(B, S, cfg.enc_mlstm_dim);               // enc_feats
  elems += tensor_elems(B, S_p, cfg.enc_mlstm_dim);             // enc_p
  elems += tensor_elems(B, latents_s, cfg.enc_mlstm_dim * r);   // latents
  elems += tensor_elems(B, latents_s, cfg.hidden_dim);          // latents_proj / x_glob
  elems += tensor_elems(B, latents_s, cfg.hidden_dim) * 3;      // norm1/attn_out/norm2 approx
  elems += tensor_elems(B, latents_s, cfg.hidden_dim);          // ff_out
  elems += tensor_elems(B, latents_s, cfg.byte_embed_dim);      // ctx_proj_out
  elems += tensor_elems(B, S_p, cfg.byte_embed_dim);            // ctx_byte
  elems += tensor_elems(B, S_p, cfg.byte_embed_dim);            // x_emb_p
  elems += tensor_elems(B, S_p, cfg.byte_embed_dim * 2);        // dec_in
  elems += tensor_elems(B, S_p, cfg.dec_mlstm_dim);             // dec_out
  elems += tensor_elems(B, S_p, cfg.vocab_size);                // logits_full
  elems += tensor_elems(B, S, cfg.vocab_size);                  // logits
  return elems;
}

static void print_resource_profile(const SuperAgent &agent, int B, int S, std::ostream &os) {
  const ParamStats stats = count_parameters(agent);
  size_t ternary_bytes = 0;
  ternary_bytes += agent.encoder.W_qkv.packed_size();
  ternary_bytes += agent.encoder.W_if.packed_size();
  ternary_bytes += agent.enc_proj.packed_size();
  ternary_bytes += agent.ctx_proj.packed_size();
  ternary_bytes += agent.decoder.W_qkv.packed_size();
  ternary_bytes += agent.decoder.W_if.packed_size();
  ternary_bytes += agent.head.packed_size();
  for (size_t i = 0; i < agent.attn.size(); ++i) {
    ternary_bytes += agent.attn[i].qkv.packed_size();
    ternary_bytes += agent.attn[i].out.packed_size();
    ternary_bytes += agent.attn[i].indexer.W_qI.packed_size();
    ternary_bytes += agent.attn[i].indexer.W_kI.packed_size();
    ternary_bytes += agent.ff[i].fc.packed_size();
  }
  const size_t float_bytes = stats.float_params * sizeof(float);
  const size_t total_param_bytes = ternary_bytes + float_bytes;
  const size_t activation_elems = estimate_forward_activation_elems(agent.cfg, B, S);
  const size_t activation_bytes_float = activation_elems * sizeof(float);
  const size_t activation_bytes_int8 = activation_elems * sizeof(int8_t);

  os << "Resource profile (estimated):\n";
  os << "  ternary weight params=" << stats.ternary_weights << " ("
     << ternary_bytes << " bytes)\n";
  os << "  float params=" << stats.float_params << " ("
     << float_bytes << " bytes)\n";
  os << "  total params bytes=" << total_param_bytes << "\n";
  os << "  activation elements=" << activation_elems << "\n";
  os << "  activation bytes (float)=" << activation_bytes_float << "\n";
  os << "  activation bytes (int8, theoretical)=" << activation_bytes_int8 << "\n";
}

int main() {
  TernaryConfig cfg;
  cfg.vocab_size = 8;
  cfg.compression_rate = 2;
  cfg.byte_embed_dim = 4;
  cfg.enc_mlstm_dim = 4;
  cfg.dec_mlstm_dim = 4;
  cfg.hidden_dim = 8;
  cfg.physical_depth = 1;
  cfg.logical_loops = 1;
  cfg.num_heads = 2;
  cfg.block_size = 2;
  cfg.top_k_blocks = 1;
  cfg.local_window_blocks = 1;
  cfg.index_num_heads = 1;
  cfg.index_head_dim = 2;
  cfg.memory_capacity = 4;

  SuperAgent agent(cfg);
  fill_agent_weights(agent);
  const int B = 1;
  const int S = 4;
  std::ofstream file_out("superagent_ternary_test_output.txt");
  std::ostringstream buffer;
  print_model_summary(agent, buffer);
  print_resource_profile(agent, B, S, buffer);
  std::vector<uint8_t> input(static_cast<size_t>(B) * S, 0);
  input[0] = 1;
  input[1] = 2;
  input[2] = 3;
  input[3] = 4;

  Tensor3 logits;
  Tensor3 x_glob;
  agent.forward(input, B, S, logits, x_glob, true);

  buffer << "Logits shape: B=" << logits.B << " S=" << logits.S << " H=" << logits.H << "\n";
  buffer << "First token logits: ";
  for (int i = 0; i < logits.H; ++i) {
    buffer << logits.at(0, 0, i) << (i + 1 == logits.H ? "\n" : ", ");
  }
  buffer << "Global latents shape: B=" << x_glob.B << " S=" << x_glob.S << " H=" << x_glob.H << "\n";

  const std::string output = buffer.str();
  std::cout << output;
  if (file_out.is_open()) {
    file_out << output;
  }

  return 0;
}
