// Implementation of the african-speech-id C API.
//
// The heavy lifting is in the ONNX graph, which carries the idf vector and the class
// weights and does the tf-idf arithmetic itself. This file only has to turn a phoneme
// string into (indices, counts) and call the session:
//
//   split on ASCII space -> emit 1..N-grams -> look each up -> count -> run
//
// The vocabulary is the one part worth being careful about. At 400k n-grams an
// unordered_map<string,int> costs tens of megabytes of small allocations, which is a lot
// to ask of a phone. Instead every n-gram lives back to back in one blob and lookup is a
// binary search over an offset table sorted by content: two allocations total, and the
// blob is exactly the bytes of ngrams.txt.

#include "african_speech_id.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>

#define ASID_VERSION_STR "0.1.0"

namespace {


// ---------------------------------------------------------------- text

// Byte offset of every codepoint start, plus the end. char_wb counts CHARACTERS, and 'kɔ'
// is two characters in three bytes: a byte-wise loop would cut ɔ (U+0254) in half and emit
// n-grams that exist nowhere in the vocabulary. That does not raise -- it just quietly
// misclassifies -- so the whole tokeniser is written in terms of these offsets.
void codepoint_offsets(const std::string &s, std::vector<size_t> *off) {
  off->clear();
  for (size_t i = 0; i < s.size();) {
    off->push_back(i);
    const unsigned char c = static_cast<unsigned char>(s[i]);
    i += (c < 0x80) ? 1 : (c >> 5) == 0x06 ? 2 : (c >> 4) == 0x0E ? 3 : (c >> 3) == 0x1E ? 4 : 1;
  }
  off->push_back(s.size());
}

// Case folding here is Unicode, not ASCII: Ɛ (U+0190) folds to ɛ (U+025B). std::tolower
// works a byte at a time and cannot express that. The table is generated at export time
// from the characters that actually occur in the vocabulary, so it is exact for this model
// and needs no ICU. Anything absent never appeared in training and passes through.
class CaseFold {
 public:
  void load(const std::string &path) {
    map_.clear();
    std::ifstream in(path);
    if (!in) return;                       // absent means the head was trained cased
    std::string line;
    while (std::getline(in, line)) {
      const size_t tab = line.find('\t');
      if (tab == std::string::npos) continue;
      std::string up = line.substr(0, tab), lo = line.substr(tab + 1);
      while (!lo.empty() && (lo.back() == '\r' || lo.back() == '\n')) lo.pop_back();
      if (!up.empty() && !lo.empty()) map_[up] = lo;
    }
  }
  size_t size() const { return map_.size(); }
  void apply(const std::string &in, std::string *out) const {
    out->clear();
    out->reserve(in.size());
    std::vector<size_t> off;
    codepoint_offsets(in, &off);
    for (size_t k = 0; k + 1 < off.size(); ++k) {
      const std::string ch = in.substr(off[k], off[k + 1] - off[k]);
      if (ch.size() == 1) {
        const unsigned char c = static_cast<unsigned char>(ch[0]);
        out->push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : ch[0]);
        continue;
      }
      const auto it = map_.find(ch);
      out->append(it == map_.end() ? ch : it->second);
    }
  }

 private:
  std::map<std::string, std::string> map_;
};

// Overlapping windows measured in CHARACTERS, matching how the head was trained. A short
// transcript is one window rather than being dropped.
void window(const std::string &s, int size, int stride, std::vector<std::string> *out) {
  out->clear();
  std::vector<size_t> off;
  codepoint_offsets(s, &off);
  const int n = static_cast<int>(off.size()) - 1;
  if (size <= 0 || n <= size) { out->push_back(s); return; }
  const int step = stride > 0 ? stride : size;
  for (int i = 0; i + size <= n; i += step) {
    out->push_back(s.substr(off[i], off[i + size] - off[i]));
  }
  const std::string tail = s.substr(off[n - size], s.size() - off[n - size]);
  if (out->empty() || out->back() != tail) out->push_back(tail);
}

// ---------------------------------------------------------------- vocabulary

// n-grams stored contiguously; `sorted` indexes them in lexicographic order so lookup is
// a binary search. Index in the model is the *line number*, which is what `line_of` holds.
class Vocab {
 public:
  bool load(const std::string &path, std::string *err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { *err = "cannot open " + path; return false; }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) { *err = "cannot size " + path; return false; }
    in.seekg(0, std::ios::beg);

    blob_.resize(static_cast<size_t>(size));
    if (size > 0 && !in.read(blob_.data(), size)) { *err = "cannot read " + path; return false; }

    // one entry per line; keep offsets rather than copying substrings out
    offsets_.clear();
    size_t start = 0;
    for (size_t i = 0; i <= blob_.size(); ++i) {
      if (i == blob_.size() || blob_[i] == '\n') {
        size_t end = i;
        if (end > start && blob_[end - 1] == '\r') --end;   // tolerate CRLF
        if (i == blob_.size() && end == start) break;       // ignore a trailing newline
        offsets_.push_back({start, end - start});
        start = i + 1;
      }
    }

    sorted_.resize(offsets_.size());
    for (size_t i = 0; i < sorted_.size(); ++i) sorted_[i] = static_cast<int32_t>(i);
    std::sort(sorted_.begin(), sorted_.end(),
              [this](int32_t a, int32_t b) { return at(a) < at(b); });
    return true;
  }

  size_t size() const { return offsets_.size(); }

  std::string_view at(int32_t i) const {
    const auto &e = offsets_[static_cast<size_t>(i)];
    return std::string_view(blob_.data() + e.off, e.len);
  }

  // -1 when absent
  int32_t find(std::string_view s) const {
    size_t lo = 0, hi = sorted_.size();
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      const std::string_view v = at(sorted_[mid]);
      if (v < s) lo = mid + 1;
      else       hi = mid;
    }
    if (lo < sorted_.size() && at(sorted_[lo]) == s) return sorted_[lo];
    return -1;
  }

 private:
  struct Entry { size_t off; size_t len; };
  std::vector<char> blob_;
  std::vector<Entry> offsets_;
  std::vector<int32_t> sorted_;
};

bool read_lines(const std::string &path, std::vector<std::string> *out, std::string *err) {
  std::ifstream in(path);
  if (!in) { *err = "cannot open " + path; return false; }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    out->push_back(line);
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------- the head

struct AsidHead {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "gsid"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Vocab vocab;
  CaseFold fold;
  std::vector<std::string> labels;
  int ngram_min = 1;
  int ngram_max = 5;
  bool char_analyzer = false;   // char_wb over codepoints, vs whitespace phoneme units
  bool lowercase = true;
  int chunk_chars = 0;          // 0 classifies the whole transcript in one pass
  int chunk_stride = 20;

  std::string in0, in1, out0, out1;   // node names, owned
};

namespace {

// Splits on ASCII space only, so multi-byte units (k͡p, kʰ, t͡ʃ) survive intact.
void split_units(const char *ipa, std::vector<std::string_view> *out) {
  const char *p = ipa;
  while (*p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') ++p;
    out->emplace_back(start, static_cast<size_t>(p - start));
  }
}

// char_wb: each whitespace-delimited word is padded with one space either side and n-grams
// are taken inside that padded word only, never across a word boundary. Iteration is over
// codepoints, so a two-byte ɔ counts as one character exactly as scikit-learn counts it.
void char_wb_hits(const AsidHead *h, const std::string &text, std::vector<int32_t> *hits) {
  std::string folded;
  if (h->lowercase) h->fold.apply(text, &folded);
  const std::string &src = h->lowercase ? folded : text;

  std::string word, padded;
  size_t i = 0;
  std::vector<size_t> off;
  while (i <= src.size()) {
    if (i == src.size() || src[i] == ' ' || src[i] == '\t' || src[i] == '\n' ||
        src[i] == '\r') {
      if (!word.empty()) {
        padded.assign(" ").append(word).append(" ");
        codepoint_offsets(padded, &off);
        const int n = static_cast<int>(off.size()) - 1;
        for (int len = h->ngram_min; len <= h->ngram_max; ++len) {
          if (len > n) break;
          for (int k = 0; k + len <= n; ++k) {
            const int32_t id = h->vocab.find(
                std::string_view(padded.data() + off[k], off[k + len] - off[k]));
            if (id >= 0) hits->push_back(id);
          }
        }
        word.clear();
      }
      if (i == src.size()) break;
    } else {
      word.push_back(src[i]);
    }
    ++i;
  }
}

// Builds (indices, counts) for every in-vocabulary n-gram of order ngram_min..ngram_max.
// Counts are accumulated by sorting the hit list -- cheaper than a hash map for the few
// hundred n-grams a single utterance produces.
int featurise(const AsidHead *h, const char *ipa,
              std::vector<int64_t> *indices, std::vector<float> *counts) {
  std::vector<int32_t> hits;
  if (h->char_analyzer) {
    char_wb_hits(h, std::string(ipa), &hits);
    if (hits.empty()) return 0;
    std::sort(hits.begin(), hits.end());
    indices->clear(); counts->clear();
    for (size_t i = 0; i < hits.size();) {
      size_t j = i;
      while (j < hits.size() && hits[j] == hits[i]) ++j;
      indices->push_back(static_cast<int64_t>(hits[i]));
      counts->push_back(static_cast<float>(j - i));
      i = j;
    }
    return static_cast<int>(hits.size());
  }

  std::vector<std::string_view> units;
  split_units(ipa, &units);
  if (units.empty()) return 0;

  std::string gram;
  for (size_t i = 0; i < units.size(); ++i) {
    gram.clear();
    for (int n = 1; n <= h->ngram_max; ++n) {
      const size_t j = i + static_cast<size_t>(n) - 1;
      if (j >= units.size()) break;
      if (n > 1) gram += ' ';
      gram.append(units[j].data(), units[j].size());
      if (n < h->ngram_min) continue;
      const int32_t id = h->vocab.find(gram);
      if (id >= 0) hits.push_back(id);
    }
  }
  if (hits.empty()) return 0;

  std::sort(hits.begin(), hits.end());
  indices->clear(); counts->clear();
  for (size_t i = 0; i < hits.size();) {
    size_t j = i;
    while (j < hits.size() && hits[j] == hits[i]) ++j;
    indices->push_back(static_cast<int64_t>(hits[i]));
    counts->push_back(static_cast<float>(j - i));
    i = j;
  }
  return static_cast<int>(hits.size());
}

// Classify every window and sum logits across them.
//
// Summing logits rather than counting winners means a window the model is sure about
// outweighs several it is not, which is the whole reason to window in the first place.
// Probabilities would be wrong to sum: softmax is applied per window, so averaging them
// discards exactly the confidence information the vote is meant to use.
bool run_voted(AsidHead *h, const char *text, std::vector<float> *scores, int *n_matched) {
  std::vector<std::string> wins;
  window(std::string(text), h->chunk_chars, h->chunk_stride, &wins);

  const size_t C = h->labels.size();
  scores->assign(C, 0.0f);
  *n_matched = 0;
  bool any = false;

  std::vector<int64_t> indices;
  std::vector<float> counts;
  for (const std::string &w : wins) {
    const int m = featurise(h, w.c_str(), &indices, &counts);
    if (m == 0) continue;
    *n_matched += m;
    const int64_t shape[1] = {static_cast<int64_t>(indices.size())};
    Ort::Value t_idx = Ort::Value::CreateTensor<int64_t>(
        h->mem, indices.data(), indices.size(), shape, 1);
    Ort::Value t_cnt = Ort::Value::CreateTensor<float>(
        h->mem, counts.data(), counts.size(), shape, 1);
    const char *in_names[2] = {h->in0.c_str(), h->in1.c_str()};
    const char *out_names[1] = {h->out0.c_str()};      // logits
    Ort::Value ins[2] = {std::move(t_idx), std::move(t_cnt)};
    auto outs = h->session->Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 1);
    const float *logits = outs[0].GetTensorData<float>();
    for (size_t c = 0; c < C; ++c) (*scores)[c] += logits[c];
    any = true;
  }
  return any;
}

// Returns probs pointer into `owned` outputs, or nullptr when nothing matched.
const float *run(AsidHead *h, const char *ipa, int *n_matched,
                 std::vector<Ort::Value> *outputs) {
  std::vector<int64_t> indices;
  std::vector<float> counts;
  *n_matched = featurise(h, ipa, &indices, &counts);
  if (*n_matched == 0) return nullptr;

  const int64_t k = static_cast<int64_t>(indices.size());
  const int64_t shape[1] = {k};

  Ort::Value t_idx = Ort::Value::CreateTensor<int64_t>(
      h->mem, indices.data(), indices.size(), shape, 1);
  Ort::Value t_cnt = Ort::Value::CreateTensor<float>(
      h->mem, counts.data(), counts.size(), shape, 1);

  const char *in_names[2] = {h->in0.c_str(), h->in1.c_str()};
  const char *out_names[2] = {h->out0.c_str(), h->out1.c_str()};
  Ort::Value ins[2] = {std::move(t_idx), std::move(t_cnt)};

  *outputs = h->session->Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 2);
  return (*outputs)[1].GetTensorData<float>();   // probs
}

}  // namespace

// ---------------------------------------------------------------- C API

extern "C" {

void asid_config_init(AsidConfig *cfg) {
  if (!cfg) return;
  cfg->onnx_path = nullptr;
  cfg->ngrams_path = nullptr;
  cfg->labels_path = nullptr;
  cfg->config_path = nullptr;
  cfg->num_threads = 1;
}

AsidHead *asid_create(const AsidConfig *cfg, char *err, size_t err_len) {
  auto fail = [&](const std::string &m) -> AsidHead * {
    if (err && err_len) { std::snprintf(err, err_len, "%s", m.c_str()); }
    return nullptr;
  };
  if (!cfg || !cfg->onnx_path || !cfg->ngrams_path || !cfg->labels_path)
    return fail("onnx_path, ngrams_path and labels_path are all required");

  auto h = std::make_unique<AsidHead>();
  std::string e;

  if (!h->vocab.load(cfg->ngrams_path, &e)) return fail(e);
  {
    // casefold.txt sits beside ngrams.txt; absent means the head was trained cased
    std::string cf(cfg->ngrams_path);
    const size_t slash = cf.find_last_of("/\\");
    cf = (slash == std::string::npos ? std::string() : cf.substr(0, slash + 1)) + "casefold.txt";
    h->fold.load(cf);
  }
  if (!read_lines(cfg->labels_path, &h->labels, &e)) return fail(e);
  while (!h->labels.empty() && h->labels.back().empty()) h->labels.pop_back();
  if (h->labels.empty()) return fail("labels.txt is empty");

  if (cfg->config_path && *cfg->config_path) {
    std::vector<std::string> lines;
    if (read_lines(cfg->config_path, &lines, &e)) {
      for (const auto &l : lines) {
        const size_t sp = l.find(' ');
        if (sp == std::string::npos) continue;
        const std::string key = l.substr(0, sp), val = l.substr(sp + 1);
        if (key == "ngram_min") h->ngram_min = std::atoi(val.c_str());
        else if (key == "ngram_max") h->ngram_max = std::atoi(val.c_str());
        else if (key == "analyzer") h->char_analyzer = (val.substr(0, 4) == "char");
        else if (key == "lowercase") h->lowercase = (std::atoi(val.c_str()) != 0);
        else if (key == "chunk_chars") h->chunk_chars = std::atoi(val.c_str());
        else if (key == "chunk_stride") h->chunk_stride = std::atoi(val.c_str());
      }
    }
  }
  if (h->ngram_min < 1 || h->ngram_max < h->ngram_min)
    return fail("invalid ngram range in head_config.txt");

  try {
    const int nt = cfg->num_threads > 0 ? cfg->num_threads : 1;
    h->opts.SetIntraOpNumThreads(nt);
    h->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    h->session = std::make_unique<Ort::Session>(h->env, cfg->onnx_path, h->opts);

    Ort::AllocatorWithDefaultOptions alloc;
    if (h->session->GetInputCount() != 2 || h->session->GetOutputCount() != 2)
      return fail("expected a 2-input / 2-output head graph");
    h->in0  = h->session->GetInputNameAllocated(0, alloc).get();
    h->in1  = h->session->GetInputNameAllocated(1, alloc).get();
    h->out0 = h->session->GetOutputNameAllocated(0, alloc).get();
    h->out1 = h->session->GetOutputNameAllocated(1, alloc).get();
  } catch (const Ort::Exception &ex) {
    return fail(std::string("onnxruntime: ") + ex.what());
  }

  return h.release();
}

void asid_destroy(AsidHead *head) { delete head; }

int asid_num_languages(const AsidHead *head) {
  return head ? static_cast<int>(head->labels.size()) : 0;
}

const char *asid_language(const AsidHead *head, int index) {
  if (!head || index < 0 || index >= static_cast<int>(head->labels.size())) return nullptr;
  return head->labels[static_cast<size_t>(index)].c_str();
}

AsidResult asid_classify(AsidHead *head, const char *ipa) {
  AsidResult r{-1, 0.0f, 0};
  if (!head || !ipa) return r;
  try {
    const int n = static_cast<int>(head->labels.size());
    if (head->chunk_chars > 0) {
      std::vector<float> scores;
      int matched = 0;
      if (!run_voted(head, ipa, &scores, &matched)) return r;
      r.num_matched = matched;
      int best = 0;
      for (int i = 1; i < n; ++i) if (scores[i] > scores[best]) best = i;
      r.index = best;
      // softmax over the summed logits, so confidence stays comparable across clip lengths
      double z = 0.0;
      for (int i = 0; i < n; ++i) z += std::exp(scores[i] - scores[best]);
      r.confidence = static_cast<float>(1.0 / z);
      return r;
    }
    std::vector<Ort::Value> outs;
    int matched = 0;
    const float *probs = run(head, ipa, &matched, &outs);
    r.num_matched = matched;
    if (!probs) return r;
    int best = 0;
    for (int i = 1; i < n; ++i) if (probs[i] > probs[best]) best = i;
    r.index = best;
    r.confidence = probs[best];
  } catch (const Ort::Exception &) {
    r.index = -1; r.confidence = 0.0f;
  }
  return r;
}

int asid_classify_probs(AsidHead *head, const char *ipa, float *probs_out) {
  if (!head || !ipa || !probs_out) return 0;
  try {
    const int n = static_cast<int>(head->labels.size());
    if (head->chunk_chars > 0) {
      std::vector<float> scores;
      int matched = 0;
      if (!run_voted(head, ipa, &scores, &matched)) return 0;
      // softmax the summed logits so callers get a distribution, not raw sums
      float mx = scores[0];
      for (int i = 1; i < n; ++i) mx = std::max(mx, scores[i]);
      double z = 0.0;
      for (int i = 0; i < n; ++i) z += std::exp(scores[i] - mx);
      for (int i = 0; i < n; ++i)
        probs_out[i] = static_cast<float>(std::exp(scores[i] - mx) / z);
      return n;
    }
    std::vector<Ort::Value> outs;
    int matched = 0;
    const float *probs = run(head, ipa, &matched, &outs);
    if (!probs) return 0;
    std::memcpy(probs_out, probs, sizeof(float) * static_cast<size_t>(n));
    return n;
  } catch (const Ort::Exception &) {
    return 0;
  }
}

const char *asid_version(void) { return ASID_VERSION_STR; }

}  // extern "C"
