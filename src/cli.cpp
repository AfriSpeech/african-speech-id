// gsid -- identify the language of an IPA phoneme string.
//
// Reads one phoneme string per line on stdin (or takes it as an argument) and prints the
// predicted language. Useful for checking a freshly exported head outside any app, and as
// the reference for what a binding has to do.
//
//   gsid --model-dir out/best/onnx "n a e s o m e b a ɾ ɪ m ɔ"
//   ghana-ipa-asr transcribe clip.wav | gsid --model-dir out/best/onnx --top 3

#include "african_speech_id.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::fprintf(stderr,
      "usage: gsid --model-dir DIR [--top N] [--threads N] [IPA STRING]\n"
      "\n"
      "  --model-dir DIR   directory holding head.onnx, ngrams.txt, labels.txt,\n"
      "                    head_config.txt (as written by scripts/export_onnx.py)\n"
      "  --top N           print the N most likely languages (default 1)\n"
      "  --threads N       onnxruntime intra-op threads (default 1)\n"
      "\n"
      "With no IPA argument, reads one string per line from stdin.\n"
      "Units must be space separated; never split k\xcd\xa1p, k\xca\xb0 or t\xcd\xa1\xca\x83.\n");
}

bool exists(const std::string &p) {
  if (FILE *f = std::fopen(p.c_str(), "rb")) { std::fclose(f); return true; }
  return false;
}

void classify_and_print(AsidHead *h, const std::string &ipa, int top,
                        std::vector<float> *buf) {
  if (ipa.empty()) return;
  const int n = asid_classify_probs(h, ipa.c_str(), buf->data());
  if (n == 0) {
    // Every n-gram was out of vocabulary, so there is no basis for a prediction.
    // Report that plainly rather than naming whichever language scored least badly.
    std::printf("unknown\t0.000\n");
    return;
  }
  std::vector<int> order(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
  const int k = std::min(top, n);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int a, int b) { return (*buf)[a] > (*buf)[b]; });
  for (int i = 0; i < k; ++i) {
    const int idx = order[static_cast<size_t>(i)];
    if (i) std::printf("\t");
    std::printf("%s\t%.3f", asid_language(h, idx), (*buf)[static_cast<size_t>(idx)]);
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char **argv) {
  std::string dir;
  int top = 1, threads = 1;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model-dir" && i + 1 < argc)   dir = argv[++i];
    else if (a == "--top" && i + 1 < argc)    top = std::atoi(argv[++i]);
    else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (a == "-h" || a == "--help")      { usage(); return 0; }
    else if (!a.empty() && a[0] == '-')       { usage(); return 2; }
    else                                       positional.push_back(a);
  }
  if (dir.empty()) { usage(); return 2; }
  if (top < 1) top = 1;

  const std::string onnx = exists(dir + "/head.onnx") ? dir + "/head.onnx"
                                                      : dir + "/head.fp16.onnx";
  AsidConfig cfg;
  asid_config_init(&cfg);
  const std::string ngrams = dir + "/ngrams.txt";
  const std::string labels = dir + "/labels.txt";
  const std::string conf   = dir + "/head_config.txt";
  cfg.onnx_path = onnx.c_str();
  cfg.ngrams_path = ngrams.c_str();
  cfg.labels_path = labels.c_str();
  cfg.config_path = exists(conf) ? conf.c_str() : nullptr;
  cfg.num_threads = threads;

  char err[512] = {0};
  AsidHead *h = asid_create(&cfg, err, sizeof(err));
  if (!h) { std::fprintf(stderr, "gsid: %s\n", err); return 1; }

  std::vector<float> buf(static_cast<size_t>(asid_num_languages(h)));

  if (!positional.empty()) {
    std::string joined = positional[0];
    for (size_t i = 1; i < positional.size(); ++i) joined += " " + positional[i];
    classify_and_print(h, joined, top, &buf);
  } else {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      classify_and_print(h, line, top, &buf);
    }
  }

  asid_destroy(h);
  return 0;
}
