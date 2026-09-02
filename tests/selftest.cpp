// Self test for the C API. Needs an exported head; point ASID_MODEL_DIR at it:
//
//   ASID_MODEL_DIR=out/best/onnx ./asid_selftest
//
// Checks the contract rather than the model's accuracy: that a head loads, that a real
// phoneme string yields a valid in-range label, that results are deterministic, and --
// the one that actually matters in production -- that a string with no known n-grams
// reports "undetermined" instead of confidently naming a language.

#include "african_speech_id.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}

bool exists(const std::string &p) {
  if (FILE *f = std::fopen(p.c_str(), "rb")) { std::fclose(f); return true; }
  return false;
}

}  // namespace

int main() {
  const char *env = std::getenv("ASID_MODEL_DIR");
  if (!env || !*env) {
    std::fprintf(stderr, "ASID_MODEL_DIR is not set; skipping (export a head first)\n");
    return 77;  // ctest "skipped"
  }
  const std::string dir = env;
  const std::string onnx = exists(dir + "/head.onnx") ? dir + "/head.onnx"
                                                      : dir + "/head.fp16.onnx";
  const std::string ngrams = dir + "/ngrams.txt";
  const std::string labels = dir + "/labels.txt";
  const std::string conf   = dir + "/head_config.txt";

  AsidConfig cfg;
  asid_config_init(&cfg);
  cfg.onnx_path = onnx.c_str();
  cfg.ngrams_path = ngrams.c_str();
  cfg.labels_path = labels.c_str();
  cfg.config_path = exists(conf) ? conf.c_str() : nullptr;
  cfg.num_threads = 1;

  char err[512] = {0};
  AsidHead *h = asid_create(&cfg, err, sizeof(err));
  if (!h) { std::fprintf(stderr, "FAIL  asid_create: %s\n", err); return 1; }
  check(true, "asid_create");

  const int n = asid_num_languages(h);
  check(n > 1, "num_languages > 1");
  check(asid_language(h, 0) != nullptr, "label 0 readable");
  check(asid_language(h, n) == nullptr, "out-of-range label returns NULL");

  // a real Asante Twi string from the corpus
  const char *twi = "n a e s o m e b a \xc9\xbe \xc9\xaa m \xc9\x94 j \xc9\x9b a l \xc9\xaa f a s \xc9\xbe \xc9\xaa";
  AsidResult r = asid_classify(h, twi);
  check(r.num_matched > 0, "real string matches vocabulary n-grams");
  check(r.index >= 0 && r.index < n, "prediction index in range");
  check(r.confidence > 0.0f && r.confidence <= 1.0f, "confidence in (0,1]");
  std::printf("      -> %s (%.3f, %d n-grams)\n",
              asid_language(h, r.index), r.confidence, r.num_matched);

  AsidResult r2 = asid_classify(h, twi);
  check(r2.index == r.index && r2.confidence == r.confidence, "deterministic");

  std::vector<float> probs(static_cast<size_t>(n));
  const int got = asid_classify_probs(h, twi, probs.data());
  check(got == n, "classify_probs returns full posterior");
  float sum = 0.0f;
  for (float p : probs) sum += p;
  check(sum > 0.99f && sum < 1.01f, "posterior sums to 1");

  // Nothing recognisable: must decline rather than guess.
  AsidResult empty = asid_classify(h, "");
  check(empty.index == -1 && empty.num_matched == 0, "empty string is undetermined");
  AsidResult junk = asid_classify(h, "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd");
  check(junk.index == -1 || junk.num_matched > 0,
        "unmatched junk is undetermined, not a confident guess");

  check(asid_classify(h, nullptr).index == -1, "NULL input handled");
  check(asid_classify_probs(h, twi, nullptr) == 0, "NULL output buffer handled");

  asid_destroy(h);
  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "OK",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
