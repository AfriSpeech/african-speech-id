// JNI glue for org.ghananlp.speechid.AfricanSpeechId.
//
// The handle is a jlong holding the AsidHead*. Java strings arrive as UTF-16 and JNI's
// GetStringUTFChars gives modified UTF-8, which differs from real UTF-8 for supplementary
// characters. None of the 176 IPA units are outside the BMP, so modified UTF-8 and UTF-8
// agree here -- but the Kotlin side still passes a ByteArray of real UTF-8 for the phoneme
// string, so the distinction can never bite if the inventory grows.

#include <jni.h>

#include <string>
#include <vector>

#include "african_speech_id.h"

namespace {

std::string jstr(JNIEnv *env, jstring s) {
  if (!s) return {};
  const char *c = env->GetStringUTFChars(s, nullptr);
  std::string out = c ? c : "";
  if (c) env->ReleaseStringUTFChars(s, c);
  return out;
}

AsidHead *as_head(jlong h) { return reinterpret_cast<AsidHead *>(h); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_ghananlp_speechid_AfricanSpeechId_nativeCreate(
    JNIEnv *env, jclass, jstring onnx, jstring ngrams, jstring labels, jstring config,
    jint threads, jobjectArray errOut) {
  const std::string s_onnx = jstr(env, onnx);
  const std::string s_ngrams = jstr(env, ngrams);
  const std::string s_labels = jstr(env, labels);
  const std::string s_config = jstr(env, config);

  AsidConfig cfg;
  asid_config_init(&cfg);
  cfg.onnx_path = s_onnx.c_str();
  cfg.ngrams_path = s_ngrams.c_str();
  cfg.labels_path = s_labels.c_str();
  cfg.config_path = s_config.empty() ? nullptr : s_config.c_str();
  cfg.num_threads = threads;

  char err[512] = {0};
  AsidHead *h = asid_create(&cfg, err, sizeof(err));
  if (!h && errOut && env->GetArrayLength(errOut) > 0) {
    env->SetObjectArrayElement(errOut, 0, env->NewStringUTF(err));
  }
  return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL
Java_org_ghananlp_speechid_AfricanSpeechId_nativeDestroy(JNIEnv *, jclass, jlong h) {
  asid_destroy(as_head(h));
}

JNIEXPORT jint JNICALL
Java_org_ghananlp_speechid_AfricanSpeechId_nativeNumLanguages(JNIEnv *, jclass, jlong h) {
  return asid_num_languages(as_head(h));
}

JNIEXPORT jstring JNICALL
Java_org_ghananlp_speechid_AfricanSpeechId_nativeLanguage(JNIEnv *env, jclass, jlong h,
                                                        jint index) {
  const char *s = asid_language(as_head(h), index);
  return s ? env->NewStringUTF(s) : nullptr;
}

/* Returns {index, confidence, numMatched} as a float[3]; index and numMatched are exact
 * in float well past the 41 classes and any realistic n-gram count. */
JNIEXPORT jfloatArray JNICALL
Java_org_ghananlp_speechid_AfricanSpeechId_nativeClassify(JNIEnv *env, jclass, jlong h,
                                                        jbyteArray ipaUtf8) {
  std::string ipa;
  if (ipaUtf8) {
    const jsize n = env->GetArrayLength(ipaUtf8);
    ipa.resize(static_cast<size_t>(n));
    env->GetByteArrayRegion(ipaUtf8, 0, n, reinterpret_cast<jbyte *>(ipa.data()));
  }
  const AsidResult r = asid_classify(as_head(h), ipa.c_str());
  const jfloat vals[3] = {static_cast<jfloat>(r.index), r.confidence,
                          static_cast<jfloat>(r.num_matched)};
  jfloatArray out = env->NewFloatArray(3);
  env->SetFloatArrayRegion(out, 0, 3, vals);
  return out;
}

/* Full posterior; returns null when nothing matched. */
JNIEXPORT jfloatArray JNICALL
Java_org_ghananlp_speechid_AfricanSpeechId_nativeClassifyProbs(JNIEnv *env, jclass, jlong h,
                                                             jbyteArray ipaUtf8) {
  std::string ipa;
  if (ipaUtf8) {
    const jsize n = env->GetArrayLength(ipaUtf8);
    ipa.resize(static_cast<size_t>(n));
    env->GetByteArrayRegion(ipaUtf8, 0, n, reinterpret_cast<jbyte *>(ipa.data()));
  }
  const int n = asid_num_languages(as_head(h));
  std::vector<float> probs(static_cast<size_t>(n));
  const int got = asid_classify_probs(as_head(h), ipa.c_str(), probs.data());
  if (got == 0) return nullptr;
  jfloatArray out = env->NewFloatArray(got);
  env->SetFloatArrayRegion(out, 0, got, probs.data());
  return out;
}

}  // extern "C"
