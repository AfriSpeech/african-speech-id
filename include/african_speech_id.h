/* african-speech-id -- language identification over IPA phoneme strings.
 *
 * Identifies which of 41 Ghanaian and West African languages a phoneme string is in.
 * The input is the IPA that ghana-ipa-asr emits, so the full pipeline on device is
 *
 *     audio --[sherpa-onnx + ghana-speech-phoneme-asr]--> IPA units --[this]--> language
 *
 * No Python, no JSON parser, no STL in the ABI. Links against onnxruntime only; the
 * speech-to-IPA stage is a separate sherpa-onnx call that this library does not wrap,
 * so you can use the head on its own wherever the phonemes already exist.
 *
 * Accuracy rises steeply with how much speech the transcript came from. Measured out of
 * domain on real audio of each length: 3 s scores 0.506, 5 s 0.657, 7 s 0.759, and whole
 * clips averaging 9.7 s reach 0.777. Five seconds is the floor and ten is where the
 * returns run out. Check the audio is mostly speech before transcribing -- silence
 * contributes nothing and inflates the apparent duration.
 *
 * Thread safety: a AsidHead is safe for concurrent asid_classify* calls. Creation and
 * destruction are not; do those from one thread.
 *
 * Licence: Apache-2.0 (code). The model weights follow the corpus licence, CC BY-NC 4.0.
 */
#ifndef GHANA_SPEECH_ID_H
#define GHANA_SPEECH_ID_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ASID_API
#  if defined(_WIN32) && defined(ASID_SHARED)
#    ifdef ASID_BUILDING
#      define ASID_API __declspec(dllexport)
#    else
#      define ASID_API __declspec(dllimport)
#    endif
#  else
#    define ASID_API
#  endif
#endif

typedef struct AsidHead AsidHead;

/* Paths to the artefacts written by scripts/export_onnx.py.
 * config_path may be NULL, in which case the n-gram range defaults to 1..5. */
typedef struct {
  const char *onnx_path;    /* head.onnx or head.fp16.onnx        */
  const char *ngrams_path;  /* ngrams.txt, one n-gram per line    */
  const char *labels_path;  /* labels.txt, one language per line  */
  const char *config_path;  /* head_config.txt, or NULL           */
  int num_threads;          /* onnxruntime intra-op threads; 0 => 1 */
} AsidConfig;

/* Fills cfg with NULLs and num_threads = 1. */
ASID_API void asid_config_init(AsidConfig *cfg);

/* Returns NULL on failure and writes a message into err (if err != NULL). */
ASID_API AsidHead *asid_create(const AsidConfig *cfg, char *err, size_t err_len);
ASID_API void asid_destroy(AsidHead *head);

ASID_API int asid_num_languages(const AsidHead *head);
/* Borrowed pointer, valid until asid_destroy. NULL if index is out of range. */
ASID_API const char *asid_language(const AsidHead *head, int index);

typedef struct {
  int index;         /* into the label list; -1 when undetermined */
  float confidence;  /* softmax probability of that label; 0 when undetermined */
  int num_matched;   /* n-grams found in the vocabulary; 0 means no decision was possible */
} AsidResult;

/* ipa: NUL-terminated UTF-8, units separated by ASCII spaces, exactly as
 * ghana-ipa-asr's Transcript.spaced() produces. Multi-codepoint units such as
 * k͡p, kʰ and t͡ʃ are single tokens and must not be split.
 *
 * A string whose n-grams are all out of vocabulary yields index = -1 with
 * num_matched = 0. Surface that to the user as "unknown" -- it means there was no
 * basis for a decision, not that some language scored poorly. */
ASID_API AsidResult asid_classify(AsidHead *head, const char *ipa);

/* As above but also writes the full posterior. probs must have room for
 * asid_num_languages floats. Returns the number written, or 0 if undetermined. */
ASID_API int asid_classify_probs(AsidHead *head, const char *ipa, float *probs);

/* Version string of this library, e.g. "0.1.0". */
ASID_API const char *asid_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* GHANA_SPEECH_ID_H */
