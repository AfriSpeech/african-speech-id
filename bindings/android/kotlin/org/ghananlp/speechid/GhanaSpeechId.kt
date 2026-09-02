package org.ghananlp.speechid

import java.io.Closeable
import java.io.File

/**
 * Language identification over IPA phoneme strings.
 *
 * The phonemes come from sherpa-onnx running ghana-speech-phoneme-asr; this class takes it
 * from there. Pass the recogniser's output unmodified -- units are space separated and
 * multi-character units such as k͡p, kʰ and t͡ʃ must not be split.
 *
 * Not thread safe to create or close concurrently; [classify] and [posterior] are safe to
 * call from multiple threads on one instance.
 */
class AfricanSpeechId private constructor(private var handle: Long) : Closeable {

    /** Labels in model order. */
    val languages: List<String> =
        (0 until nativeNumLanguages(handle)).map { nativeLanguage(handle, it) ?: "?" }

    data class Result(
        val language: String,
        val index: Int,
        /**
         * Model score. Calibrated only for a logistic-regression head; with the SVM head
         * use [margin] instead as the confidence signal.
         */
        val confidence: Float,
        /** n-grams found in the vocabulary. */
        val matchedNGrams: Int,
    )

    /** Returns null when no n-gram matched -- there was no basis for a decision. */
    fun classify(ipa: String): Result? {
        val r = nativeClassify(handle, ipa.toByteArray(Charsets.UTF_8))
        val idx = r[0].toInt()
        if (idx < 0 || idx >= languages.size) return null
        return Result(languages[idx], idx, r[1], r[2].toInt())
    }

    /** Full posterior in model order, or null when nothing matched. */
    fun posterior(ipa: String): Map<String, Float>? {
        val p = nativeClassifyProbs(handle, ipa.toByteArray(Charsets.UTF_8)) ?: return null
        return languages.zip(p.toTypedArray()).toMap()
    }

    /** Top-1 minus top-2. A better confidence signal than the raw score for an SVM head. */
    fun margin(ipa: String): Float? {
        val p = nativeClassifyProbs(handle, ipa.toByteArray(Charsets.UTF_8)) ?: return null
        if (p.size < 2) return null
        p.sort()
        return p[p.size - 1] - p[p.size - 2]
    }

    override fun close() {
        if (handle != 0L) { nativeDestroy(handle); handle = 0L }
    }

    companion object {
        init { System.loadLibrary("african_speech_id_jni") }

        /**
         * @param modelDir directory holding head.onnx (or head.fp16.onnx), ngrams.txt,
         *   labels.txt and head_config.txt, as written by scripts/export_onnx.py.
         *   Assets must be extracted to real files first -- onnxruntime and the vocabulary
         *   reader both need filesystem paths, not an AssetManager stream.
         * @param threads onnxruntime intra-op threads; 1 is right for a phone.
         */
        @JvmStatic
        @JvmOverloads
        fun create(modelDir: File, threads: Int = 1): AfricanSpeechId {
            val onnx = File(modelDir, "head.onnx").let {
                if (it.exists()) it else File(modelDir, "head.fp16.onnx")
            }
            val ngrams = File(modelDir, "ngrams.txt")
            val labels = File(modelDir, "labels.txt")
            val config = File(modelDir, "head_config.txt")
            for (f in listOf(onnx, ngrams, labels)) {
                require(f.exists()) { "missing model file: ${f.name}" }
            }
            val err = arrayOfNulls<String>(1)
            val h = nativeCreate(
                onnx.absolutePath, ngrams.absolutePath, labels.absolutePath,
                if (config.exists()) config.absolutePath else "", threads, err,
            )
            check(h != 0L) { "could not create language identifier: ${err[0]}" }
            return AfricanSpeechId(h)
        }

        @JvmStatic private external fun nativeCreate(
            onnx: String, ngrams: String, labels: String, config: String,
            threads: Int, err: Array<String?>,
        ): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
        @JvmStatic private external fun nativeNumLanguages(handle: Long): Int
        @JvmStatic private external fun nativeLanguage(handle: Long, index: Int): String?
        @JvmStatic private external fun nativeClassify(handle: Long, ipa: ByteArray): FloatArray
        @JvmStatic private external fun nativeClassifyProbs(handle: Long, ipa: ByteArray): FloatArray?
    }
}
