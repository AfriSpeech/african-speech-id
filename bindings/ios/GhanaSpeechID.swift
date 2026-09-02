// Swift wrapper over the african-speech-id C API.
//
// The C header is imported directly through module.modulemap, so there is no Objective-C
// shim and no bridging header. Add libafrican_speech_id.a and onnxruntime to the target,
// point the module map at include/african_speech_id.h, and this file is the whole binding.
//
//   let id = try LanguageIdentifier(modelDirectory: bundleURL)
//   if let r = id.classify(ipa: transcript) { print(r.language, r.confidence) }
//
// The IPA string comes from sherpa-onnx running ghana-speech-phoneme-asr; this library
// takes it from there. Units are space separated and multi-scalar units such as k͡p and t͡ʃ
// must stay whole, which they do as long as you pass the recogniser's output unmodified.

import Foundation

public struct LanguageResult: Sendable, Equatable {
    /// Predicted language label, e.g. "Twi_twi".
    public let language: String
    /// Index into `LanguageIdentifier.languages`.
    public let index: Int
    /// Model score for `language`. Calibrated only when the head was trained with
    /// logistic regression; for the SVM head prefer `margin` as a confidence signal.
    public let confidence: Float
    /// Number of n-grams that were found in the vocabulary.
    public let matchedNGrams: Int
}

public enum LanguageIdentifierError: Error, CustomStringConvertible {
    case missingArtefact(String)
    case createFailed(String)

    public var description: String {
        switch self {
        case .missingArtefact(let f): return "missing model file: \(f)"
        case .createFailed(let m):    return "could not create language identifier: \(m)"
        }
    }
}

public final class LanguageIdentifier {
    private let handle: OpaquePointer
    /// Labels in model order.
    public let languages: [String]

    /// - Parameters:
    ///   - modelDirectory: directory containing head.onnx (or head.fp16.onnx), ngrams.txt,
    ///     labels.txt and head_config.txt, as written by scripts/export_onnx.py.
    ///   - threads: onnxruntime intra-op threads. 1 is right for a phone; the head is a
    ///     single sparse gather and costs microseconds.
    public init(modelDirectory: URL, threads: Int32 = 1) throws {
        let fm = FileManager.default
        func path(_ name: String) -> String {
            modelDirectory.appendingPathComponent(name).path
        }
        var onnx = path("head.onnx")
        if !fm.fileExists(atPath: onnx) { onnx = path("head.fp16.onnx") }
        let ngrams = path("ngrams.txt")
        let labels = path("labels.txt")
        let config = path("head_config.txt")

        for f in [onnx, ngrams, labels] where !fm.fileExists(atPath: f) {
            throw LanguageIdentifierError.missingArtefact((f as NSString).lastPathComponent)
        }

        var cfg = AsidConfig()
        asid_config_init(&cfg)
        cfg.num_threads = threads

        var err = [CChar](repeating: 0, count: 512)
        // The C strings must outlive asid_create, so keep them alive for the whole call.
        let created: OpaquePointer? = onnx.withCString { pOnnx in
            ngrams.withCString { pNgrams in
                labels.withCString { pLabels in
                    config.withCString { pConfig in
                        cfg.onnx_path = pOnnx
                        cfg.ngrams_path = pNgrams
                        cfg.labels_path = pLabels
                        cfg.config_path = fm.fileExists(atPath: config) ? pConfig : nil
                        return asid_create(&cfg, &err, err.count)
                    }
                }
            }
        }
        guard let h = created else {
            throw LanguageIdentifierError.createFailed(String(cString: err))
        }
        handle = h

        let n = Int(asid_num_languages(h))
        languages = (0..<n).map { i in
            asid_language(h, Int32(i)).map { String(cString: $0) } ?? "?"
        }
    }

    deinit { asid_destroy(handle) }

    /// Returns nil when no n-gram matched, meaning there was no basis for a decision.
    /// Surface that as "unknown" rather than picking a language.
    public func classify(ipa: String) -> LanguageResult? {
        let r = ipa.withCString { asid_classify(handle, $0) }
        guard r.index >= 0, Int(r.index) < languages.count else { return nil }
        return LanguageResult(language: languages[Int(r.index)],
                              index: Int(r.index),
                              confidence: r.confidence,
                              matchedNGrams: Int(r.num_matched))
    }

    /// Full posterior in model order, or nil when nothing matched.
    public func posterior(ipa: String) -> [String: Float]? {
        var buf = [Float](repeating: 0, count: languages.count)
        let n = ipa.withCString { s in
            buf.withUnsafeMutableBufferPointer { asid_classify_probs(handle, s, $0.baseAddress) }
        }
        guard n > 0 else { return nil }
        return Dictionary(uniqueKeysWithValues: zip(languages, buf))
    }

    /// Top-1 score minus top-2. For the SVM head this separates confident from marginal
    /// decisions far better than the raw softmax value does.
    public func margin(ipa: String) -> Float? {
        var buf = [Float](repeating: 0, count: languages.count)
        let n = ipa.withCString { s in
            buf.withUnsafeMutableBufferPointer { asid_classify_probs(handle, s, $0.baseAddress) }
        }
        guard n > 1 else { return nil }
        let sorted = buf.sorted(by: >)
        return sorted[0] - sorted[1]
    }

    public static var version: String { String(cString: asid_version()) }
}
