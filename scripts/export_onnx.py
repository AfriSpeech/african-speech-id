"""Export the IPA language head to ONNX using standard ops only.

skl2onnx would emit ai.onnx.contrib Tokenizer/TfIdfVectorizer nodes, which mobile
onnxruntime builds (and the one sherpa-onnx bundles) do not carry. So the graph here takes
n-gram *indices and counts* and does the tf-idf arithmetic itself out of Log/Gather/Mul/
ReduceSum/Sqrt/Div/Add -- all opset-13 core ops, available in every runtime.

The app side stays small and is fully specified by ngrams.txt:
  1. split the IPA string on whitespace          (never on characters -- k͡p is one unit)
  2. emit every 1..N-gram, units joined by a space
  3. look each up in ngrams.txt -> index; count occurrences; drop misses
  4. feed (indices int64[K], counts float32[K]) to the graph

Graph reproduces sklearn's TfidfVectorizer(sublinear_tf=True, norm='l2') exactly:
  tf = 1 + log(count);  w = tf * idf[i];  w /= ||w||_2;  logits = W[i]^T w + b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_graph(W: np.ndarray, b: np.ndarray, idf: np.ndarray, fp16: bool) -> onnx.ModelProto:
    """W: [D, C] class weights, b: [C] intercept, idf: [D]."""
    wdt = np.float16 if fp16 else np.float32

    init = [
        numpy_helper.from_array(W.astype(wdt), "W"),
        numpy_helper.from_array(b.astype(np.float32), "b"),
        numpy_helper.from_array(idf.astype(np.float32), "idf"),
        numpy_helper.from_array(np.array([1.0], np.float32), "one"),
        numpy_helper.from_array(np.array([1e-12], np.float32), "eps"),
        numpy_helper.from_array(np.array([-1, 1], np.int64), "col_shape"),
        numpy_helper.from_array(np.array([0], np.int64), "axis0"),
    ]
    n = helper.make_node
    nodes = [
        # sublinear tf: 1 + log(count)
        n("Log", ["counts"], ["logc"]),
        n("Add", ["logc", "one"], ["tf"]),
        # multiply by idf for the present indices
        n("Gather", ["idf", "indices"], ["idf_sel"], axis=0),
        n("Mul", ["tf", "idf_sel"], ["w_raw"]),
        # l2 normalise over the present entries
        n("Mul", ["w_raw", "w_raw"], ["w_sq"]),
        n("ReduceSum", ["w_sq"], ["ss"], keepdims=0),
        n("Sqrt", ["ss"], ["nrm0"]),
        n("Add", ["nrm0", "eps"], ["nrm"]),
        n("Div", ["w_raw", "nrm"], ["w"]),
        # logits = sum_i w_i * W[i, :]  + b
        n("Gather", ["W", "indices"], ["Wsel"], axis=0),          # [K, C]
    ]
    if fp16:
        nodes.append(n("Cast", ["Wsel"], ["Wsel_f"], to=TensorProto.FLOAT))
    else:
        nodes.append(n("Identity", ["Wsel"], ["Wsel_f"]))
    nodes += [
        n("Reshape", ["w", "col_shape"], ["w_col"]),               # [K, 1]
        n("Mul", ["Wsel_f", "w_col"], ["contrib"]),                # [K, C]
        n("ReduceSum", ["contrib", "axis0"], ["acc"], keepdims=0),
        n("Add", ["acc", "b"], ["logits"]),
        n("Softmax", ["logits"], ["probs"], axis=0),
    ]

    graph = helper.make_graph(
        nodes, "ipa_lang_id", initializer=init,
        inputs=[helper.make_tensor_value_info("indices", TensorProto.INT64, ["K"]),
                helper.make_tensor_value_info("counts", TensorProto.FLOAT, ["K"])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["C"]),
                 helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["C"])],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.joblib from train_head.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fp16", action="store_true", help="halves the weight matrix on disk")
    ap.add_argument("--data", default="/mnt/volume_d2wey28/projects/african-speech-id/data/ipa_text.parquet")
    ap.add_argument("--n-check", type=int, default=300)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="record the window size the head was trained on so runtimes vote "
                         "over the same spans")
    ap.add_argument("--chunk-stride", type=int, default=20)
    args = ap.parse_args()

    import joblib
    bundle = joblib.load(args.model)
    vec, clf, labels = bundle["vec"], bundle["clf"], bundle["labels"]

    if not hasattr(clf, "coef_"):
        raise SystemExit("need a linear head (logreg or svm) to export this way")
    W = clf.coef_.T.astype(np.float32)          # [D, C]
    b = np.asarray(clf.intercept_, np.float32).reshape(-1)
    if b.size == 1:
        b = np.repeat(b, W.shape[1])
    idf = vec.idf_.astype(np.float32)
    assert idf.shape[0] == W.shape[0], (idf.shape, W.shape)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = build_graph(W, b, idf, args.fp16)
    mp = outdir / ("head.fp16.onnx" if args.fp16 else "head.onnx")
    onnx.save(m, mp)

    # the app-side vocabulary: one n-gram per line, index = line number
    inv = {i: g for g, i in vec.vocabulary_.items()}
    (outdir / "ngrams.txt").write_text(
        "\n".join(inv[i] for i in range(len(inv))), encoding="utf-8")
    (outdir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    # plain key/value alongside the JSON so the C++ runtime needs no JSON parser
    # analyzer and lowercase are load-bearing: the runtimes read them to decide between
    # phoneme-unit and char_wb tokenisation. Omitting them makes a char head silently
    # tokenise as words, which costs accuracy without raising.
    analyzer = "char" if vec.analyzer == "char_wb" else "word"

    # Case folding is Unicode, not ASCII: Ɛ (U+0190) folds to ɛ (U+025B), which std::tolower
    # cannot do. Rather than depend on ICU or hand-write a table, derive the exact mapping
    # from the characters that actually occur in the vocabulary and ship it. Anything absent
    # here never appears in training, so a runtime can pass it through unchanged.
    fold = {}
    for gram in vec.vocabulary_:
        for ch in gram:
            up = ch.upper()
            if len(up) == 1 and up != ch and up.lower() == ch:
                fold[up] = ch
    (outdir / "casefold.txt").write_text(
        "".join(f"{u}\t{l}\n" for u, l in sorted(fold.items())), encoding="utf-8")
    (outdir / "head_config.txt").write_text(
        f"ngram_min {vec.ngram_range[0]}\n"
        f"ngram_max {vec.ngram_range[1]}\n"
        f"analyzer {analyzer}\n"
        f"lowercase {1 if vec.lowercase else 0}\n"
        f"chunk_chars {args.chunk_chars}\n"
        f"chunk_stride {args.chunk_stride}\n"
        f"sublinear_tf 1\n"
        f"norm l2\n"
        f"n_features {W.shape[0]}\n"
        f"n_classes {W.shape[1]}\n", encoding="utf-8")
    (outdir / "head_config.json").write_text(json.dumps({
        "ngram_range": list(vec.ngram_range),
        "sublinear_tf": True, "norm": "l2", "lowercase": bool(vec.lowercase),
        "analyzer": analyzer,
        "tokenisation": ("char_wb: each word padded with a space either side, n-grams taken "
                         "within the padded word over CODEPOINTS not bytes"
                         if analyzer == "char" else
                         "whitespace-split; units are atomic (k͡p, kʰ, t͡ʃ are single tokens)"),
        "n_features": int(W.shape[0]), "n_classes": int(W.shape[1]),
        "inputs": {"indices": "int64[K] ngram ids", "counts": "float32[K] occurrence counts"},
        "outputs": {"logits": "float32[C]", "probs": "float32[C] softmax"},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # numeric parity against sklearn on real validation strings
    import onnxruntime as ort
    import pyarrow.parquet as pq

    sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
    analyzer = vec.build_analyzer()
    voc = vec.vocabulary_

    print(f"\nexported {mp}  ({mp.stat().st_size/1e6:.1f} MB)")
    print(f"  {W.shape[0]} features x {W.shape[1]} classes")
    print(f"  ngrams.txt {(outdir/'ngrams.txt').stat().st_size/1e6:.1f} MB, "
          f"labels.txt {len(labels)} lines")
    print(f"  analyzer {analyzer}, lowercase {bool(vec.lowercase)}, "
          f"casefold.txt {len(fold)} mappings")
    if args.chunk_chars:
        print(f"  windows {args.chunk_chars} chars / stride {args.chunk_stride}")

    # The African corpus has no "split" column and its text lives in "text"; fall back
    # rather than fail, since this parity check is the only thing that catches a char_wb
    # mismatch between the trainer and the runtimes.
    have = set(pq.read_schema(args.data).names)
    tcol = args.text_col if args.text_col in have else ("ipa" if "ipa" in have else "text")
    cols = [tcol, "language"] + (["split"] if "split" in have else [])
    t = pq.read_table(args.data, columns=cols).to_pydict()
    if "split" not in t:
        t["split"] = ["validation"] * len(t[tcol])
    samples = [(s_, l) for s_, l, sp in zip(t[tcol], t["language"], t["split"])
               if sp == "validation" and s_ and len(s_.split()) >= 5][:args.n_check]
    if not samples:
        print("  no validation rows found for parity check"); return

    from collections import Counter
    agree = 0; maxdiff = 0.0; empties = 0
    for s_, _lang in samples:
        feats = Counter(g for g in analyzer(s_) if g in voc)
        if not feats:
            empties += 1; continue
        idx = np.array([voc[g] for g in feats], np.int64)
        cnt = np.array([feats[g] for g in feats], np.float32)
        onnx_logits = sess.run(["logits"], {"indices": idx, "counts": cnt})[0]
        sk = clf.decision_function(vec.transform([s_]))[0]
        maxdiff = max(maxdiff, float(np.abs(onnx_logits - sk).max()))
        agree += int(np.argmax(onnx_logits) == np.argmax(sk))
    n = len(samples) - empties
    print(f"  parity over {n} validation strings: "
          f"argmax agreement {agree}/{n}, max |logit diff| {maxdiff:.2e}")
    if empties:
        print(f"  ({empties} strings had no in-vocabulary n-grams)")
    if agree != n:
        print("  WARNING: ONNX and sklearn disagree -- do not ship this export")


if __name__ == "__main__":
    main()
