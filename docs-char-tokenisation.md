# Reproducing `char_wb` outside scikit-learn

If the orthographic head ships, the C++ library and any other binding must reproduce
scikit-learn's `analyzer="char_wb"` exactly. A mismatch does not raise — it silently
degrades accuracy on device, which is the worst way for this to fail.

Verified behaviour (sklearn 1.9):

## Word padding

Text is split on whitespace and **each word is padded with one space on each side**, then
n-grams are taken within that padded word. N-grams never cross a word boundary.

```
"me kɔ", n=2  ->  ' m', 'me', 'e ', ' k', 'kɔ', 'ɔ '
"me kɔ", n=3  ->  ' me', 'me ', ' kɔ', 'kɔ '
"ab",    n=3  ->  ' ab', 'ab '
```

With `ngram_range=(1, n)` every order from 1 to n is emitted, including the bare `' '`.

## Characters are codepoints, not bytes

`kɔ` is **2 characters and 3 bytes**. `char_wb` counts codepoints.

```
'kɔ' -> 2 chars, 3 bytes
'ɔ ' -> 2 chars, 3 bytes
```

A byte-based implementation would cut `ɔ` (U+0254) in half and generate n-grams that never
appear in the vocabulary — producing a model that loads, runs, and quietly misclassifies.
**Iterate by UTF-8 codepoint.**

## Case folding

`lowercase=True` applies Unicode case folding, which covers Ghanaian capitals:

```
Ɛ -> ɛ    Ɔ -> ɔ    Ŋ -> ŋ    Ʋ -> ʋ    É -> é
```

`std::tolower` works on single bytes and cannot do this: `Ɛ` is U+0190 folding to U+025B,
a different codepoint of the same byte length, and neither is ASCII.

A C++ port therefore needs either an ICU dependency or a hand-written folding table for
every capital appearing in the 42 orthographies. **Training with `--no-lowercase` removes
the requirement entirely**, at whatever accuracy it costs. Measure before inheriting it.
