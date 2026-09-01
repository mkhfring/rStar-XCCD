# Constructing Python-Centric Cross-Lingual Clone Datasets from CLCCD

## 1. Source data

The datasets described here are derived from **CLCCD** (Cross-Lingual Code Clone
Detection), the data and code release accompanying "The Struggles of LLMs in
Cross-lingual Code Clone Detection" (TruX-DTF/CLCCD,
https://github.com/TruX-DTF/CLCCD, commit `2e4c813`). CLCCD packages two
JSON files of labeled cross-language code pairs:

- `data/java_cn.json` — 6,000 pairs derived from **Project CodeNet**, an AtCoder
  competitive-programming submission corpus. Each of 10 language pairs
  (Java paired with C, C#, C++, Go, JavaScript, OCaml, PHP, Python, Ruby, and
  Rust) contributes 300 clone and 300 non-clone examples.
- `data/java_xl.json` — 6,000 pairs derived from **XLCoST**, contributing 5
  language pairs (Java with C#, C++, JavaScript, PHP, and Python), each with
  600 clone and 600 non-clone examples.

In both files, every pair is constructed with **Java as one member of the
pair** (`ll1 = "Java"`); the second language varies (`ll2`). Each record has
the schema:

```json
{
  "codeA": "<Java source code>",
  "codeB": "<source code in the second language>",
  "ll1": "Java",
  "ll2": "<second language>",
  "problem_id_1": "<CodeNet/XLCoST problem id for codeA>",
  "problem_id_2": "<CodeNet/XLCoST problem id for codeB>",
  "type": "clone" | "nonclone"
}
```

For `type = "clone"`, `problem_id_1 == problem_id_2`: both snippets are
accepted solutions to the *same* programming problem, and are therefore
treated as semantic clones regardless of surface-level (textual) similarity.
For `type = "nonclone"`, `problem_id_1 != problem_id_2`: the two snippets
solve different problems and are treated as non-clones. `java_cn.json` uses
CodeNet's `pXXXXX`-style problem identifiers; `java_xl.json` uses XLCoST's
own numeric identifiers. The two identifier namespaces are disjoint, so
records from the two files can be pooled without collision.

CLCCD does not ship every language pair with equal support: only Java is
guaranteed to co-occur with every other language. Because our target task is
Python-centric clone detection, two different construction strategies were
used depending on whether the second language is Java or not.

## 2. Target schema

All datasets produced for this project follow the schema already used
throughout `rstar-xccd/eval_data`, i.e. one JSON object per line with three
fields:

```json
{"index": <int>, "question": "<prompt>", "answer": "clone" | "non-clone"}
```

`question` follows a fixed template:

```
Determine whether the following two code snippets are semantic code clones.

Code 1: Python
```python
<python source code>
```

Code 2: <Language>
```<language tag>
<other-language source code>
```
```

`answer` is the ground-truth label, `"clone"` or `"non-clone"`. Python is
always presented as "Code 1"; the paired language is always "Code 2".

## 3. Construction strategy A — Python–Java (direct pairing)

Since Java is the pivot language in CLCCD, Python–Java pairs occur directly
in the raw data (`ll2 = "Python"`), with no additional processing needed. All
matching records from `java_cn.json` and `java_xl.json` were pooled:

| Source        | Clone pairs | Non-clone pairs |
|---------------|------------:|-----------------:|
| `java_cn.json`|         300 |               300 |
| `java_xl.json`|         600 |               600 |
| **Total**     |     **900** |           **900** |

For each record, `codeB` (Python) becomes "Code 1" and `codeA` (Java) becomes
"Code 2", and `type` is mapped to `answer` (`clone` → `"clone"`,
`nonclone` → `"non-clone"`). The 1,800 resulting records were shuffled and
re-indexed (`0`–`1799`). This produced `test_python_java_CLCCD.jsonl`.

## 4. Construction strategy B — Python–X pairing via a Java bridge

For X ∈ {Ruby, Rust, Go, C}, CLCCD contains no direct Python–X records.
However, `java_cn.json` contains, for a shared pool of 300 CodeNet problems,
*both* a Java–Python clone pair and a Java–X clone pair keyed to the same
`problem_id`. Because a "clone" record's `codeB` is, by construction, an
accepted solution to `problem_id_1`, this makes it possible to derive genuine
Python–X pairs by joining on the shared problem identifier, discarding the
Java code entirely:

1. From all `type = "clone"` records with `ll2 = "Python"`, build a mapping
   `problem_id → Python code`.
2. From all `type = "clone"` records with `ll2 = X`, build a mapping
   `problem_id → X code`.
3. Intersect the two problem-id sets. For `java_cn.json` this intersection
   is exactly 300 problems for every X ∈ {Ruby, Rust, Go, C, C#, C++, PHP,
   JavaScript, OCaml} — i.e., CodeNet coverage is uniform across languages.
4. For each shared problem `p`, emit `(Python[p], X[p])` as a **clone** pair:
   both snippets solve the same problem, hence are semantic clones by the
   same criterion CLCCD itself uses for Java-anchored pairs.

Note that the raw `type = "nonclone"` records **cannot** be reused the same
way. CLCCD builds `problem_id_2` for a non-clone record from a distance
computed purely between problem *descriptions*, independent of language; the
same `(problem_id_1, problem_id_2)` mapping is reused across every language
paired with `problem_id_1`. Joining two languages' non-clone records on
`problem_id_1` would therefore, in most cases, yield two snippets that both
solve `problem_id_2` — i.e. an actual clone pair mislabeled as non-clone.
Non-clone Python–X pairs were instead generated directly from the clone-pool
codes:

5. Build one or more random derangements of the 300 shared problem ids (a
   permutation with no fixed point, so no code is ever paired with the
   solution to its own problem). Successive derangements are constrained to
   introduce no `(python_problem, X_problem)` combination already used in an
   earlier round, so repeated non-clone pairs do not occur.
6. For each derangement round and each problem `p`, emit
   `(Python[p], X[shuffled(p)])` as a **non-clone** pair: the two snippets
   are guaranteed to solve different problems.

For Ruby, a single derangement round was used (300 clone + 300 non-clone =
600 total), matching CLCCD's native 1:1 clone/non-clone balance. For Rust,
Go, and C, three derangement rounds were used to reach a target size above
1,000 samples, at the cost of a 1:3 clone/non-clone ratio (300 clone + 900
non-clone = 1,200 total per language). This trade-off reflects a hard
constraint of the source data: CLCCD provides only 300 *problems* with both a
Python and a Ruby/Rust/Go/C solution, so 300 is the maximum number of
genuinely distinct clone pairs obtainable for these language pairs without
introducing synthetic code; non-clone pairs, by contrast, can be legitimately
multiplied because any two different-problem snippets are a valid negative
example.

## 5. Resulting datasets

All files live in `rstar-xccd/eval_data/` and share the `{index, question,
answer}` schema described in Section 2.

| File                              | Language pair  | Construction | Clone | Non-clone | Total |
|------------------------------------|----------------|--------------|------:|----------:|------:|
| `test_python_java_CLCCD.jsonl`     | Python–Java    | Direct (A)   |   900 |       900 | 1,800 |
| `test_python_ruby_CLCCD.jsonl`     | Python–Ruby    | Bridged (B)  |   300 |       300 |   600 |
| `test_python_rust_CLCCD.jsonl`     | Python–Rust    | Bridged (B)  |   300 |       900 | 1,200 |
| `test_python_go_CLCCD.jsonl`       | Python–Go      | Bridged (B)  |   300 |       900 | 1,200 |
| `test_python_c_CLCCD.jsonl`        | Python–C       | Bridged (B)  |   300 |       900 | 1,200 |

## 6. Limitations

- **Pool size.** The bridged pairs (Ruby, Rust, Go, C) are limited to the 300
  CodeNet problems for which CLCCD happens to retain both a Python and an X
  solution. This is smaller than the Java-anchored pairs (900, pooling two
  source datasets) and smaller than the hand-curated, non-CLCCD Python–Java
  and Python–Rust files already present elsewhere in this project.
- **Class balance.** Ruby and Java pairs are balanced 1:1. Rust, Go, and C
  pairs are 1:3 (clone:non-clone) as a direct consequence of oversampling
  non-clone examples to exceed 1,000 total samples; this should be accounted
  for (e.g. via class weighting or balanced accuracy) in any evaluation that
  uses these files.
- **Indirect provenance.** Unlike the direct Java pairs, the Python–X (X ≠
  Java) clone/non-clone labels are not literally present in CLCCD; they are
  derived from CLCCD's problem-id annotations under the same "same problem ⇒
  clone" assumption CLCCD itself uses to construct its Java-anchored pairs.
- **Single problem source.** All bridged pairs originate from `java_cn.json`
  (CodeNet) only, since `java_xl.json` (XLCoST) does not include Ruby, Rust,
  Go, or C.
