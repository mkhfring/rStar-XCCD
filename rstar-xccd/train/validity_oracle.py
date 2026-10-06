"""Input-validity oracle for code-step training (TRAINING_PLAN_CODESTEP_2026-10-05.txt, Phase 0.3).

Used ONLY by the verifier / reward code after a run; never shown to the model and never used
during search.

    valid(problem, stdin)  :=  >= 2 independent accepted reference solutions (not Code 1 or
                               Code 2) finish within their time limit, exit 0, and print the
                               same output (whitespace-insensitive).

Why not "Code 1 runs": a correct program can loop forever on inputs outside the problem's
rules (PolyHuman pair 98, CF 146B: the accepted Python loops when b is not a lucky number).
What "valid" means here: the input has one well-defined answer that independent correct
solutions agree on. Inputs slightly outside the stated limits can still count as valid if
every reference handles them; that is fine for the verifier, which only needs a defined answer.
Multi-answer and float-tolerance problems must be excluded when the pool is built (1.4).

Layout under `root`:
    <root>/<problem_key>/meta.json          refs kept, their commands, time limits
    <root>/<problem_key>/ref<i>/...         sources + build outputs
    <root>/<problem_key>/verdicts.jsonl     cache: sha1(stdin) -> verdict (append-only)

Usage:
    o = ValidityOracle(root)
    o.add_problem(key, [("CPP", src), ("JAVA", src), ("PY3", src), ...], tests=[(inp, out), ...],
                  time_limit=2.0)
    v = o.check(key, "39999 4774\n")   # Verdict(valid=True, expected="40774", reason="3 refs agree")
"""
import collections
import fcntl
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys

Verdict = collections.namedtuple("Verdict", "valid expected reason")   # valid: True / False / None (unknown)

LANG_ORDER = ("CPP", "JAVA", "PY3")            # preferred: fast and independent implementations
TIME_MULT = {"CPP": 1.0, "JAVA": 2.0, "PY3": 5.0}
MAX_REF_TIMEOUT = 10.0
MEM_BYTES = 2 * 1024 ** 3


def norm(text):
    return " ".join(str(text).split())


def _limit_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))


def _java_class(src):
    m = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", src) or re.search(r"class\s+(\w+)", src)
    return m.group(1) if m else "Main"


def build(lang, src, workdir):
    """Write + compile one reference. Returns (cmd, error or "")."""
    os.makedirs(workdir, exist_ok=True)
    if lang == "PY3":
        path = os.path.join(workdir, "sol.py")
        open(path, "w").write(src)
        return [sys.executable, path], ""
    if lang == "CPP":
        path = os.path.join(workdir, "sol.cpp")
        open(path, "w").write(src)
        exe = os.path.join(workdir, "sol")
        p = subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, path],
                           capture_output=True, text=True, timeout=120)
        return ([exe], "") if p.returncode == 0 else (None, p.stderr[-500:])
    if lang == "JAVA":
        cls = _java_class(src)
        path = os.path.join(workdir, f"{cls}.java")
        open(path, "w").write(src)
        p = subprocess.run(["javac", "-nowarn", "-d", workdir, path],
                           capture_output=True, text=True, timeout=180)
        return (["java", "-Xss64m", "-cp", workdir, cls], "") if p.returncode == 0 else (None, p.stderr[-500:])
    return None, f"unsupported language {lang}"


def run(cmd, stdin, timeout, lang):
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout,
                           preexec_fn=None if lang == "JAVA" else _limit_memory)
        if p.returncode == 0:
            return True, p.stdout, ""
        lines = [l.strip() for l in p.stderr.splitlines() if l.strip() and not l.startswith("Picked up")]
        hits = [l for l in lines if re.search(r"(Error|Exception|panicked)", l)]
        return False, p.stdout, ((hits or lines or ["exit %d" % p.returncode])[0 if hits else -1])[:200]
    except subprocess.TimeoutExpired:
        return False, "", "timeout"


class ValidityOracle:
    def __init__(self, root, max_refs=3):
        self.root = root
        self.max_refs = max_refs
        os.makedirs(root, exist_ok=True)
        self._meta = {}
        self._cache = {}

    # ---------------------------------------------------------------- setup
    def _pdir(self, key):
        return os.path.join(self.root, re.sub(r"[^A-Za-z0-9_.-]", "_", key))

    def add_problem(self, key, solutions, tests, time_limit=2.0, max_attempts=12):
        """Keep up to max_refs solutions that build and pass ALL given tests, preferring
        different languages (CPP, JAVA, PY3). Idempotent: returns the stored meta if present.
        Returns meta with n_refs (< 2 -> check() answers None for this problem)."""
        pdir = self._pdir(key)
        mpath = os.path.join(pdir, "meta.json")
        if os.path.exists(mpath):
            self._meta[key] = json.load(open(mpath))
            return self._meta[key]
        os.makedirs(pdir, exist_ok=True)
        by_lang = collections.defaultdict(list)
        for lang, src in solutions:
            by_lang[lang].append(src)
        order = []                              # round-robin over languages for independence
        for i in range(max(map(len, by_lang.values()), default=0)):
            for lang in LANG_ORDER:
                if i < len(by_lang[lang]):
                    order.append((lang, by_lang[lang][i]))
        refs, rejected, tried = [], collections.Counter(), set()

        def try_ref(idx, lang, src):
            tried.add(idx)
            wd = os.path.join(pdir, f"ref{len(refs)}")
            cmd, _ = build(lang, src, wd)
            if cmd is None:
                rejected["build"] += 1
                shutil.rmtree(wd, ignore_errors=True)
                return
            tl = min(MAX_REF_TIMEOUT, TIME_MULT[lang] * time_limit + 1.0)
            for inp, out in tests:
                ok, got, _ = run(cmd, inp, tl, lang)
                if not (ok and norm(got) == norm(out)):
                    rejected["fails_tests"] += 1
                    shutil.rmtree(wd, ignore_errors=True)
                    return
            refs.append(dict(lang=lang, cmd=cmd, timeout=tl))

        # pass 1: one reference per language (independent implementations); pass 2: fill up.
        for one_per_lang in (True, False):
            for idx, (lang, src) in enumerate(order):
                if len(refs) >= self.max_refs or len(tried) >= max_attempts:
                    break
                if idx in tried or (one_per_lang and any(r["lang"] == lang for r in refs)):
                    continue
                try_ref(idx, lang, src)
        meta = dict(key=key, refs=refs, n_refs=len(refs), n_tests=len(tests),
                    time_limit=time_limit, rejected=dict(rejected), tried=len(tried))
        json.dump(meta, open(mpath, "w"), indent=1)
        self._meta[key] = meta
        return meta

    # ---------------------------------------------------------------- query
    def _load(self, key):
        if key not in self._meta:
            mpath = os.path.join(self._pdir(key), "meta.json")
            if not os.path.exists(mpath):
                return None
            self._meta[key] = json.load(open(mpath))
        if key not in self._cache:
            self._cache[key] = {}
            vpath = os.path.join(self._pdir(key), "verdicts.jsonl")
            if os.path.exists(vpath):
                for line in open(vpath):
                    try:
                        d = json.loads(line)
                        self._cache[key][d["h"]] = Verdict(d["valid"], d["expected"], d["reason"])
                    except (ValueError, KeyError):
                        pass                    # a torn last line from a killed writer
        return self._meta[key]

    def check(self, key, stdin):
        meta = self._load(key)
        if meta is None:
            return Verdict(None, None, "unknown problem")
        if meta["n_refs"] < 2:
            return Verdict(None, None, f"only {meta['n_refs']} reference(s)")
        stdin = str(stdin)
        h = hashlib.sha1(stdin.encode()).hexdigest()
        if h in self._cache[key]:
            return self._cache[key][h]
        outs, why = [], []
        for r in sorted(meta["refs"], key=lambda r: LANG_ORDER.index(r["lang"])):   # fastest first
            ok, out, e = run(r["cmd"], stdin, r["timeout"], r["lang"])
            if not ok:
                why.append(f"{r['lang']}: {e}")
                break                           # one failing reference already makes it invalid
            outs.append(norm(out))
        if why:
            v = Verdict(False, None, "reference failed (" + "; ".join(why) + ")")
        elif len(set(outs)) > 1:
            v = Verdict(False, None, "references disagree")
        else:
            v = Verdict(True, outs[0], f"{len(outs)} refs agree")
        self._cache[key][h] = v
        with open(os.path.join(self._pdir(key), "verdicts.jsonl"), "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(dict(h=h, valid=v.valid, expected=v.expected, reason=v.reason)) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
        return v


def classify_calls(oracle, key, calls):
    """Annotate a node's run_both_calls (python_tool._record dicts) with validity.
    Returns a list of dicts: the call + valid / expected / reason, and for valid inputs which
    program matched the references (match1 / match2)."""
    out = []
    for c in calls:
        if c.get("skipped") or c.get("stdin_len", 0) > len(c.get("stdin", "")):
            out.append(dict(c, valid=None, expected=None, reason="skipped or truncated stdin"))
            continue
        v = oracle.check(key, c["stdin"])
        d = dict(c, valid=v.valid, expected=v.expected, reason=v.reason)
        if v.valid:
            d["match1"] = bool(c["ok1"] and norm(c["out1"]) == v.expected)
            d["match2"] = bool(c["ok2"] and norm(c["out2"]) == v.expected)
        out.append(d)
    return out
