"""Freeze protocol inputs: build validated temporal features from repaired_jra.db.

Reads  : tmp/repaired_jra.db  (READ-ONLY: uri mode=ro + PRAGMA query_only=ON)
Writes : tmp/features.db (tables entries, history_long, meta)
         tmp/split_manifest.csv (race_id, race_date, split — ALL races)
         tmp/exclusions.csv (every excluded race/row with reason)
Runs leakage/label self-checks and prints them. Idempotent: rebuilds from scratch.
Usage: <venv python> build_features.py
"""
import csv
import argparse
import os
import sqlite3
from bisect import bisect_left
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "repaired_jra.db")
OUT_DB = os.path.join(HERE, "features.db")
MANIFEST = os.path.join(HERE, "split_manifest.csv")
EXCLUSIONS = os.path.join(HERE, "exclusions.csv")

DQ_RACE = "201009040210"  # no rank-2 finisher on source page; suspected DQ -> exclude
UNCERTAIN_HISTORY_RACES = {
    DQ_RACE: "unreconciled finish ranks / incomplete field",
    "201407010204": "incomplete source runner field (jump race)",
}
SEEDS = (0, 1, 2)
NONSTARTER = ("取消", "除外")  # pre-start withdrawals (non-starters) per JRA
NONFINISH_NEGATIVE = ("中止", "失格")  # stopped/disqualified after start -> negatives


def split_of(yyyymmdd):
    y = int(yyyymmdd[:4])
    if 2010 <= y <= 2021:
        return "train"
    if 2022 <= y <= 2023:
        return "validation"
    if 2024 <= y <= 2025:
        return "test"
    raise ValueError("out-of-range date %s" % yyyymmdd)


def ordinal(yyyymmdd):
    return datetime.strptime(yyyymmdd, "%Y%m%d").date().toordinal()


def history_buckets(runners, races):
    """Only eligible, reconciled observed starts may enter a history chain."""
    buckets = {}
    for n in runners:
        if (n["result_status"] in NONSTARTER
                or n["race_id"] in UNCERTAIN_HISTORY_RACES):
            continue
        rd = races[n["race_id"]]["race_date"]
        buckets.setdefault(n["horse_id_netkeiba"], []).append(
            (int(rd), n["race_id"], n))
    for bucket in buckets.values():
        bucket.sort(key=lambda t: (t[0], t[1]))
    return buckets


def previous_three(bucket, target_date):
    """Same-day and later races never enter the three most recent starts."""
    pos = bisect_left([(t[0], t[1]) for t in bucket], (target_date, ""))
    return list(reversed(bucket[max(0, pos - 3):pos]))


def main():
    con = sqlite3.connect("file:%s?mode=ro" % SRC.replace("\\", "/"), uri=True)
    con.execute("PRAGMA query_only=ON")
    con.row_factory = sqlite3.Row

    races = {r["race_id"]: dict(r) for r in con.execute("SELECT * FROM races_r")}
    runners = [dict(r) for r in con.execute(
        "SELECT * FROM runners_r ORDER BY race_id, horse_number")]

    # --- sanity on raw statuses: finished <=> finish_pos present
    bad = [n for n in runners
           if (n["result_status"] == "finished") != (n["finish_pos"] is not None)]
    assert not bad, "status/position mismatch rows: %d" % len(bad)

    # eligible-starter counts per race (non-withdrawal), for field size + norm finish
    elig_count = {}
    for n in runners:
        if n["result_status"] not in NONSTARTER:
            elig_count[n["race_id"]] = elig_count.get(n["race_id"], 0) + 1

    exclusions = []  # (race_id, horse_id, horse_number, level, reason)
    manifest = []
    entries = []
    for race_id, r in races.items():
        manifest.append((race_id, r["race_date"], split_of(r["race_date"])))
    for race_id, r in races.items():
        if r["race_type"] != "flat":
            exclusions.append((race_id, "", "", "race",
                               "non-flat race (race_type=%s)" % r["race_type"]))
        elif race_id == DQ_RACE:
            exclusions.append((race_id, "", "", "race",
                               "suspected DQ: source page shows no rank-2 finisher "
                               "(only 2 positives); excluded, not guessed"))

    excluded_races = {e[0] for e in exclusions}
    for n in runners:
        if n["race_id"] in excluded_races:
            if n["result_status"] in NONSTARTER and n["race_id"] != DQ_RACE:
                pass  # counted under its race exclusion; scratch noted below anyway
            if n["race_id"] == DQ_RACE and n["result_status"] in NONSTARTER:
                exclusions.append((n["race_id"], n["horse_id_netkeiba"],
                                   n["horse_number"], "row",
                                   "pre-start withdrawal (non-starter, status=%s; "
                                   "race also excluded as DQ)" % n["result_status"]))
            continue
        if n["result_status"] in NONSTARTER:
            exclusions.append((n["race_id"], n["horse_id_netkeiba"],
                               n["horse_number"], "row",
                               "pre-start withdrawal (non-starter, status=%s)"
                               % n["result_status"]))
            continue
        r = races[n["race_id"]]
        label = 1 if (n["finish_pos"] in (1, 2, 3)) else 0
        entries.append({
            "race_id": n["race_id"], "horse_id_netkeiba": n["horse_id_netkeiba"],
            "horse_number": n["horse_number"], "split": split_of(r["race_date"]),
            "label": label, "surface": r["surface"], "venue": r["venue"],
            "distance": r["distance"], "going": r["going"],
            "race_class": r["race_class"],
            "field_size": elig_count[n["race_id"]], "bracket": n["bracket"],
            "weight_carried": n["weight_carried"], "age": n["age"],
            "sex": n["sex"], "body_weight": (None if n["body_weight"] == 999 else n["body_weight"]),
            "body_weight_change": n["body_weight_change"]})

    # --- history: prior OBSERVED JRA starts (including reconciled jump races),
    # --- pre-start withdrawals (取消/除外) are not starts; buckets rebuilt so an
    # --- older legitimate start slides into the latest-three. Strictly predating.
    for rid, reason in sorted(UNCERTAIN_HISTORY_RACES.items()):
        if rid in races:
            exclusions.append((rid, "", "", "history_source", reason))
    buckets = history_buckets(runners, races)
    rdate_int = {rid: int(r["race_date"]) for rid, r in races.items()}

    history = []
    for e in entries:
        b = buckets.get(e["horse_id_netkeiba"], [])
        tdate = rdate_int[e["race_id"]]
        priors = previous_three(b, tdate)
        e["hist_count"] = len(priors)
        e["has_hist1"] = 1 if len(priors) > 0 else 0
        e["has_hist2"] = 1 if len(priors) > 1 else 0
        e["has_hist3"] = 1 if len(priors) > 2 else 0
        tord = ordinal(races[e["race_id"]]["race_date"])
        for rank, (pdate, prid, pn) in enumerate(priors, start=1):
            pr = races[prid]
            pfs = elig_count.get(prid)
            norm = (pn["finish_pos"] / pfs) if pn["finish_pos"] and pfs else None
            assert norm is None or 0 < norm <= 1, (prid, pn["horse_id_netkeiba"], norm)
            history.append({
                "race_id": e["race_id"],
                "horse_id_netkeiba": e["horse_id_netkeiba"],
                "horse_number": e["horse_number"], "hist_rank": rank,
                "prior_race_id": prid, "prior_race_date": pr["race_date"],
                "prior_finish_pos": pn["finish_pos"],
                "prior_result_status": pn["result_status"],
                "prior_norm_finish": norm, "prior_field_size": pfs,
                "prior_distance": pr["distance"], "prior_surface": pr["surface"],
                "prior_going": pr["going"], "prior_race_class": pr["race_class"],
                "prior_weight_carried": pn["weight_carried"],
                "days_since": tord - ordinal(pr["race_date"])})

    # --- write outputs to staging paths; publish only after all checks pass
    # This keeps an existing derived dataset usable if a rebuild assertion fails.
    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)
    out = sqlite3.connect(OUT_DB)
    out.row_factory = sqlite3.Row
    out.execute("""CREATE TABLE entries (
        race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
        split TEXT, label INTEGER, surface TEXT, venue TEXT, distance INTEGER,
        going TEXT, race_class TEXT, field_size INTEGER, bracket INTEGER,
        weight_carried REAL, age INTEGER, sex TEXT, body_weight INTEGER,
        body_weight_change INTEGER, hist_count INTEGER,
        has_hist1 INTEGER, has_hist2 INTEGER, has_hist3 INTEGER,
        PRIMARY KEY (race_id, horse_id_netkeiba, horse_number))""")
    out.execute("""CREATE TABLE history_long (
        race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
        hist_rank INTEGER, prior_race_id TEXT, prior_race_date TEXT,
        prior_finish_pos INTEGER, prior_result_status TEXT,
        prior_norm_finish REAL, prior_field_size INTEGER, prior_distance INTEGER,
        prior_surface TEXT, prior_going TEXT, prior_race_class TEXT,
        prior_weight_carried REAL, days_since INTEGER,
        PRIMARY KEY (race_id, horse_id_netkeiba, horse_number, hist_rank))""")
    out.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    ecols = ["race_id", "horse_id_netkeiba", "horse_number", "split", "label",
             "surface", "venue", "distance", "going", "race_class",
             "field_size", "bracket", "weight_carried", "age", "sex",
             "body_weight", "body_weight_change", "hist_count", "has_hist1",
             "has_hist2", "has_hist3"]
    out.executemany("INSERT INTO entries (%s) VALUES (%s)" % (
        ",".join(ecols), ",".join("?" * len(ecols))),
        [[e[c] for c in ecols] for e in entries])
    hcols = ["race_id", "horse_id_netkeiba", "horse_number", "hist_rank",
             "prior_race_id", "prior_race_date", "prior_finish_pos",
             "prior_result_status", "prior_norm_finish", "prior_field_size",
             "prior_distance", "prior_surface", "prior_going",
             "prior_race_class", "prior_weight_carried", "days_since"]
    out.executemany("INSERT INTO history_long (%s) VALUES (%s)" % (
        ",".join(hcols), ",".join("?" * len(hcols))),
        [[h[c] for c in hcols] for h in history])
    out.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
        ("source_db", "tmp/repaired_jra.db (read-only)"),
        ("built_at_utc", datetime.now(timezone.utc).isoformat()),
        ("dq_race_excluded", DQ_RACE),
        ("uncertain_history_races_excluded", ",".join(sorted(UNCERTAIN_HISTORY_RACES))),
        ("seeds", ",".join(map(str, SEEDS))),
        ("history_source", "prior observed JRA starts (same DB; pre-start "
                            "withdrawals 取消/除外 excluded; "
                            "unreconciled history races excluded; weekend collection; "
                            "left-censored at 2010 boundary; NAR/overseas starts absent)"),
        ("odds_policy", "no odds/popularity columns in features.db by design; "
                         "market settings rejoin from repaired_jra.db on entry keys"),
        ("label_policy", "finished 1-3 incl. dead heats=1; 中止/失格=0; "
                         "取消/除外 excluded (pre-start non-starters)")])
    out.execute("CREATE INDEX idx_hist_target ON history_long (race_id)")
    out.execute("CREATE INDEX idx_entries_race ON entries (race_id)")
    out.execute("CREATE INDEX idx_entries_horse ON entries (horse_id_netkeiba)")
    out.commit()

    manifest.sort()
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["race_id", "race_date", "split"])
        w.writerows(manifest)
    exclusions.sort()
    with open(EXCLUSIONS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["race_id", "horse_id_netkeiba", "horse_number",
                    "level", "reason"])
        w.writerows(exclusions)

    # ================= self-checks (printed) =================
    print("== CHECK 1: every history row strictly predates its target ==")
    # (verified in Python from written rows; no cross-DB join needed)
    rdate = {rid: r["race_date"] for rid, r in races.items()}
    late = sum(1 for h in history if h["prior_race_date"] >= rdate[h["race_id"]])
    nonpos = sum(1 for h in history if h["days_since"] <= 0)
    print("history rows=%d late(strictly-predating violations)=%d nonpositive-days_since=%d"
          % (len(history), late, nonpos))
    assert late == 0 and nonpos == 0

    print("== CHECK 2: no target info in features ==")
    banned = {"odds", "popularity", "time", "margin", "last_3f", "corner",
              "finish_pos", "result_status", "top_three"}
    ecols_db = [c[1] for c in out.execute("PRAGMA table_info(entries)")]
    hcols_db = [c[1] for c in out.execute("PRAGMA table_info(history_long)")]
    leak = [c for c in ecols_db if any(b in c.lower() for b in banned)]
    print("entries cols: %s" % ",".join(ecols_db))
    print("target-leak columns in entries: %s" % (leak or "NONE"))
    assert not leak and "odds" not in ecols_db and "popularity" not in ecols_db

    print("== CHECK 3: label counts per split ==")
    for row in out.execute("""SELECT split, COUNT(*), SUM(label),
        ROUND(100.0*SUM(label)/COUNT(*),2) FROM entries GROUP BY split"""):
        print("split=%-10s entries=%7d positives=%6d pos_rate=%s%%" % tuple(row))
    print("-- per surface per split --")
    for row in out.execute("""SELECT split, surface, COUNT(DISTINCT e.race_id),
        COUNT(*), SUM(e.label) FROM entries e GROUP BY split, surface
        ORDER BY split, surface"""):
        print("split=%-10s surface=%-5s races=%5d entries=%7d positives=%6d"
              % tuple(row))

    print("== CHECK 4: dead-heat races keep >3 positives ==")
    dh = out.execute("""SELECT race_id, SUM(label) p FROM entries GROUP BY race_id
        HAVING p > 3 ORDER BY race_id""").fetchall()
    print("races with >3 positives kept: %d" % len(dh))
    for race_id, p in dh[:10]:
        print("  %s positives=%d" % (race_id, p))
    dh5 = out.execute("""SELECT race_id FROM entries GROUP BY race_id
        HAVING SUM(label) = 5""").fetchall()
    print("5-positive race kept (expect 201208040611): %s" % [r[0] for r in dh5])
    assert (len(dh5) == 1 and dh5[0][0] == "201208040611")

    print("== CHECK 5: hand-worked verification of 3 entries ==")
    raw = {(n["race_id"], n["horse_id_netkeiba"], n["horse_number"]): n
           for n in runners}
    demo = []
    d1 = out.execute("""SELECT race_id, horse_id_netkeiba, horse_number FROM entries
        WHERE split='train' AND hist_count=0 ORDER BY race_id, horse_number
        LIMIT 1""").fetchone()
    demo.append(("debut runner, no history (train)", tuple(d1)))
    d2 = out.execute("""SELECT race_id, horse_id_netkeiba, horse_number FROM entries
        WHERE race_id='201208040611' AND label=1 ORDER BY horse_number
        LIMIT 1""").fetchone()
    demo.append(("dead-heat positive (5-positive race)", tuple(d2)))
    d3 = out.execute("""SELECT race_id, horse_id_netkeiba, horse_number FROM entries
        WHERE split='test' AND hist_count=3 ORDER BY race_id, horse_number
        LIMIT 1""").fetchone()
    demo.append(("full 3-start chain (test)", tuple(d3)))
    for title, (rid, hid, hnum) in demo:
        e = out.execute("SELECT * FROM entries WHERE race_id=? AND "
                        "horse_id_netkeiba=? AND horse_number=?",
                        (rid, hid, hnum)).fetchone()
        n = raw[(rid, hid, hnum)]
        print("--- %s: race=%s horse=%s(%s) no.=%s" %
              (title, rid, hid, n["horse_name"], hnum))
        print("    raw: status=%s finish=%s -> label=%d (expect %d)" %
              (n["result_status"], n["finish_pos"], e["label"],
               1 if n["finish_pos"] in (1, 2, 3) else 0))
        assert e["label"] == (1 if n["finish_pos"] in (1, 2, 3) else 0)
        for h in out.execute("SELECT * FROM history_long WHERE race_id=? AND "
                             "horse_id_netkeiba=? AND horse_number=? "
                             "ORDER BY hist_rank", (rid, hid, hnum)):
            pn = raw[(h["prior_race_id"], hid,
                      con.execute("SELECT horse_number FROM runners_r WHERE "
                                  "race_id=? AND horse_id_netkeiba=?",
                                  (h["prior_race_id"], hid)).fetchone()[0])]
            ok = h["prior_race_date"] < rdate[rid]
            print("    rank%d prior=%s date=%s (< %s: %s) raw_finish=%s/%s "
                  "stored_norm=%s days=%s (recomputed %d)" % (
                      h["hist_rank"], h["prior_race_id"], h["prior_race_date"],
                      rdate[rid], ok, pn["finish_pos"], pn["result_status"],
                      h["prior_norm_finish"], h["days_since"],
                      ordinal(rdate[rid]) - ordinal(h["prior_race_date"])))
            assert ok and h["days_since"] == ordinal(rdate[rid]) - ordinal(
                h["prior_race_date"])
        print("    hist_count=%d rows_shown=%d" % (
            e["hist_count"], out.execute(
                "SELECT COUNT(*) FROM history_long WHERE race_id=? AND "
                "horse_id_netkeiba=? AND horse_number=?",
                (rid, hid, hnum)).fetchone()[0]))

    print("== EXCLUSIONS by reason ==")
    from collections import Counter
    c = Counter(e[4] for e in exclusions)
    for reason, k in sorted(c.items()):
        print("  %5d x %s" % (k, reason))
    print("entries=%d history_rows=%d manifest_races=%d exclusion_rows=%d"
          % (len(entries), len(history), len(manifest), len(exclusions)))
    out.close()
    con.close()
    final_db = OUT_DB[:-len(".staging")]
    final_manifest = MANIFEST[:-len(".staging")]
    final_exclusions = EXCLUSIONS[:-len(".staging")]
    os.replace(OUT_DB, final_db)
    os.replace(MANIFEST, final_manifest)
    os.replace(EXCLUSIONS, final_exclusions)
    print("WROTE features.db, split_manifest.csv, exclusions.csv — ALL CHECKS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SRC)
    parser.add_argument("--output-dir", default=HERE)
    args = parser.parse_args()
    SRC = os.path.abspath(args.source)
    os.makedirs(args.output_dir, exist_ok=True)
    OUT_DB = os.path.join(args.output_dir, "features.db.staging")
    MANIFEST = os.path.join(args.output_dir, "split_manifest.csv.staging")
    EXCLUSIONS = os.path.join(args.output_dir, "exclusions.csv.staging")
    final_db = os.path.join(args.output_dir, "features.db")
    source_real = os.path.normcase(os.path.realpath(SRC))
    if source_real in {os.path.normcase(os.path.realpath(OUT_DB)),
                       os.path.normcase(os.path.realpath(final_db))}:
        parser.error("Source database and output database must be different files")
    main()
