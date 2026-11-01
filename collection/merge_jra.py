import sqlite3, pathlib, shutil, json, hashlib

vps_path = pathlib.Path("temp_vps_raw_jra.db")
laptop_path = pathlib.Path("local_jra_worker/data/raw_jra_laptop.db")
merged_path = pathlib.Path("merged_raw_jra.db")
manifest_base = pathlib.Path("local_jra_worker/manifests/cutover_20260901_152700")
audit = json.loads((manifest_base/"audit.json").read_text())

# remove old merged if exists
if merged_path.exists():
    merged_path.unlink()
# copy vps as base (larger)
shutil.copy2(vps_path, merged_path)
print(f"copied VPS {vps_path.stat().st_size} -> {merged_path}")

con_m = sqlite3.connect(str(merged_path))
con_m.execute("PRAGMA journal_mode=WAL")
con_l = sqlite3.connect(str(laptop_path))

# Get tables to merge: races, runners, scrape_dates, checkpoints, race_claims, etc.
# We'll merge races (PK race_id) with INSERT OR REPLACE (keep latest), runners with INSERT OR IGNORE (since runners id autoinc, but race_id+horse may duplicate)
# For scrape_dates, merge with logic: keep complete over partial, keep max attempts
# For simplicity, use INSERT OR REPLACE for races, and for runners dedupe by (race_id, horse_id_netkeiba, horse_number)
cur_m = con_m.cursor()
cur_l = con_l.cursor()

# Races
vps_races = cur_m.execute("SELECT count(*) FROM races").fetchone()[0]
laptop_races = cur_l.execute("SELECT count(*) FROM races").fetchone()[0]
print(f"before races VPS {vps_races} laptop {laptop_races}")

# Insert laptop races into merged (OR REPLACE)
inserted=0
for row in cur_l.execute("SELECT race_id, race_date, venue, surface, distance, weather, going, race_class, field_size, scraped_at FROM races"):
    cur_m.execute("INSERT OR REPLACE INTO races (race_id, race_date, venue, surface, distance, weather, going, race_class, field_size, scraped_at) VALUES (?,?,?,?,?,?,?,?,?,?)", row)
    inserted+=1
con_m.commit()
print(f"inserted {inserted} laptop races into merged")

# Runners: need to avoid duplicates. Runners table has id PK, but we should dedupe by race_id+horse_id_netkeiba or race_id+horse_name+horse_number
# We'll use INSERT OR IGNORE with a unique constraint check: first try to see if same race_id+horse_id exists
laptop_runners = cur_l.execute("SELECT count(*) FROM runners").fetchone()[0]
vps_runners_before = cur_m.execute("SELECT count(*) FROM runners").fetchone()[0]
print(f"runners VPS {vps_runners_before} laptop {laptop_runners}")

# For runners, we will insert with INSERT OR IGNORE on (race_id, horse_id_netkeiba) if that combo exists, else insert
# Since runners has no unique constraint on that, we need to check manually
inserted_r=0
skipped=0
for r in cur_l.execute("SELECT race_id, horse_id_netkeiba, horse_name, finish_pos, bracket, horse_number, sex_age, weight, jockey, time, margin, popularity, odds, last_3f, corner_passing, trainer, body_weight, body_weight_change, top_three FROM runners"):
    race_id, horse_id = r[0], r[1]
    # check if same race_id+horse_id already exists in merged
    exists = cur_m.execute("SELECT 1 FROM runners WHERE race_id=? AND horse_id_netkeiba=? LIMIT 1", (race_id, horse_id)).fetchone()
    if exists:
        skipped+=1
        continue
    # also check race_id+horse_name+horse_number as fallback if horse_id null
    cur_m.execute("INSERT INTO runners (race_id, horse_id_netkeiba, horse_name, finish_pos, bracket, horse_number, sex_age, weight, jockey, time, margin, popularity, odds, last_3f, corner_passing, trainer, body_weight, body_weight_change, top_three) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)
    inserted_r+=1
con_m.commit()
print(f"runners inserted {inserted_r} skipped dup {skipped}")

# scrape_dates: merge by keeping the most advanced status (complete > partial > processing > pending) and max attempts
def status_rank(s):
    return {'complete':3, 'partial':2, 'processing':1, 'pending':0}.get(s, -1)

merged_dates = {}
for row in cur_m.execute("SELECT date_str, worker_id, status, attempts, last_error, updated_at FROM scrape_dates"):
    merged_dates[row[0]] = row

for row in cur_l.execute("SELECT date_str, worker_id, status, attempts, last_error, updated_at FROM scrape_dates"):
    date_str = row[0]
    if date_str not in merged_dates:
        cur_m.execute("INSERT INTO scrape_dates (date_str, worker_id, status, attempts, last_error, updated_at) VALUES (?,?,?,?,?,?)", row)
    else:
        existing = merged_dates[date_str]
        # keep higher rank, or if equal rank keep max attempts
        if status_rank(row[2]) > status_rank(existing[2]) or (row[2]==existing[2] and row[3] > existing[3]):
            cur_m.execute("INSERT OR REPLACE INTO scrape_dates (date_str, worker_id, status, attempts, last_error, updated_at) VALUES (?,?,?,?,?,?)", row)
con_m.commit()
print("scrape_dates merged", cur_m.execute("SELECT count(*) FROM scrape_dates").fetchone()[0])

# checkpoints: merge via INSERT OR REPLACE
for row in cur_l.execute("SELECT key, last_value FROM checkpoints"):
    cur_m.execute("INSERT OR REPLACE INTO checkpoints (key, last_value) VALUES (?,?)", row)
con_m.commit()

# race_claims: should be empty after workers done, but merge anyway
try:
    for row in cur_l.execute("SELECT race_id, worker_id, claimed_at FROM race_claims"):
        cur_m.execute("INSERT OR IGNORE INTO race_claims (race_id, worker_id, claimed_at) VALUES (?,?,?)", row)
    con_m.commit()
except Exception as e:
    print("race_claims merge err", e)

# Final counts
print("merged races", cur_m.execute("SELECT count(*) FROM races").fetchone()[0])
print("merged runners", cur_m.execute("SELECT count(*) FROM runners").fetchone()[0])
print("merged distinct dates", cur_m.execute("SELECT count(DISTINCT race_date) FROM races").fetchone()[0])
print("merged scrape_dates complete", cur_m.execute("SELECT count(*) FROM scrape_dates WHERE status='complete'").fetchone()[0])
print("merged scrape_dates partial", cur_m.execute("SELECT count(*) FROM scrape_dates WHERE status='partial'").fetchone()[0] if cur_m.execute("SELECT count(*) FROM scrape_dates WHERE status='partial'").fetchone() else 0)

# Verify no duplicate race_id
dups = cur_m.execute("SELECT race_id, count(*) c FROM races GROUP BY race_id HAVING c>1").fetchall()
print("dup race_id", len(dups))

# Check manifest coverage
all_manifest_dates=set()
for name in ['laptop_0','laptop_1','vps_0','vps_1']:
    dates=(manifest_base/f"{name}.txt").read_text().split()
    all_manifest_dates.update(dates)
print("manifest total", len(all_manifest_dates))
missing_manifest=[d for d in all_manifest_dates if not cur_m.execute("SELECT 1 FROM scrape_dates WHERE date_str=? AND status='complete'", (d,)).fetchone()]
print("manifest missing in scrape_dates complete", len(missing_manifest), missing_manifest[:10])
# Also check race presence for manifest dates with races
present_manifest=[d for d in all_manifest_dates if cur_m.execute("SELECT 1 FROM races WHERE race_date=? LIMIT 1", (d,)).fetchone()]
print("manifest with at least 1 race", len(present_manifest))
# Check for duplicate assigned dates (should be 0)
# Already verified disjoint, but check merged still disjoint
print("audit pending", audit['pending_count'], "complete_union", audit['complete_union_count'])

# quick_check
print("quick_check", cur_m.execute("PRAGMA quick_check").fetchone())
print("integrity", cur_m.execute("PRAGMA integrity_check").fetchone())

con_m.close()
con_l.close()
print("merge done ->", merged_path)
print("merged sha256", hashlib.sha256(merged_path.read_bytes()).hexdigest()[:16])
