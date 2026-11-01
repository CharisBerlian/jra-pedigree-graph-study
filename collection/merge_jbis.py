"""Merge 4 raw_jbis.db (2 VPS shards in one DB + 2 laptop shards in one DB) into merged_raw_jbis.db.
Run only after all 4 workers reach Worker complete. Interim runs show progress only.
"""
import sqlite3, pathlib
vps_path = pathlib.Path("temp_vps_raw_jbis.db")  # download from /opt/jra_scraper/data/raw_jbis.db before running
laptop_path = pathlib.Path("local_jra_worker/data/raw_jbis_laptop.db")
merged_path = pathlib.Path("merged_raw_jbis.db")
import shutil
if not vps_path.exists():
    print(f"missing {vps_path} - scp /opt/jra_scraper/data/raw_jbis.db to {vps_path} first")
    raise SystemExit(2)
if merged_path.exists():
    merged_path.unlink()
shutil.copy2(vps_path, merged_path)
print(f"copied VPS {vps_path.stat().st_size}")
con_m = sqlite3.connect(str(merged_path))
con_m.execute("PRAGMA journal_mode=WAL")
con_m.execute(f"ATTACH DATABASE '{laptop_path.resolve()}' AS laptop")
print("vps edges", con_m.execute("SELECT count(*) FROM pedigree_edges").fetchone()[0])
print("laptop edges", con_m.execute("SELECT count(*) FROM laptop.pedigree_edges").fetchone()[0])
con_m.execute("INSERT OR IGNORE INTO main.pedigree_edges SELECT * FROM laptop.pedigree_edges")
con_m.execute("INSERT OR IGNORE INTO main.pedigree_checkpoint SELECT * FROM laptop.pedigree_checkpoint")
con_m.execute("INSERT OR IGNORE INTO main.netkeiba_to_jbis SELECT * FROM laptop.netkeiba_to_jbis")
con_m.execute("INSERT OR IGNORE INTO main.search_checkpoint SELECT * FROM laptop.search_checkpoint")
con_m.commit()
con_m.execute("DETACH DATABASE laptop")
print("merged edges", con_m.execute("SELECT count(*) FROM pedigree_edges").fetchone()[0])
print("merged pedigree_checkpoint", con_m.execute("SELECT count(*) FROM pedigree_checkpoint").fetchone()[0])
print("gen counts", list(con_m.execute("SELECT generation, count(*) FROM pedigree_edges GROUP BY generation")))
print("quick_check", con_m.execute("PRAGMA quick_check").fetchone()[0])
con_m.close()
print("done", merged_path.stat().st_size)
