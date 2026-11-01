"""Build a versioned, read-only results release from locked artifacts."""
import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path


FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")
CONTRASTS = ("S6-S2", "S6-S3", "S6-S5", "S7-M2", "S7-M3", "S7-S6")
METRICS = ("average_precision", "mean_hit_at_3",
           "ndcg_binary_all_races", "ndcg_graded_at_3")


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_source(path, destination):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {"path": str(path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def parse_ci(value):
    if value in (None, "", "None"):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return value


def flatten_contrasts(rows):
    output = []
    for row in rows:
        ci = parse_ci(row.get("ci95"))
        item = {key: value for key, value in row.items()
                if key not in ("distribution", "seed_point_estimates", "ci95")}
        item["ci95_lower"] = ci[0] if ci else None
        item["ci95_upper"] = ci[1] if ci else None
        output.append(item)
    return output


def ndcg_rows(summary_rows, supplement_report):
    denominators = supplement_report["cohorts"]
    output = []
    for row in summary_rows:
        for metric in ("ndcg_binary_all_races", "ndcg_binary_graded_races",
                       "ndcg_graded_at_3"):
            ci = parse_ci(row.get(f"{metric}_ci95"))
            denominator = (denominators[row["surface"]]["races"]
                           if metric == "ndcg_binary_all_races"
                           else denominators[row["surface"]]["graded_races"])
            output.append({
                "family": row["family"], "surface": row["surface"],
                "setting": row["setting"], "metric": metric,
                "mean": row.get(f"{metric}_mean"),
                "seed_sd": row.get(f"{metric}_seed_sd"),
                "ci95_lower": ci[0] if ci else None,
                "ci95_upper": ci[1] if ci else None,
                "race_denominator": denominator,
                "bootstrap_replicates": supplement_report["bootstrap"]["replicates"],
            })
    return output


def prediction_inventory(root):
    root = Path(root)
    output = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name not in ("test_predictions.csv.gz",
                                                    "test_predictions.csv"):
            continue
        relative = path.relative_to(root).as_posix()
        output.append({"relative_path": relative, "sha256": sha256_file(path),
                       "bytes": path.stat().st_size})
    if len(output) != 180:
        raise ValueError(f"expected 180 prediction files, found {len(output)}")
    return output


def svg_escape(value):
    return html.escape(str(value), quote=True)


def forest_svg(path, records, title, label_key, estimate_key="estimate",
               lower_key="ci95_lower", upper_key="ci95_upper", width=1400):
    records = list(records)
    if not records:
        raise ValueError("forest plot has no records")
    values = []
    for row in records:
        values.extend(float(row[key]) for key in (estimate_key, lower_key, upper_key)
                      if row.get(key) not in (None, "", "None"))
    low, high = min(values), max(values)
    span = high - low or 1.0
    low -= span * 0.08
    high += span * 0.08
    left, right = 380, width - 70
    top, row_height = 62, 26
    height = top + row_height * len(records) + 52

    def x(value):
        return left + (float(value) - low) / (high - low) * (right - left)

    colors = {"turf": "#1f77b4", "dirt": "#d62728"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#202020}.small{font-size:12px}'
             '.title{font-size:18px;font-weight:bold}.axis{stroke:#777;stroke-width:1}'
             '.grid{stroke:#ddd;stroke-width:1}.zero{stroke:#444;stroke-width:1.3}'
             '</style>',
             f'<text x="20" y="26" class="title">{svg_escape(title)}</text>']
    zero = x(0.0) if low <= 0 <= high else None
    if zero is not None:
        parts.append(f'<line x1="{zero:.2f}" y1="45" x2="{zero:.2f}" '
                     f'y2="{height - 30}" class="zero"/>')
    for tick in range(6):
        value = low + (high - low) * tick / 5
        xpos = x(value)
        parts.append(f'<line x1="{xpos:.2f}" y1="42" x2="{xpos:.2f}" '
                     f'y2="{height - 30}" class="grid"/>')
        parts.append(f'<text x="{xpos:.2f}" y="{height - 10}" text-anchor="middle" '
                     f'class="small">{value:.3f}</text>')
    for index, row in enumerate(records):
        ypos = top + index * row_height
        color = colors.get(str(row.get("surface", "")), "#555")
        parts.append(f'<text x="{left - 12}" y="{ypos + 4}" text-anchor="end" '
                     f'class="small">{svg_escape(row[label_key])}</text>')
        lower = row.get(lower_key)
        upper = row.get(upper_key)
        estimate = row.get(estimate_key)
        if lower not in (None, "", "None") and upper not in (None, "", "None"):
            parts.append(f'<line x1="{x(lower):.2f}" y1="{ypos}" x2="{x(upper):.2f}" '
                         f'y2="{ypos}" stroke="{color}" stroke-width="3"/>')
        if estimate not in (None, "", "None"):
            parts.append(f'<circle cx="{x(estimate):.2f}" cy="{ypos}" r="4" '
                         f'fill="{color}"/>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def shap_svg(path, results, width=1200):
    groups = ("graph_embeddings", "graph_cluster", "graph_availability")
    panel_width, panel_height = 285, 210
    margin_x, margin_y = 20, 52
    height = margin_y + panel_height * len(results) + 30
    colors = {"graph_embeddings": "#1f77b4", "graph_cluster": "#ff7f0e",
              "graph_availability": "#2ca02c"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#202020}.title{font-size:15px;font-weight:bold}'
             '.axis{stroke:#777;stroke-width:1}</style>',
             '<text x="20" y="26" style="font-size:19px;font-weight:bold">Validation-only grouped SHAP mean absolute attribution</text>']
    for index, result in enumerate(results):
        x0 = margin_x + (index % 4) * panel_width
        y0 = margin_y + (index // 4) * panel_height
        if index >= 4:
            x0 = margin_x + (index % 4) * panel_width
            y0 = margin_y + (index // 4) * panel_height
        max_value = max(float(result["group_mean_abs_sum"].get(group, 0.0))
                        for group in groups) or 1.0
        parts.append(f'<text x="{x0}" y="{y0 + 14}" class="title">'
                     f'{svg_escape(result["family"])} / {svg_escape(result["surface"])} / '
                     f'{svg_escape(result["setting"])} / seed {result["seed"]}</text>')
        baseline = y0 + 184
        parts.append(f'<line x1="{x0 + 8}" y1="{baseline}" x2="{x0 + 265}" y2="{baseline}" class="axis"/>')
        for group_index, group in enumerate(groups):
            value = float(result["group_mean_abs_sum"].get(group, 0.0))
            bar_height = 135 * value / max_value
            bx = x0 + 22 + group_index * 78
            by = baseline - bar_height
            parts.append(f'<rect x="{bx}" y="{by:.2f}" width="42" height="{bar_height:.2f}" '
                         f'fill="{colors[group]}"/>')
            parts.append(f'<text x="{bx + 21}" y="{baseline + 14}" text-anchor="middle" '
                         f'style="font-size:10px">{group.replace("graph_", "")}</text>')
            parts.append(f'<text x="{bx + 21}" y="{by - 4:.2f}" text-anchor="middle" '
                         f'style="font-size:10px">{value:.3f}</text>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def tikz_forest(path, records, title):
    values = []
    for row in records:
        values.extend(float(row[key]) for key in ("estimate", "ci95_lower", "ci95_upper")
                      if row.get(key) not in (None, "", "None"))
    low, high = min(values), max(values)
    span = high - low or 1.0
    low -= span * 0.08
    high += span * 0.08
    n = len(records)
    paper_height = max(6.0, 2.0 + 0.27 * n)

    def tx(value):
        return 6.5 + (float(value) - low) / (high - low) * 13.0

    def tex(value):
        return (str(value).replace("\\", "\\textbackslash{}").replace("&", "\\&")
                .replace("%", "\\%").replace("_", "\\_").replace("#", "\\#"))

    colors = {"turf": "blue!70!black", "dirt": "red!70!black"}
    lines = ["\\documentclass{article}",
             f"\\usepackage[paperwidth=21.5cm,paperheight={paper_height:.2f}cm,margin=0pt]{{geometry}}",
             "\\usepackage{tikz}", "\\pagestyle{empty}", "\\begin{document}",
             "\\begin{tikzpicture}[x=1cm,y=1cm]",
             f"\\node[anchor=west,font=\\bfseries\\large] at (0.4,{paper_height - 0.5:.3f}) {{{tex(title)}}};"]
    for index, row in enumerate(records):
        y = paper_height - 1.2 - index * 0.27
        color = colors.get(str(row.get("surface", "")), "black")
        label = f'{row.get("family", "")} / {row.get("surface", "")} / {row.get("contrast", "")}'
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at (6.3,{y:.3f}) {{{tex(label)}}};")
        if row.get("ci95_lower") not in (None, "", "None"):
            lines.append(f"\\draw[{color},line width=1.2pt] ({tx(row['ci95_lower']):.4f},{y:.3f}) -- ({tx(row['ci95_upper']):.4f},{y:.3f});")
        lines.append(f"\\fill[{color}] ({tx(row['estimate']):.4f},{y:.3f}) circle (1.4pt);")
    if low <= 0 <= high:
        y_top = paper_height - 0.95
        lines.append(f"\\draw[gray,dashed] ({tx(0):.4f},0.55) -- ({tx(0):.4f},{y_top:.3f});")
    lines.append("\\draw[gray] (6.5,0.55) -- (19.5,0.55);")
    for tick in range(6):
        value = low + (high - low) * tick / 5
        xpos = tx(value)
        lines.append(f"\\draw[gray] ({xpos:.4f},0.55) -- ({xpos:.4f},0.68);")
        lines.append(f"\\node[font=\\tiny] at ({xpos:.4f},0.28) {{{value:.3f}}};")
    lines.extend(["\\end{tikzpicture}", "\\end{document}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def tikz_shap(path, results):
    groups = ("graph_embeddings", "graph_cluster", "graph_availability")
    colors = ("blue!70!black", "orange!90!black", "green!60!black")
    lines = ["\\documentclass{article}",
             "\\usepackage[paperwidth=21.5cm,paperheight=12.5cm,margin=0pt]{geometry}",
             "\\usepackage{tikz}", "\\pagestyle{empty}", "\\begin{document}",
             "\\begin{tikzpicture}[x=1cm,y=1cm]",
             "\\node[anchor=west,font=\\bfseries\\large] at (0.6,11.8) "
             "{Validation-only grouped SHAP mean absolute attribution};"]
    for index, result in enumerate(results):
        panel_x = 0.7 + (index % 2) * 10.4
        baseline = 6.8 if index < 2 else 1.9
        values = [float(result["group_mean_abs_sum"].get(group, 0.0))
                  for group in groups]
        maximum = max(values) or 1.0
        label = (f'{result["family"]} / {result["surface"]} / '
                 f'{result["setting"]} / seed {result["seed"]}')
        lines.append(f"\\node[anchor=west,font=\\small] at ({panel_x:.2f},{baseline + 3.9:.2f}) "
                     f"{{{label}}};")
        lines.append(f"\\draw[gray] ({panel_x:.2f},{baseline:.2f}) -- "
                     f"({panel_x + 9.3:.2f},{baseline:.2f});")
        for group_index, (group, value, color) in enumerate(zip(groups, values, colors)):
            center = panel_x + 1.2 + group_index * 2.8
            height = 3.0 * value / maximum
            lines.append(f"\\fill[{color}] ({center - 0.38:.2f},{baseline:.2f}) "
                         f"rectangle ({center + 0.38:.2f},{baseline + height:.3f});")
            lines.append(f"\\node[font=\\scriptsize] at ({center:.2f},"
                         f"{baseline + height + 0.18:.3f}) {{{value:.3f}}};")
            label = ("embeddings" if group == "graph_embeddings"
                     else "cluster" if group == "graph_cluster" else "availability")
            lines.append(f"\\node[font=\\tiny,align=center] at ({center:.2f},"
                         f"{baseline - 0.28:.2f}) {{{label}}};")
    lines.extend(["\\end{tikzpicture}", "\\end{document}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_figure(tex_path, build_dir, figure_stem):
    pdflatex = shutil.which("pdflatex")
    pdftocairo = shutil.which("pdftocairo")
    if not pdflatex or not pdftocairo:
        raise RuntimeError("pdflatex and pdftocairo are required for PDF/PNG figure export")
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
               "-output-directory", build_dir.resolve().as_posix(),
               tex_path.resolve().as_posix()]
    result = subprocess.run(command, cwd=build_dir, capture_output=True,
                            text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:] + result.stderr[-4000:])
    pdf = build_dir / f"{figure_stem}.pdf"
    destination = tex_path.parent.parent / "figures" / figure_stem
    shutil.copy2(pdf, destination.with_suffix(".pdf"))
    subprocess.run([pdftocairo, "-png", "-singlefile", "-r", "160",
                    pdf.resolve().as_posix(), destination.resolve().as_posix()],
                   capture_output=True, text=True, check=True)


def export_forest(output, records, figure_stem, title, width=1400):
    forest_svg(output / "figures" / f"{figure_stem}.svg", records, title,
               "label", width=width)
    tex_path = output / "figure_sources" / f"{figure_stem}.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tikz_forest(tex_path, records, title)
    compile_figure(tex_path, output / "figure_build", figure_stem)


def artifact_inventory(root):
    root = Path(root)
    return [{"path": path.relative_to(root).as_posix(),
             "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "release_metadata.json"]


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite release: {output}")
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    supplement = read_json(Path(args.supplement) / "final_reporting_supplement_report.json")
    base = read_json(args.base_evaluation_report)
    shap = read_json(args.shap_report)
    seed_rows = read_csv(Path(args.supplement) / "model_seed_metrics_extended.csv")
    summary_rows = read_csv(Path(args.supplement) / "model_summaries_extended.csv")
    primary = flatten_contrasts(base["primary_contrasts"])
    temporal = flatten_contrasts(base["temporal_sensitivity"])
    surface = read_csv(Path(args.supplement) / "surface_contrasts.csv")
    subgroup = read_csv(Path(args.supplement) / "subgroup_metrics_with_ci.csv")
    b0 = read_csv(Path(args.supplement) / "b0_metrics.csv")

    write_csv(output / "tables/model_seed_metrics_extended.csv", seed_rows)
    write_csv(output / "tables/model_summaries_extended.csv", summary_rows)
    write_csv(output / "tables/primary_contrasts.csv", primary)
    write_csv(output / "tables/temporal_contrasts.csv", temporal)
    write_csv(output / "tables/surface_contrasts.csv", surface)
    write_csv(output / "tables/subgroup_metrics_with_ci.csv", subgroup)
    write_csv(output / "tables/b0_metrics.csv", b0)
    write_csv(output / "tables/ndcg_summary.csv", ndcg_rows(summary_rows, supplement))
    shap_rows = []
    for result in shap["results"]:
        for group, value in sorted(result["group_mean_abs_sum"].items()):
            shap_rows.append({"family": result["family"], "surface": result["surface"],
                              "setting": result["setting"], "seed": result["seed"],
                              "group": group, "mean_abs_attribution": value,
                              "validation_entry_count": result["validation_entry_count"],
                              "validation_race_count": result["validation_race_count"]})
    write_csv(output / "tables/grouped_shap.csv", shap_rows)
    write_csv(output / "tables/prediction_inventory.csv",
              prediction_inventory(args.predictions_root))

    report_sources = {
        "spec": args.spec, "manifest": args.manifest,
        "scientific_audit": args.scientific_audit,
        "inference_report": args.inference_report,
        "base_evaluation_report": args.base_evaluation_report,
        "supplement_report": Path(args.supplement) / "final_reporting_supplement_report.json",
        "shap_report": args.shap_report, "launch_lock": args.launch_lock,
        "run_record": args.run_record, "gap_audit": args.gap_audit,
        "graph_stability": args.graph_stability,
        "metric_parity": args.metric_parity,
        "manuscript_outline": args.manuscript_outline,
        "reference_audit": args.reference_audit,
        "evaluate_locked": Path("kaggle/evaluate_locked.py"),
        "evaluate_locked_test_results": Path("kaggle/evaluate_locked_test_results.py"),
        "complete_locked_reporting": Path("kaggle/complete_locked_reporting.py"),
        "build_results_release": Path("kaggle/build_results_release.py"),
    }
    sources = {}
    for name, source in report_sources.items():
        destination = output / "reports" / Path(source).name
        sources[name] = copy_source(source, destination)

    forest_records = []
    for row in primary:
        if row["metric"] != "entry_average_precision":
            continue
        item = dict(row)
        item["label"] = f'{row["family"]} / {row["surface"]} / {row["contrast"]}'
        forest_records.append(item)
    forest_records.sort(key=lambda row: (FAMILIES.index(row["family"]),
                                         SURFACES.index(row["surface"]),
                                         CONTRASTS.index(row["contrast"])))
    export_forest(output, forest_records, "primary_ap_contrasts",
                  "Primary AP contrasts: estimate and 95% whole-race interval")
    hit3_records = []
    for row in primary:
        if row["metric"] != "race_mean_hit_at_3":
            continue
        item = dict(row)
        item["label"] = f'{row["family"]} / {row["surface"]} / {row["contrast"]}'
        hit3_records.append(item)
    hit3_records.sort(key=lambda row: (FAMILIES.index(row["family"]),
                                       SURFACES.index(row["surface"]),
                                       CONTRASTS.index(row["contrast"])))
    export_forest(output, hit3_records, "primary_hit3_contrasts",
                  "Primary Mean Hit@3 contrasts: estimate and 95% whole-race interval")
    history_records = []
    for row in subgroup:
        if row["setting"] != "S6" or row["subgroup_type"] != "history_exact":
            continue
        ci = parse_ci(row["entry_average_precision_ci95"])
        history_records.append({"family": row["family"], "surface": row["surface"],
                                "contrast": row["subgroup"],
                                "label": f'{row["family"]} / {row["surface"]} / {row["subgroup"]}',
                                "estimate": float(row["entry_average_precision_mean"]),
                                "ci95_lower": ci[0] if ci else None,
                                "ci95_upper": ci[1] if ci else None})
    history_records.sort(key=lambda row: (FAMILIES.index(row["family"]),
                                          SURFACES.index(row["surface"]),
                                          ("0", "1", "2", "3+").index(row["label"].split("/")[-1].strip())))
    export_forest(output, history_records, "s6_history_ap_subgroups",
                  "S6 history subgroup AP: point and race-bootstrap interval")
    coverage_records = []
    for row in subgroup:
        if (row["setting"] not in ("S6", "S7")
                or row["subgroup_type"] != "coverage"
                or row["subgroup"] == "missing"):
            continue
        ci = parse_ci(row["entry_average_precision_ci95"])
        if ci is None or row["entry_average_precision_mean"] in ("", "None"):
            continue
        label = (f'{row["family"]} / {row["surface"]} / {row["setting"]} / '
                 f'{row["subgroup"]} (n={float(row["entry_count_mean"]):.0f})')
        coverage_records.append({
            "family": row["family"], "surface": row["surface"],
            "contrast": f'{row["setting"]}-{row["subgroup"]}',
            "label": label,
            "estimate": float(row["entry_average_precision_mean"]),
            "ci95_lower": ci[0], "ci95_upper": ci[1]})
    coverage_records.sort(key=lambda row: (
        FAMILIES.index(row["family"]), SURFACES.index(row["surface"]),
        ("S6", "S7").index(row["contrast"].split("-")[0]),
        ("direct", "ancestor_fallback").index(row["contrast"].split("-")[1])))
    export_forest(output, coverage_records, "graph_coverage_ap_subgroups",
                  "S6/S7 graph-coverage subgroup AP: estimate and 95% interval",
                  width=1800)
    shap_svg(output / "figures/grouped_shap_validation.svg", shap["results"])
    shap_tex_path = output / "figure_sources/grouped_shap_validation.tex"
    shap_tex_path.parent.mkdir(parents=True, exist_ok=True)
    tikz_shap(shap_tex_path, shap["results"])
    compile_figure(shap_tex_path, output / "figure_build",
                   "grouped_shap_validation")

    metadata = {
        "status": "complete",
        "release_date": "2026-09-16",
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_status": read_json(args.manifest)["status"],
        "test_gate": {"locked_inference_complete": True,
                       "test_performance_opened": True,
                       "test_evaluation_authorized": True},
        "counts": {"model_seed_rows": len(read_csv(Path(args.supplement) / "model_seed_metrics_extended.csv")),
                    "model_summaries": len(summary_rows), "primary_contrasts": len(primary),
                    "temporal_contrasts": len(temporal), "surface_contrasts": len(surface),
                    "subgroup_rows": len(subgroup), "b0_rows": len(b0),
                    "shap_group_rows": len(shap_rows)},
        "cohorts": supplement["cohorts"],
        "bootstrap": supplement["bootstrap"],
        "source_artifacts": sources,
        "source_tables": {
            "model_seed_metrics_extended": copy_source(
                Path(args.supplement) / "model_seed_metrics_extended.csv",
                output / "reports/source_model_seed_metrics_extended.csv"),
            "model_summaries_extended": copy_source(
                Path(args.supplement) / "model_summaries_extended.csv",
                output / "reports/source_model_summaries_extended.csv"),
            "subgroup_metrics_with_ci": copy_source(
                Path(args.supplement) / "subgroup_metrics_with_ci.csv",
                output / "reports/source_subgroup_metrics_with_ci.csv"),
            "surface_contrasts": copy_source(
                Path(args.supplement) / "surface_contrasts.csv",
                output / "reports/source_surface_contrasts.csv"),
            "b0_metrics": copy_source(
                Path(args.supplement) / "b0_metrics.csv",
                output / "reports/source_b0_metrics.csv"),
        },
        "reproduction": {
            "supplement": "python kaggle/complete_locked_reporting.py ... --bootstrap-replicates 2000",
            "release": "python kaggle/build_results_release.py ...",
            "note": "Use the exact paths and arguments recorded in the source run logs; no training is performed by either command.",
        },
        "private_data_policy": "Prediction inventory records hashes and paths; raw databases and credentials are not copied into this release.",
    }
    (output / "README.md").write_text(
        "# Locked results release — 2026-09-16\n\n"
        "This directory contains the verified post-test reporting bundle. The immutable validation selection manifest is copied under `reports/` and remains byte-identical to the source hash recorded in `release_metadata.json`.\n\n"
        "The supplement adds B0, NDCG summaries and denominators, subgroup race-bootstrap intervals, and turf-minus-dirt contrast differences from the locked prediction files. It performs no model fitting, retuning, seed selection, threshold changes, or manifest edits. SHAP artifacts are validation-only.\n\n"
        "Figures include AP and Mean Hit@3 contrast forests plus history, graph-coverage, and validation-SHAP summaries. The JSON artifact index records a SHA-256 and byte count for every file except the index itself.\n\n"
        "See `release_metadata.json` for source hashes, cohort denominators, counts, and reproduction commands. `tables/prediction_inventory.csv` inventories the 180 private prediction files without copying them.\n",
        encoding="utf-8")
    metadata["artifact_inventory"] = artifact_inventory(output)
    metadata["artifact_count"] = len(metadata["artifact_inventory"])
    write_json(output / "release_metadata.json", metadata)
    write_json(args.index_path, metadata)
    print(json.dumps({"status": "complete", "output": str(output),
                      "primary_ap_rows": len(forest_records),
                      "primary_hit3_rows": len(hit3_records),
                      "graph_coverage_rows": len(coverage_records),
                      "artifact_count": metadata["artifact_count"],
                      "shap_results": len(shap["results"])}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--base-evaluation-report", type=Path, required=True)
    parser.add_argument("--shap-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--scientific-audit", type=Path, required=True)
    parser.add_argument("--launch-lock", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--gap-audit", type=Path, required=True)
    parser.add_argument("--graph-stability", type=Path, required=True)
    parser.add_argument("--metric-parity", type=Path, required=True)
    parser.add_argument("--manuscript-outline", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
