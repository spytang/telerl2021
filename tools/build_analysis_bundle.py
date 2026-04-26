"""Build an analysis bundle for the latest experiment artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
BUNDLE_DIR = REPO_ROOT / "analysis_bundle"

REQUIRED_ROOT_FILES = [
    "README.md",
    "train.py",
    "evaluate_and_plot.py",
    "robustness_test.py",
]

REQUIRED_RUN_FILES = {
    "config.json": ["config.json"],
    "summary.json": ["summary.json"],
    "evaluation_metrics.csv": ["evaluation/evaluation_metrics.csv"],
    "robustness_metrics.csv": ["robustness/robustness_metrics.csv"],
    "ai_report.md": ["ai_report.md"],
}

BASELINE_PRINCIPLE = (
    "Default baseline must stay unchanged and reproducible: "
    "Poisson arrival + default reward. In research mode, new traffic_model/reward_profile "
    "variants are allowed only when explicitly labeled, comparable against baseline, "
    "reversible, and never presented as baseline."
)


def _latest_run_with(rel_path: str) -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None

    candidates = [
        p
        for p in RUNS_DIR.iterdir()
        if p.is_dir() and (p / rel_path).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / rel_path).stat().st_mtime)


def _collect_sources() -> Tuple[Dict[str, Path], List[str], Dict[str, str]]:
    found: Dict[str, Path] = {}
    missing: List[str] = []
    source_meta: Dict[str, str] = {}

    for name in REQUIRED_ROOT_FILES:
        src = REPO_ROOT / name
        if src.exists():
            found[name] = src
            source_meta[name] = str(src.relative_to(REPO_ROOT))
        else:
            missing.append(name)

    for bundle_name, search_paths in REQUIRED_RUN_FILES.items():
        chosen: Optional[Tuple[Path, str]] = None
        for rel_path in search_paths:
            run_dir = _latest_run_with(rel_path)
            if run_dir is None:
                continue
            candidate = run_dir / rel_path
            if chosen is None or candidate.stat().st_mtime > chosen[0].stat().st_mtime:
                chosen = (candidate, rel_path)

        if chosen is None:
            missing.append(bundle_name)
            continue

        src_file, rel_used = chosen
        found[bundle_name] = src_file
        source_meta[bundle_name] = str(src_file.relative_to(REPO_ROOT))
        source_meta[f"{bundle_name}__run_dir"] = src_file.parents[len(Path(rel_used).parts) - 1].name

    return found, missing, source_meta


def _extract_config_preview(config_path: Optional[Path]) -> Dict[str, object]:
    if config_path is None or not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"note": "config.json exists but is not valid JSON."}

    preview_keys = [
        "run_name",
        "mode",
        "traffic_model",
        "reward_profile",
        "seed",
        "episodes",
        "env",
        "training",
    ]
    preview: Dict[str, object] = {}
    for key in preview_keys:
        if key in data:
            preview[key] = data[key]
    return preview


def _write_next_prompt(
    bundle_dir: Path,
    found: Dict[str, Path],
    missing: List[str],
    source_meta: Dict[str, str],
) -> None:
    config_preview = _extract_config_preview(found.get("config.json"))
    available_files = sorted(found.keys())

    lines = [
        "# Next Codex Prompt",
        "",
        "请基于 `analysis_bundle/` 中的文件分析当前实验状态，并给出下一轮建议。",
        "",
        "## Baseline Guardrail (必须遵守)",
        BASELINE_PRINCIPLE,
        "",
        "## 本轮配置（来自 config.json）",
        "```json",
        json.dumps(config_preview, indent=2, ensure_ascii=False) if config_preview else "{}",
        "```",
        "",
        "## 本轮已收集结果文件",
    ]

    lines.extend([f"- {name} (source: {source_meta.get(name, 'unknown')})" for name in available_files])

    lines.extend([
        "",
        "## 缺失文件",
    ])
    if missing:
        lines.extend([f"- {name}" for name in sorted(missing)])
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 请 Codex 执行",
            "- 分析当前模型问题与主要瓶颈（训练稳定性、URLLC违约率、eMBB outage/方差权衡）。",
            "- 给出下一轮实验建议（先小步、可回退、可对照 baseline）。",
            "- 给出下一轮建议运行命令（train / eval / robustness）。",
            "- 若使用 research mode，请显式标注并保持与 baseline 的公平对照。",
        ]
    )

    (bundle_dir / "next_codex_prompt.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    found, missing, source_meta = _collect_sources()

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for bundle_name, src in found.items():
        shutil.copy2(src, BUNDLE_DIR / bundle_name)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(BUNDLE_DIR.relative_to(REPO_ROOT)),
        "included": sorted(found.keys()),
        "missing": sorted(missing),
        "source_paths": source_meta,
        "baseline_guardrail": BASELINE_PRINCIPLE,
        "reserved_interfaces": {
            "mode": ["baseline", "research"],
            "traffic_model": ["poisson"],
            "reward_profile": ["default"],
        },
    }
    (BUNDLE_DIR / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "# Analysis Bundle Report",
        "",
        f"- Generated at (UTC): {manifest['generated_at_utc']}",
        f"- Included files: {', '.join(manifest['included']) if manifest['included'] else 'none'}",
        f"- Missing files: {', '.join(manifest['missing']) if manifest['missing'] else 'none'}",
        "",
        "## Baseline Guardrail",
        BASELINE_PRINCIPLE,
    ]
    (BUNDLE_DIR / "bundle_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    _write_next_prompt(BUNDLE_DIR, found, missing, source_meta)

    print(f"[OK] analysis bundle generated at: {BUNDLE_DIR}")
    if missing:
        print("[WARN] missing files:")
        for item in sorted(missing):
            print(f"  - {item}")


if __name__ == "__main__":
    main()
