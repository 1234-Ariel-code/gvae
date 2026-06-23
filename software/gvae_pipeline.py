#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gVAE software pipeline runner.

This script provides a user-facing, configuration-driven wrapper around the
manuscript-facing gVAE implementation. It does not duplicate the scientific
code; instead, it calls the main scripts in the gvae/ package using a YAML
configuration file.

Example
-------
python software/gvae_pipeline.py \
  --config software/config_template.yaml \
  --steps smoke train xai

python software/gvae_pipeline.py \
  --config software/config_template.yaml \
  --steps all \
  --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to use the gVAE software pipeline. "
        "Install it with: pip install pyyaml"
    ) from exc


STEP_ORDER = ["smoke", "train", "xai", "predict", "enrich", "gwas_xai"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("The configuration file must define a YAML mapping/dictionary.")

    return config


def repo_root_from_config(config: dict[str, Any]) -> Path:
    """Resolve repository root from config, defaulting to the parent of software/."""
    configured = config.get("project", {}).get("repo_root")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from a dictionary."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def ensure_dir(path: str | Path | None) -> None:
    """Create a directory if a path is provided."""
    if path:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def add_option(cmd: list[str], flag: str, value: Any) -> None:
    """Append a command-line option when the value is not empty."""
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        return
    cmd.extend([flag, str(value)])


def add_flag(cmd: list[str], flag: str, value: Any) -> None:
    """Append a boolean flag when the value is truthy."""
    if bool(value):
        cmd.append(flag)


def add_extra_args(cmd: list[str], extra_args: Any) -> None:
    """Append user-defined extra arguments from a list or mapping."""
    if not extra_args:
        return

    if isinstance(extra_args, list):
        cmd.extend(str(x) for x in extra_args)
        return

    if isinstance(extra_args, dict):
        for key, value in extra_args.items():
            flag = str(key)
            if not flag.startswith("--"):
                flag = f"--{flag}"
            if isinstance(value, bool):
                add_flag(cmd, flag, value)
            elif isinstance(value, (list, tuple)):
                cmd.append(flag)
                cmd.extend(str(x) for x in value)
            else:
                add_option(cmd, flag, value)
        return

    raise TypeError("extra_args must be either a list or a dictionary.")


def print_command(cmd: list[str]) -> None:
    """Print a shell-safe representation of a command."""
    print("\n$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> None:
    """Run a command with consistent logging and error handling."""
    print_command(cmd)
    if dry_run:
        return

    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError:
        if continue_on_error:
            print("[WARN] Command failed but continuing because --continue-on-error was set.")
            return
        raise


def common(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a common parameter from config['analysis']."""
    return get_nested(config, "analysis", key, default=default)


def step_config(config: dict[str, Any], step: str) -> dict[str, Any]:
    """Return the configuration block for a pipeline step."""
    block = get_nested(config, "steps", step, default={})
    return block if isinstance(block, dict) else {}


def build_smoke_commands(config: dict[str, Any], python_exe: str) -> list[list[str]]:
    """Build import and CLI smoke-test commands."""
    commands = [
        [
            python_exe,
            "-c",
            (
                "from gvae.model import GVAE, BaselineVAE, BetaVAE; "
                "from gvae.metrics import evaluate_r_square; "
                "print('gVAE imports OK')"
            ),
        ],
        [python_exe, "-m", "gvae.gvae", "--help"],
        [python_exe, "-m", "gvae.snp_prioritization", "--help"],
    ]

    extra_commands = step_config(config, "smoke").get("extra_commands", [])
    for item in extra_commands:
        if isinstance(item, list):
            commands.append([str(x) for x in item])
        elif isinstance(item, str):
            commands.append(item.split())
        else:
            raise TypeError("smoke.extra_commands entries must be strings or lists.")
    return commands


def build_train_command(config: dict[str, Any], python_exe: str) -> list[str]:
    """Build the gVAE model-training command."""
    block = step_config(config, "train")
    output_dir = block.get("output_dir", common(config, "model_output_dir", "outputs/model"))
    ensure_dir(output_dir)

    cmd = [python_exe, "-m", "gvae.gvae"]
    add_option(cmd, "--disease", common(config, "disease"))
    add_option(cmd, "--bed_prefix", block.get("bed_prefix", common(config, "bed_prefix")))
    add_option(cmd, "--latent_dim", block.get("latent_dim", common(config, "latent_dim")))
    add_option(cmd, "--num_sample", block.get("num_sample", common(config, "num_samples")))
    add_option(cmd, "--num_layer", block.get("num_layer", common(config, "num_layers")))
    add_option(cmd, "--epochs", block.get("epochs", common(config, "epochs", 20)))
    add_option(cmd, "--batch_size", block.get("batch_size", common(config, "batch_size", 32)))
    add_option(cmd, "--feature_mode", block.get("feature_mode", "gwas_top"))
    add_option(cmd, "--downsample_d", block.get("downsample_d", common(config, "downsample_d", 50000)))
    add_option(cmd, "--gwas_assoc_path", block.get("gwas_assoc_path", common(config, "gwas_assoc_path")))
    add_option(cmd, "--tped_file", block.get("tped_file", common(config, "tped_file")))
    add_option(cmd, "--bim_file", block.get("bim_file", common(config, "bim_file")))
    add_option(cmd, "--beta", block.get("beta"))
    add_option(cmd, "--beta_list", block.get("beta_list"))
    add_option(cmd, "--output_dir", output_dir)
    add_extra_args(cmd, block.get("extra_args"))
    return cmd


def build_xai_command(config: dict[str, Any], python_exe: str) -> list[str]:
    """Build the SHAP-based SNP-prioritization command."""
    block = step_config(config, "xai")
    output_dir = block.get("output_dir", common(config, "xai_output_dir", "outputs/xai"))
    ensure_dir(output_dir)

    cmd = [python_exe, "-m", "gvae.snp_prioritization"]
    add_option(cmd, "--disease", common(config, "disease"))
    add_option(cmd, "--base_path", block.get("base_path", common(config, "base_path")))
    add_option(cmd, "--latent_dim", block.get("latent_dim", common(config, "latent_dim")))
    add_option(cmd, "--num_samples", block.get("num_samples", common(config, "num_samples")))
    add_option(cmd, "--num_layers", block.get("num_layers", common(config, "num_layers")))
    add_option(cmd, "--shap_top_k", block.get("shap_top_k", common(config, "shap_top_k", 10)))
    add_option(cmd, "--tped_file", block.get("tped_file", common(config, "tped_file")))
    add_option(cmd, "--bim_file", block.get("bim_file", common(config, "bim_file")))
    add_option(cmd, "--assoc_path", block.get("assoc_path", common(config, "gwas_assoc_path")))
    add_option(cmd, "--output_dir", output_dir)
    add_option(cmd, "--epochs", block.get("epochs", common(config, "epochs", 20)))
    add_option(cmd, "--batch_size", block.get("batch_size", common(config, "batch_size", 32)))
    add_extra_args(cmd, block.get("extra_args"))
    return cmd


def build_predict_command(config: dict[str, Any], python_exe: str, repo_root: Path) -> list[str]:
    """Build the latent-space prediction command."""
    block = step_config(config, "predict")
    output_dir = block.get("out_root", common(config, "prediction_output_dir", "outputs/prediction"))
    ensure_dir(output_dir)

    script = repo_root / "gvae" / "latent_classification.py"
    cmd = [python_exe, str(script)]
    add_option(cmd, "--disease", common(config, "disease"))
    add_option(cmd, "--base_path", block.get("base_path", common(config, "base_path")))
    add_option(cmd, "--model_type", block.get("model_type", "gvae"))
    add_option(cmd, "--latent_dim", block.get("latent_dim", common(config, "latent_dim")))
    add_option(cmd, "--num_samples", block.get("num_samples", common(config, "num_samples")))
    add_option(cmd, "--num_layers", block.get("num_layers", common(config, "num_layers")))
    add_option(cmd, "--feature_mode", block.get("feature_mode", "gwas_top"))
    add_option(cmd, "--downsample_d", block.get("downsample_d", common(config, "downsample_d", 50000)))
    add_option(cmd, "--assoc_path", block.get("assoc_path", common(config, "gwas_assoc_path")))
    add_option(cmd, "--tped_file", block.get("tped_file", common(config, "tped_file")))
    add_option(cmd, "--train_vae_epochs", block.get("train_vae_epochs", common(config, "epochs", 50)))
    add_option(cmd, "--vae_batch_size", block.get("vae_batch_size", common(config, "batch_size", 256)))
    add_option(cmd, "--batch_size", block.get("batch_size", common(config, "batch_size", 256)))
    add_option(cmd, "--epochs", block.get("epochs", 120))
    add_flag(cmd, "--cache_latents", block.get("cache_latents", True))
    add_flag(cmd, "--make_plots", block.get("make_plots", True))
    add_option(cmd, "--out_root", output_dir)
    add_extra_args(cmd, block.get("extra_args"))
    return cmd


def build_enrich_command(config: dict[str, Any], python_exe: str, repo_root: Path) -> list[str]:
    """Build the SNP-to-gene and pathway enrichment command."""
    block = step_config(config, "enrich")
    output_dir = block.get("out_root", common(config, "enrichment_output_dir", "outputs/enrichment"))
    ensure_dir(output_dir)

    script = repo_root / "gvae" / "gene-pathway_enrichment.py"
    cmd = [python_exe, str(script)]
    add_option(cmd, "--disease", common(config, "disease"))
    add_option(cmd, "--base_dir", block.get("base_dir", common(config, "xai_output_dir", "outputs/xai")))
    add_option(cmd, "--s2g_path", block.get("s2g_path", common(config, "s2g_path")))
    add_option(cmd, "--bim_file", block.get("bim_file", common(config, "bim_file")))
    add_flag(cmd, "--run_gene_analysis", block.get("run_gene_analysis", True))
    add_option(cmd, "--disgenet_mode", block.get("disgenet_mode", "tsv"))
    add_option(cmd, "--disgenet_tsv", block.get("disgenet_tsv", common(config, "disgenet_tsv")))
    add_option(cmd, "--disgenet_disease_name", block.get("disgenet_disease_name", common(config, "disgenet_disease_name")))
    add_option(cmd, "--out_root", output_dir)
    add_extra_args(cmd, block.get("extra_args"))
    return cmd


def build_gwas_xai_command(config: dict[str, Any], rscript_exe: str, repo_root: Path) -> list[str]:
    """Build the GWAS-XAI R command."""
    block = step_config(config, "gwas_xai")
    script = block.get("script", str(repo_root / "gvae" / "gwas-xai.R"))
    cmd = [rscript_exe, script]
    add_extra_args(cmd, block.get("extra_args"))
    return cmd


def resolve_steps(steps: list[str]) -> list[str]:
    """Expand 'all' into the full ordered pipeline."""
    if "all" in steps:
        return STEP_ORDER.copy()

    resolved: list[str] = []
    for step in steps:
        if step not in STEP_ORDER:
            raise ValueError(f"Unknown step: {step}")
        if step not in resolved:
            resolved.append(step)
    return resolved


def run_pipeline(args: argparse.Namespace) -> None:
    """Run selected pipeline steps."""
    config = load_config(args.config)
    repo_root = repo_root_from_config(config)

    python_exe = args.python or get_nested(config, "project", "python", default=sys.executable)
    rscript_exe = args.rscript or get_nested(config, "project", "rscript", default="Rscript")
    steps = resolve_steps(args.steps)

    print(f"[INFO] Repository root: {repo_root}")
    print(f"[INFO] Configuration: {Path(args.config).resolve()}")
    print(f"[INFO] Steps: {', '.join(steps)}")
    print(f"[INFO] Dry run: {args.dry_run}")

    for step in steps:
        print(f"\n[STEP] {step}")
        if step == "smoke":
            for cmd in build_smoke_commands(config, python_exe):
                run_command(cmd, cwd=repo_root, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        elif step == "train":
            run_command(build_train_command(config, python_exe), cwd=repo_root, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        elif step == "xai":
            run_command(build_xai_command(config, python_exe), cwd=repo_root, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        elif step == "predict":
            run_command(build_predict_command(config, python_exe, repo_root), cwd=repo_root, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        elif step == "enrich":
            run_command(build_enrich_command(config, python_exe, repo_root), cwd=repo_root, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        elif step == "gwas_xai":
            run_command(build_gwas_xai_command(config, rscript_exe, repo_root), cwd=repo_root, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled step: {step}")

    print("\n[DONE] gVAE software pipeline completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configuration-driven gVAE software pipeline runner."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        required=True,
        choices=STEP_ORDER + ["all"],
        help="Pipeline steps to run. Use 'all' for the full workflow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next step if one command fails.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use. Defaults to the current interpreter or config value.",
    )
    parser.add_argument(
        "--rscript",
        default=None,
        help="Rscript executable to use for GWAS-XAI comparison. Defaults to Rscript.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
