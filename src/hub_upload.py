"""
Standalone tool: upload trained models to Hugging Face Hub, skipping any that
have already been uploaded.

Does NOT retrain or re-evaluate anything -- it only reads what src/training.py
already wrote to Models/<domain>/<dataset_name>/<model_name>/ and pushes it.
Reuses training.push_to_hub() so the uploaded artifact set (model file,
feature_schema.json, test_metrics.json, model card) and the "record the commit
SHA back into model_info.json" behavior stay identical to a push done during
training itself.

"Already uploaded" = model_info.json's "hub_commit_sha" field is set (not
None/missing). That field is only ever written by a successful push (either
during training with push_to_hub=True, or by this tool), so it's a reliable
skip marker -- nothing else in the pipeline touches it.

Usage:
    from src.hub_upload import upload_pending_models

    summary = upload_pending_models(
        models_root="../Models",
        hub_namespace="yourname",
        dry_run=True,   # see what WOULD happen first, uploads nothing
    )
    # ... check summary, then:
    summary = upload_pending_models(models_root="../Models", hub_namespace="yourname")
"""

import os
import json

from src.utils import Logger
from src.training import push_to_hub, _HAS_HFHUB


def _default_repo_id(hub_namespace, dataset_name, model_name):
    return f"{hub_namespace}/xai-{dataset_name.replace('_', '-')}-{model_name}"


def scan_models(models_root="Models", summary_subdir="summary"):
    """Walks Models/<domain>/<dataset_name>/<model_name>/ and returns one dict
    per run that has a metadata/model_info.json (i.e. training actually
    completed for it). Runs without a model_info.json are skipped entirely
    (nothing to upload) rather than erroring, since a partial/failed training
    run shouldn't block uploading everything else.
    """
    runs = []
    if not os.path.isdir(models_root):
        return runs

    for domain in sorted(os.listdir(models_root)):
        domain_path = os.path.join(models_root, domain)
        if not os.path.isdir(domain_path) or domain == summary_subdir:
            continue
        for dataset_name in sorted(os.listdir(domain_path)):
            dataset_path = os.path.join(domain_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            for model_name in sorted(os.listdir(dataset_path)):
                run_dir = os.path.join(dataset_path, model_name)
                model_dir = os.path.join(run_dir, "model")
                metadata_dir = os.path.join(run_dir, "metadata")
                info_path = os.path.join(metadata_dir, "model_info.json")
                if not os.path.exists(info_path):
                    continue

                with open(info_path) as f:
                    model_info = json.load(f)

                runs.append({
                    "domain": domain,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "model_dir": model_dir,
                    "metadata_dir": metadata_dir,
                    "model_info_path": info_path,
                    "model_info": model_info,
                    "already_uploaded": bool(model_info.get("hub_commit_sha")),
                })
    return runs


def upload_pending_models(models_root="Models", hub_namespace=None, repo_id_fn=None,
                            hub_private=True, dry_run=False, force=False):
    """Scans models_root for completed training runs and uploads to Hugging
    Face Hub any that aren't uploaded yet, skipping the rest.

    hub_namespace: e.g. "yourname" -- used to build repo_id as
        "{hub_namespace}/xai-{dataset_name}-{model_name}" for any run whose
        model_info.json doesn't already have a hub_repo_id recorded. Not
        needed if repo_id_fn is given, or if every pending run already has
        a hub_repo_id in its model_info.json.
    repo_id_fn: optional callable (domain, dataset_name, model_name) -> repo_id,
        overriding the default naming scheme above.
    dry_run: if True, prints/logs exactly what WOULD be uploaded (repo ids,
        files) without calling the Hub at all. Uploads nothing. Run this
        first.
    force: if True, re-uploads runs that already have a hub_commit_sha
        instead of skipping them. Off by default -- the whole point of this
        tool is to only touch what's missing.

    Returns a summary dict: {"uploaded": [...], "skipped": [...], "failed": [...]}.
    Each entry identifies the run (domain/dataset_name/model_name) and, for
    uploaded/failed, the repo_id and (for uploaded) the resulting commit SHA
    or (for failed) the error message.
    """
    os.makedirs(models_root, exist_ok=True)
    logger = Logger(models_root, filename="hub_upload_log.txt")
    logger.section("HUB UPLOAD" + (" (DRY RUN)" if dry_run else ""))

    if not dry_run and not _HAS_HFHUB:
        raise ImportError("huggingface_hub is not installed; cannot push to the Hub. "
                           "Install it, or pass dry_run=True to preview without uploading.")

    runs = scan_models(models_root)
    logger.log(f"Found {len(runs)} completed training run(s) under {models_root}")

    summary = {"uploaded": [], "skipped": [], "failed": []}

    for run in runs:
        label = f"{run['domain']}/{run['dataset_name']}/{run['model_name']}"

        if run["already_uploaded"] and not force:
            logger.log(f"SKIP  {label}  (already uploaded, commit {run['model_info']['hub_commit_sha']})")
            summary["skipped"].append({**{k: run[k] for k in ('domain', 'dataset_name', 'model_name')},
                                        "hub_commit_sha": run["model_info"]["hub_commit_sha"]})
            continue

        model_info = run["model_info"]
        repo_id = model_info.get("hub_repo_id")
        if not repo_id:
            if repo_id_fn:
                repo_id = repo_id_fn(run["domain"], run["dataset_name"], run["model_name"])
            elif hub_namespace:
                repo_id = _default_repo_id(hub_namespace, run["dataset_name"], run["model_name"])
            else:
                logger.log(f"FAIL  {label}  (no hub_repo_id recorded and no hub_namespace/repo_id_fn given)")
                summary["failed"].append({**{k: run[k] for k in ('domain', 'dataset_name', 'model_name')},
                                           "error": "no repo_id available"})
                continue

        # Locate the saved model artifact (.skops preferred, .joblib fallback --
        # matches save_model_artifact's own preference order in training.py).
        model_path = None
        for fname in ("model.skops", "model.joblib"):
            candidate = os.path.join(run["model_dir"], fname)
            if os.path.exists(candidate):
                model_path = candidate
                break
        schema_path = os.path.join(run["model_dir"], "feature_schema.json")
        test_metrics_path = os.path.join(run["metadata_dir"], "test_metrics.json")

        missing = [p for p in (model_path, schema_path, test_metrics_path) if not p or not os.path.exists(p)]
        if missing:
            logger.log(f"FAIL  {label}  (missing expected file(s): {missing})")
            summary["failed"].append({**{k: run[k] for k in ('domain', 'dataset_name', 'model_name')},
                                       "error": f"missing files: {missing}"})
            continue

        if dry_run:
            logger.log(f"WOULD UPLOAD  {label}  ->  {repo_id}  (private={hub_private})")
            summary["uploaded"].append({**{k: run[k] for k in ('domain', 'dataset_name', 'model_name')},
                                         "repo_id": repo_id, "hub_commit_sha": None, "dry_run": True})
            continue

        with open(test_metrics_path) as f:
            test_metrics = json.load(f)

        try:
            commit_sha = push_to_hub(
                repo_id, model_path, schema_path, test_metrics_path, run["model_info_path"],
                run["dataset_name"], run["domain"], model_info.get("model_type", run["model_name"]),
                test_metrics, hub_private, logger,
            )
            logger.log(f"UPLOADED  {label}  ->  {repo_id}  (commit {commit_sha})")
            summary["uploaded"].append({**{k: run[k] for k in ('domain', 'dataset_name', 'model_name')},
                                         "repo_id": repo_id, "hub_commit_sha": commit_sha})
        except Exception as e:
            logger.log(f"FAIL  {label}  ->  {repo_id}  ({e})")
            summary["failed"].append({**{k: run[k] for k in ('domain', 'dataset_name', 'model_name')},
                                       "repo_id": repo_id, "error": str(e)})

    logger.section("HUB UPLOAD SUMMARY")
    logger.log(f"Uploaded: {len(summary['uploaded'])}  |  Skipped (already uploaded): {len(summary['skipped'])}  |  Failed: {len(summary['failed'])}")
    for bucket in ("uploaded", "skipped", "failed"):
        for r in summary[bucket]:
            logger.log(f"  [{bucket}] {r['domain']}/{r['dataset_name']}/{r['model_name']}")

    return summary
