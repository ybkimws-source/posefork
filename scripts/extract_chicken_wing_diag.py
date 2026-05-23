#!/usr/bin/env python
"""
Extract TensorBoard scalars for chicken-wing diagnostics.

Prints a table of arm_angle/{mode,z,y} means over the last N iterations of a run.
Chicken-wing hypothesis: lower workspace z -> elbow flares laterally
(phi more negative at z_lo; phi more negative at y_lat).
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def load_scalars(run_dir: Path, tags: list[str]) -> dict[str, list[tuple[int, float]]]:
    ev_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not ev_files:
        raise FileNotFoundError(f"no event files under {run_dir}")
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    out = {}
    for tag in tags:
        if tag not in ea.Tags()["scalars"]:
            out[tag] = []
            continue
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def tail_mean(series: list[tuple[int, float]], last_n: int) -> float | None:
    if not series:
        return None
    return sum(v for _, v in series[-last_n:]) / min(len(series), last_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory path")
    ap.add_argument("--last_n", type=int, default=300, help="tail window size (iterations)")
    args = ap.parse_args()

    run_dir = Path(args.run)

    n_modes = 3
    tags = []
    for k in range(n_modes):
        tags.append(f"arm_angle/mode{k}_mean")
    for region in ["z_lo", "z_hi", "y_fwd", "y_lat"]:
        tags.append(f"arm_angle/{region}_mean")
        tags.append(f"arm_angle/{region}_std")
    tags += [
        "arm_angle/mean",
        "arm_angle/std",
        "arm_angle/gate_mean",
        "diag/arm_angle_mode_span",
        "diag/arm_angle_z_span",
        "diag/arm_angle_y_span",
    ]

    series = load_scalars(run_dir, tags)
    last_n = args.last_n

    print(f"== {run_dir.name} (last {last_n} iterations) ==\n")

    print("[mode means — target φ_k = (-1.15, -0.97, -0.80)]")
    for k in range(n_modes):
        v = tail_mean(series.get(f"arm_angle/mode{k}_mean", []), last_n)
        print(f"  mode{k}_mean = {v:.3f}" if v is not None else f"  mode{k}_mean = (missing)")

    print("\n[region means — chicken-wing hypothesis: diagnose via z_lo vs z_hi gap]")
    for region in ["z_lo", "z_hi", "y_fwd", "y_lat"]:
        m = tail_mean(series.get(f"arm_angle/{region}_mean", []), last_n)
        s = tail_mean(series.get(f"arm_angle/{region}_std", []), last_n)
        if m is not None:
            print(f"  {region}: mean = {m:+.3f}, std = {s:.3f}")
        else:
            print(f"  {region}: (missing)")

    print("\n[span diagnostics]")
    for tag in ["diag/arm_angle_mode_span", "diag/arm_angle_z_span", "diag/arm_angle_y_span"]:
        v = tail_mean(series.get(tag, []), last_n)
        if v is not None:
            print(f"  {tag} = {v:.3f}")

    print("\n[global]")
    for tag in ["arm_angle/mean", "arm_angle/std", "arm_angle/gate_mean"]:
        v = tail_mean(series.get(tag, []), last_n)
        if v is not None:
            print(f"  {tag} = {v:.3f}")

    print("\n[interpretation hint]")
    z_lo = tail_mean(series.get("arm_angle/z_lo_mean", []), last_n)
    z_hi = tail_mean(series.get("arm_angle/z_hi_mean", []), last_n)
    if z_lo is not None and z_hi is not None:
        print(f"  z_lo - z_hi = {z_lo - z_hi:+.3f} rad")
        print(f"  (negative -> phi more negative at low targets = forced elbow-up = chicken-wing signal)")
    y_lat = tail_mean(series.get("arm_angle/y_lat_mean", []), last_n)
    y_fwd = tail_mean(series.get("arm_angle/y_fwd_mean", []), last_n)
    if y_lat is not None and y_fwd is not None:
        print(f"  y_lat - y_fwd = {y_lat - y_fwd:+.3f} rad")
        print(f"  (large negative -> elbow flares laterally at side targets = chicken-wing signal)")


if __name__ == "__main__":
    main()
