#!/usr/bin/env python3
"""Upload a TFG try folder to the cluster and submit one or more Slurm jobs.

This is a consolidated replacement for the many near-identical upload scripts.
It supports the existing tries via presets and also accepts explicit local and
Slurm paths for future experiments.

Examples:
    python cluster/upload_and_submit_experiments.py --preset ninth --gpus 1
    python cluster/upload_and_submit_experiments.py --preset thirteenth --gpus 1
    python cluster/upload_and_submit_experiments.py --local-dir TFGCityRegimeTry15 --slurm cluster/run_cityregime_try15_1gpu.slurm --gpus 1
    python cluster/upload_and_submit_experiments.py --preset ninth --gpus 1 --upload-only
"""
from __future__ import annotations

import argparse
import io
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

_CLUSTER_DIR = Path(__file__).resolve().parent
if str(_CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_DIR))

import upload_dataset_helpers as udh

try:
    import paramiko
except ImportError:
    os.system(f"{sys.executable} -m pip install paramiko -q")
    import paramiko


HOST = "sert.ac.upc.edu"
USER = "gmoreno"
REMOTE_BASE = "/scratch/nas/3/gmoreno/TFGpractice"
EXCLUDE_DIRS = {"outputs", "__pycache__", ".git", ".venv", ".cursor"}
EXCLUDE_EXTS = {".h5", ".pt", ".pth", ".pyc", ".pyo"}


@dataclass(frozen=True)
class Preset:
    local_dir: str
    slurms_by_gpus: dict[int, tuple[str, ...]] | None = None
    default_slurms: tuple[str, ...] = ()


PRESETS: dict[str, Preset] = {
    "fifteenth": Preset(
        local_dir="TFGFifteenthTry15",
        slurms_by_gpus={
            1: ("cluster/run_fifteenthtry15_cityregime_1gpu.slurm",),
            2: ("cluster/run_fifteenthtry15_cityregime_2gpu.slurm",),
        },
    ),
    "sixteenth": Preset(
        local_dir="TFGSixteenthTry16",
        slurms_by_gpus={
            1: ("cluster/run_sixteenthtry16_cityregime_1gpu.slurm",),
            2: ("cluster/run_sixteenthtry16_cityregime_2gpu.slurm",),
        },
    ),
    "seventeenth": Preset(
        local_dir="TFGSeventeenthTry17",
        slurms_by_gpus={
            1: ("cluster/run_seventeenthtry17_cityregime_1gpu.slurm",),
            2: ("cluster/run_seventeenthtry17_cityregime_2gpu.slurm",),
        },
    ),
    "eighteenth": Preset(
        local_dir="TFGEighteenthTry18",
        slurms_by_gpus={
            1: ("cluster/run_eighteenthtry18_cityregime_1gpu.slurm",),
            2: ("cluster/run_eighteenthtry18_cityregime_2gpu.slurm",),
        },
    ),
    "nineteenth": Preset(
        local_dir="TFGNineteenthTry19",
        slurms_by_gpus={
            1: ("cluster/run_nineteenthtry19_cityregime_1gpu.slurm",),
            2: ("cluster/run_nineteenthtry19_cityregime_2gpu.slurm",),
        },
    ),
    "twentieth": Preset(
        local_dir="TFGTwentiethTry20",
        slurms_by_gpus={
            1: ("cluster/run_twentiethtry20_cityregime_1gpu.slurm",),
            2: ("cluster/run_twentiethtry20_cityregime_2gpu.slurm",),
        },
    ),
    "twentyfirst": Preset(
        local_dir="TFGTwentyFirstTry21",
        slurms_by_gpus={
            2: ("cluster/run_twentyfirsttry21_multiscale_2gpu.slurm",),
        },
    ),
    "twentysecond": Preset(
        local_dir="TFGTwentySecondTry22",
        slurms_by_gpus={
            2: ("cluster/run_twentysecondtry22_decoder_multiscale_2gpu.slurm",),
        },
    ),
    "twentythird": Preset(
        local_dir="TFGTwentyThirdTry23",
        slurms_by_gpus={
            2: ("cluster/run_twentythirdtry23_delay_angular_multiscale_2gpu.slurm",),
        },
    ),
    "twentyfifth": Preset(
        local_dir="TFGTwentyFifthTry25",
        slurms_by_gpus={
            1: ("cluster/run_twentyfifthtry25_bottleneck_attention_1gpu.slurm",),
        },
    ),
    "twentysixth": Preset(
        local_dir="TFGTwentySixthTry26",
        slurms_by_gpus={
            1: ("cluster/run_twentysixthtry26_delay_angular_gradient_1gpu.slurm",),
            2: ("cluster/run_twentysixthtry26_delay_angular_gradient_2gpu.slurm",),
        },
    ),
    "twentyseventh": Preset(
        local_dir="TFGTwentySeventhTry27",
        slurms_by_gpus={
            1: ("cluster/run_twentyseventhtry27_topology_edge_pathloss_1gpu.slurm",),
        },
    ),
    "twentyeighth": Preset(
        local_dir="TFGTwentyEighthTry28",
        slurms_by_gpus={
            1: ("cluster/run_twentyeighthtry28_attention_topology_1gpu.slurm",),
            2: ("cluster/run_twentyeighthtry28_attention_topology_2gpu.slurm",),
            4: ("cluster/run_twentyeighthtry28_attention_topology_4gpu.slurm",),
        },
    ),
    "twentyninth": Preset(
        local_dir="TFGTwentyNinthTry29",
        slurms_by_gpus={
            2: ("cluster/run_twentyninthtry29_radial_pathloss_2gpu.slurm",),
        },
    ),
    "thirtieth": Preset(
        local_dir="TFGThirtiethTry30",
        slurms_by_gpus={
            2: ("cluster/run_thirtiethtry30_spread_priority_2gpu.slurm",),
        },
    ),
    "thirtyfirst": Preset(
        local_dir="TFGThirtyFirstTry31",
        slurms_by_gpus={
            2: ("cluster/run_thirtyfirsttry31_prior_residual_pathloss_2gpu.slurm",),
        },
    ),
    "thirtysecond": Preset(
        local_dir="TFGThirtySecondTry32",
        slurms_by_gpus={
            2: ("cluster/run_thirtysecondtry32_support_amplitude_spread_2gpu.slurm",),
        },
    ),
    "thirtythird": Preset(
        local_dir="TFGThirtyThirdTry33",
        slurms_by_gpus={
            2: ("cluster/run_thirtythirdtry33_buildingmask_pathloss_2gpu.slurm",),
        },
    ),
    "thirtyfourth": Preset(
        local_dir="TFGThirtyFourthTry34",
        slurms_by_gpus={
            2: ("cluster/run_thirtyfourthtry34_hybrid_two_ray_prior_input_2gpu.slurm",),
            1: ("cluster/run_thirtyfourthtry34_hybrid_two_ray_prior_input_1gpu.slurm",),

        },
    ),
    "thirtyfifth": Preset(
        local_dir="TFGThirtyFifthTry35",
        slurms_by_gpus={
            1: ("cluster/run_thirtyfifthtry35_spread_buildingmask_1gpu.slurm",),
        },
    ),
    "thirtysixth": Preset(
        local_dir="TFGThirtySixthTry36",
        slurms_by_gpus={
            1: ("cluster/run_thirtysixthtry36_spread_buildingmask_1gpu.slurm",),
            2: ("cluster/run_thirtysixthtry36_spread_buildingmask_2gpu.slurm",),
        },
    ),
    "thirtyseventh": Preset(
        local_dir="TFGThirtySeventhTry37",
        slurms_by_gpus={
            1: ("cluster/run_thirtyseventhtry37_buildingmask_pathloss_1gpu.slurm",),
            2: ("cluster/run_thirtyseventhtry37_buildingmask_pathloss_2gpu.slurm",),
        },
    ),
    "thirtyeighth": Preset(
        local_dir="TFGThirtyEighthTry38",
        slurms_by_gpus={
            1: ("cluster/run_thirtyeighthtry38_hybrid_two_ray_prior_input_1gpu.slurm",),
            2: ("cluster/run_thirtyeighthtry38_hybrid_two_ray_prior_input_2gpu.slurm",),
        },
    ),
    "thirtyninth": Preset(
        local_dir="TFGThirtyNinthTry39",
        slurms_by_gpus={
            1: ("cluster/run_thirtyninthtry39_spread_buildingmask_1gpu.slurm",),
            2: ("cluster/run_thirtyninthtry39_spread_buildingmask_2gpu.slurm",),
        },
    ),
    "fortieth": Preset(
        local_dir="TFGFortiethTry40",
        slurms_by_gpus={
            1: ("cluster/run_fortiethtry40_spread_buildingmask_1gpu.slurm",),
            2: ("cluster/run_fortiethtry40_spread_buildingmask_2gpu.slurm",),
        },
    ),
    "fortyfirst": Preset(
        local_dir="TFGFortyFirstTry41",
        slurms_by_gpus={
            1: ("cluster/run_fortyfirsttry41_prior_residual_formula_1gpu.slurm",),
            2: ("cluster/run_fortyfirsttry41_prior_residual_formula_2gpu.slurm",),
        },
    ),
    "fortysecond": Preset(
        local_dir="TFGFortySecondTry42",
        slurms_by_gpus={
            1: ("cluster/run_fortysecondtry42_pmnet_prior_residual_1gpu.slurm",),
            2: ("cluster/run_fortysecondtry42_pmnet_prior_residual_2gpu.slurm",),
        },
    ),
    "fortythird": Preset(
        local_dir="TFGFortyThirdTry43",
        slurms_by_gpus={
            1: ("cluster/run_fortythirdtry43_pmnet_no_prior_1gpu.slurm",),
            2: ("cluster/run_fortythirdtry43_pmnet_no_prior_2gpu.slurm",),
        },
    ),
    "fortyfourth": Preset(
        local_dir="TFGFortyFourthTry44",
        slurms_by_gpus={
            1: ("cluster/run_fortyfourthtry44_pmnet_v3_no_prior_1gpu.slurm",),
            2: ("cluster/run_fortyfourthtry44_pmnet_v3_no_prior_2gpu.slurm",),
        },
    ),
    "fortyfifth": Preset(
        local_dir="TFGFortyFifthTry45",
        slurms_by_gpus={
            1: ("cluster/run_fortyfifthtry45_pmnet_moe_enhanced_prior_1gpu.slurm",),
            2: ("cluster/run_fortyfifthtry45_pmnet_moe_enhanced_prior_2gpu.slurm",),
        },
    ),
    "fortysixth": Preset(
        local_dir="TFGFortySixthTry46",
        slurms_by_gpus={
            1: ("cluster/run_fortysixthtry46_los_nlos_moe_prior_1gpu.slurm",),
            2: ("cluster/run_fortysixthtry46_los_nlos_moe_prior_2gpu.slurm",),
        },
    ),
    "fortyseventh": Preset(
        local_dir="TFGFortySeventhTry47",
        slurms_by_gpus={
            2: ("cluster/run_fortyseventhtry47_unet_prior_nlos_moe_2gpu.slurm",),
        },
    ),
    "ninth": Preset(
        local_dir="TFGNinthTry9",
        slurms_by_gpus={
            1: ("cluster/run_ninthtry9_1gpu.slurm",),
            2: ("cluster/run_ninthtry9_2gpu.slurm",),
        },
    ),
    "tenth-los": Preset(
        local_dir="TFGTenthTry10",
        slurms_by_gpus={
            1: ("cluster/run_tenthtry10_los_1gpu.slurm",),
            2: ("cluster/run_tenthtry10_los_2gpu.slurm",),
        },
    ),
    "tenth-nlos": Preset(
        local_dir="TFGTenthTry10",
        slurms_by_gpus={
            1: ("cluster/run_tenthtry10_nlos_1gpu.slurm",),
            2: ("cluster/run_tenthtry10_nlos_2gpu.slurm",),
        },
    ),
    "eleventh": Preset(
        local_dir="TFGEleventhTry11",
        slurms_by_gpus={
            1: ("cluster/run_eleventhtry11_1gpu.slurm",),
            2: ("cluster/run_eleventhtry11_2gpu.slurm",),
        },
    ),
    "twelfth-los": Preset(
        local_dir="TFGTwelfthTry12",
        slurms_by_gpus={
            1: ("cluster/run_twelfthtry12_los_1gpu.slurm",),
            2: ("cluster/run_twelfthtry12_los_2gpu.slurm",),
        },
    ),
    "twelfth-nlos": Preset(
        local_dir="TFGTwelfthTry12",
        slurms_by_gpus={
            1: ("cluster/run_twelfthtry12_nlos_1gpu.slurm",),
            2: ("cluster/run_twelfthtry12_nlos_2gpu.slurm",),
        },
    ),
    "thirteenth": Preset(
        local_dir="TFGThirteenthTry13",
        default_slurms=("cluster/run_thirteenthtry13_film_1gpu.slurm",),
    ),
    "fourteenth-los": Preset(
        local_dir="TFGFourteenthTry14",
        slurms_by_gpus={
            1: ("cluster/run_fourteenthtry14_film_los_1gpu.slurm",),
            2: ("cluster/run_fourteenthtry14_film_los_2gpu.slurm",),
        },
    ),
    "fourteenth-nlos": Preset(
        local_dir="TFGFourteenthTry14",
        slurms_by_gpus={
            1: ("cluster/run_fourteenthtry14_film_nlos_1gpu.slurm",),
            2: ("cluster/run_fourteenthtry14_film_nlos_2gpu.slurm",),
        },
    ),
    "seventythird": Preset(
        local_dir="TFGSeventyThirdTry73",
        slurms_by_gpus={
            1: ("cluster/run_try73_cnn_1gpu.slurm",),
            2: ("cluster/run_try73_cnn_2gpu.slurm",),
            4: ("cluster/run_try73_cnn_4gpu.slurm",),
        },
    ),
}


def should_skip(name: str) -> bool:
    if name in EXCLUDE_DIRS:
        return True
    return os.path.splitext(name)[1].lower() in EXCLUDE_EXTS


def collect_files(local_root: Path) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for root, dirs, filenames in os.walk(local_root):
        dirs[:] = [directory for directory in dirs if not should_skip(directory)]
        for filename in filenames:
            if should_skip(filename):
                continue
            local_path = Path(root) / filename
            rel_path = os.path.relpath(local_path, local_root).replace("\\", "/")
            files.append((str(local_path), rel_path))
    return files


def mkdir_p(sftp, remote_path: str) -> None:
    udh.mkdir_p_with_quota_hint(sftp, remote_path)


def clean_remote_outputs(sftp, remote_dir: str) -> None:
    def rm_r(path: str) -> None:
        try:
            entries = sftp.listdir_attr(path)
        except FileNotFoundError:
            return
        for entry in entries:
            full = f"{path}/{entry.filename}"
            if stat.S_ISDIR(entry.st_mode) and entry.filename in {"outputs", "__pycache__"}:
                rm_r(full)
                try:
                    sftp.rmdir(full)
                except OSError:
                    pass
            elif stat.S_ISDIR(entry.st_mode):
                rm_r(full)

    rm_r(remote_dir)


def put_file(sftp, local_path: str, remote_path: str, rel_path: str) -> None:
    if rel_path.lower().endswith(".slurm"):
        with open(local_path, "rb") as handle:
            blob = handle.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        sftp.putfo(io.BytesIO(blob), remote_path)
        return
    sftp.put(local_path, remote_path)


def resolve_local_dir(root: Path, raw_local_dir: str | None, preset: Preset | None) -> Path:
    if raw_local_dir:
        candidate = Path(raw_local_dir)
        if candidate.is_absolute():
            return candidate
        return root / candidate
    if preset is None:
        raise SystemExit("Provide either --preset or --local-dir")
    return root / preset.local_dir


def resolve_slurms(args: argparse.Namespace, preset: Preset | None) -> tuple[str, ...]:
    if args.slurm:
        return tuple(args.slurm)
    if preset is None:
        return ()
    if preset.slurms_by_gpus:
        slurms = preset.slurms_by_gpus.get(args.gpus)
        if slurms is not None:
            return slurms
        first_key = sorted(preset.slurms_by_gpus)[0]
        return preset.slurms_by_gpus[first_key]
    return preset.default_slurms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument("--local-dir")
    parser.add_argument("--slurm", nargs="*", default=None)
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--no-clean-outputs", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--remote-base", default=REMOTE_BASE)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--ssh-key", default=None, help="Path to a private SSH key for passwordless login.")
    args = parser.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password and not args.ssh_key:
        raise SystemExit(f"Set environment variable {args.password_env} or pass --ssh-key")

    preset = PRESETS.get(args.preset) if args.preset else None
    root = Path(__file__).resolve().parent.parent
    local_dir = resolve_local_dir(root, args.local_dir, preset)
    if not local_dir.is_dir():
        raise SystemExit(f"Missing folder {local_dir}")

    slurms = resolve_slurms(args, preset)
    if not args.upload_only and not slurms:
        raise SystemExit("No Slurm scripts selected. Use --preset, --slurm, or --upload-only.")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {
        "hostname": args.host,
        "username": args.user,
        "timeout": 30,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if password:
        connect_kwargs["password"] = password
    if args.ssh_key:
        connect_kwargs["key_filename"] = args.ssh_key
    client.connect(**connect_kwargs)
    udh.prepare_ssh_transport_for_large_uploads(client)

    try:
        sftp = client.open_sftp()
        if not args.skip_datasets:
            antenna_h5 = udh.resolve_antenna_height_h5(root)
            if antenna_h5 is not None:
                udh.upload_if_missing_file(sftp, mkdir_p, antenna_h5, udh.remote_antenna_h5_path(args.remote_base))
            else:
                print(f"Warning: no local antenna HDF5 found in {udh.LOCAL_ANTENNA_H5_CANDIDATES}")

            main_h5 = udh.resolve_main_maps_h5(root)
            if main_h5 is not None:
                udh.upload_if_missing_file(sftp, mkdir_p, main_h5, udh.remote_main_h5_path(args.remote_base))
            else:
                print(f"Warning: no local main HDF5 found at {udh.REMOTE_MAIN_MAPS_H5_NAME}")

        remote_dir = f"{args.remote_base.rstrip('/')}/{local_dir.name}"
        mkdir_p(sftp, remote_dir)
        if not args.no_clean_outputs:
            clean_remote_outputs(sftp, remote_dir)

        files = collect_files(local_dir)
        print(f"Uploading {len(files)} files from {local_dir} to {remote_dir}")
        for index, (local_path, rel_path) in enumerate(files, start=1):
            remote_path = f"{remote_dir}/{rel_path}"
            mkdir_p(sftp, os.path.dirname(remote_path).replace("\\", "/"))
            put_file(sftp, local_path, remote_path, rel_path)
            if index == 1 or index == len(files) or index % 40 == 0:
                print(f"  {index}/{len(files)} {rel_path}")

        sftp.close()

        if args.upload_only:
            print("Upload finished; no Slurm jobs submitted.")
            return

        for slurm in slurms:
            cmd = f"cd {remote_dir} && sbatch {slurm}"
            print(cmd)
            _, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            if out:
                print(out)
            if err:
                print(f"stderr: {err}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
