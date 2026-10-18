#!/usr/bin/env python3
"""Generate one CogVideoX-1.5 video with RECAP.

    python scripts/cog_t2v_inference.py \
        --model-path /path/to/cogvideox1.5-5b \
        --config configs/cogvideox1.5_fast.json \
        --prompt "A serene lakeside at dawn" \
        --out video.mp4

Add --dense to run the unmodified model with the same prompt and seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recap import install, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True,
                        help="CogVideoX-1.5 checkpoint directory or Hub id")
    parser.add_argument("--config", default="configs/cogvideox1.5_fast.json",
                        help="RECAP config JSON")
    parser.add_argument("--prompt", default=None,
                        help="text prompt (or pass --example to use the preset's)")
    parser.add_argument("--example", action="store_true",
                        help="use the prompt and seed recorded in the config's "
                             "'example' block -- an example prompt to "
                             "check an install")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--out", default="video.mp4")
    parser.add_argument("--trace", default=None,
                        help="Write the per-step schedule trace to this JSON file")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: the example seed with --example, else 0)")
    parser.add_argument("--no-offload", action="store_true",
                        help="Keep every module resident (needs more VRAM)")
    parser.add_argument("--dense", action="store_true",
                        help="Skip RECAP entirely: the unmodified reference trajectory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.example:
        example = json.loads(Path(args.config).read_text()).get("example")
        if not example:
            raise SystemExit(f"{args.config} has no 'example' block")
        if args.prompt is None:
            args.prompt = example["prompt"]
        if args.seed is None:
            args.seed = int(example.get("seed", 0))
        print(f"[example] prompt={args.prompt!r} seed={args.seed}")
    if args.seed is None:
        args.seed = 0
    if args.prompt is None:
        raise SystemExit("give --prompt, or --example to use the preset's own")

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if args.no_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    session = None
    if not args.dense:
        session = install(pipe, config)
        session.reset()
        print(json.dumps(session.model_info, indent=2))

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=config.schedule.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames[0]
    torch.cuda.synchronize()
    latency = time.perf_counter() - started

    export_to_video(frames, args.out, fps=args.fps)
    print(f"wrote {args.out}  ({latency:.2f} s, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")

    if session is not None:
        summary = session.summary()
        print("schedule:", "".join({"exact": "E", "refresh": "F", "sparse": "S", "reuse": "U"}[a]
                                  for a in summary["schedule"]))
        print("counts:", summary["counts"],
              "| endpoints:", summary["computed_endpoints"],
              "| reuse gain:", summary["reuse_gain_final"])
        if args.trace:
            Path(args.trace).write_text(json.dumps(session.actions, indent=1) + "\n")


if __name__ == "__main__":
    main()
