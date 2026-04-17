#!/usr/bin/env python3
"""
usage:
  python inference.py --lora_dir <path> [--guidance 7.5] [--gpu 0]
"""

import sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lora_dir",  required=True)
parser.add_argument("--guidance",  type=float, default=7.5)
parser.add_argument("--gpu",       default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f"  GPU={args.gpu}  guidance={args.guidance}")

import torch, csv
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw

DEVICE     = "cuda"
BASE_MODEL = "/home/qianqz/Model/stable-diffsion-v1.5-fp32"
NUM_STEPS  = 30
SEEDS      = [0, 42, 1337, 2025, 9999]

# -- Prompt Sets --------------------------------------------------------------
prompts_subject = [
    ("subj_forest", "a photo of sks dragon toy in a forest"),
    ("subj_beach",  "a photo of sks dragon toy on a beach"),
    ("subj_snow",   "a photo of sks dragon toy in the snow"),
    ("subj_city",   "a photo of sks dragon toy on a city street"),
    ("subj_table",  "a photo of sks dragon toy on a wooden table"),
]
prompts_style = [
    ("prmpt_oil",   "a photo of sks dragon toy, oil painting style"),
    ("prmpt_anime", "a photo of sks dragon toy, anime style"),
    ("prmpt_lego",  "a photo of sks dragon toy made of lego bricks"),
    ("prmpt_rain",  "a photo of sks dragon toy in the rain"),
    ("prmpt_neon",  "a photo of sks dragon toy in a neon lit room"),
    ("prmpt_hat",   "a photo of sks dragon toy wearing a red hat"),
    ("prmpt_scarf", "a photo of sks dragon toy with a scarf around its neck"),
]
prompts_prior = [
    ("prior_1", "a photo of dragon toy"),
    ("prior_2", "a photo of a dragon toy on a shelf"),
    ("prior_3", "a photo of a green dragon toy"),
    ("prior_4", "a photo of a dragon toy in a box"),
]
ALL_TASKS = [
    ("A_subject_fidelity",   prompts_subject),
    ("B_prompt_fidelity",    prompts_style),
    ("C_prior_preservation", prompts_prior),
]

# -- Path Configuration -------------------------------------------------------
lora_dir = os.path.expanduser(args.lora_dir)
eval_dir = os.path.join(lora_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

# -- Grid Generation ----------------------------------------------------------
def make_grid(task_dir, out_path):
    thumb, hdr = 256, 28
    subs = sorted([s for s in os.listdir(task_dir)
                   if os.path.isdir(os.path.join(task_dir, s))])
    if not subs: return
    grid = Image.new("RGB", (thumb*len(SEEDS), (thumb+hdr)*len(subs)), (30,30,30))
    draw = ImageDraw.Draw(grid)
    for r, sub in enumerate(subs):
        draw.text((6, r*(thumb+hdr)+4), sub, fill=(220,220,220))
        imgs = sorted([f for f in os.listdir(os.path.join(task_dir,sub))
                       if f.endswith(".png")])
        for c, fn in enumerate(imgs[:len(SEEDS)]):
            img = Image.open(os.path.join(task_dir,sub,fn)).resize((thumb,thumb))
            grid.paste(img, (c*thumb, r*(thumb+hdr)+hdr))
    grid.save(out_path)
    print(f"  📊 Grid → {out_path}")

# -- Inference ----------------------------------------------------------------
print(f"\n  Loading model: {lora_dir}")
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lora_dir)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

for task_name, prompt_list in ALL_TASKS:
    task_dir = os.path.join(eval_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    print(f"\n  [{task_name}]")
    for tag, prompt in prompt_list:
        sub_dir = os.path.join(task_dir, tag)
        os.makedirs(sub_dir, exist_ok=True)
        for seed in SEEDS:
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            img = pipe(prompt, num_inference_steps=NUM_STEPS,
                       guidance_scale=args.guidance, generator=gen,
                       height=512, width=512).images[0]
            img.save(os.path.join(sub_dir, f"seed{seed}.png"))
        print(f"    {tag}")
    make_grid(task_dir, os.path.join(eval_dir, f"{task_name}_grid.png"))

del pipe; torch.cuda.empty_cache()

# -- Diversity Analysis -------------------------------------------------------
print(f"\n  Performing diversity analysis...")
results = []
for task_name, _ in ALL_TASKS:
    task_dir = os.path.join(eval_dir, task_name)
    for sub in sorted(os.listdir(task_dir)):
        sub_path = os.path.join(task_dir, sub)
        if not os.path.isdir(sub_path): continue
        imgs = sorted([os.path.join(sub_path, f)
                       for f in os.listdir(sub_path) if f.endswith(".png")])
        arrs = [np.array(Image.open(p).resize((128,128))).astype(float) for p in imgs[:5]]
        diffs = [np.mean(np.abs(arrs[i]-arrs[j]))/255.0
                 for i in range(len(arrs)) for j in range(i+1,len(arrs))]
        mad = round(float(np.mean(diffs)) if diffs else 0.0, 4)
        results.append({"task": task_name, "prompt": sub, "MAD": mad, "LPIPS": None})

try:
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)   # Move the LPIPS model to the active device.
    for r in results:
        task_dir = os.path.join(eval_dir, r["task"])
        sub_path = os.path.join(task_dir, r["prompt"])
        imgs = sorted([os.path.join(sub_path, f)
                       for f in os.listdir(sub_path) if f.endswith(".png")])
        tensors = [torch.from_numpy(
                       np.array(Image.open(p).resize((256,256))
                       ).astype(np.float32)/127.5-1
                   ).permute(2,0,1).unsqueeze(0).to(DEVICE)   # Move tensors to the active device.
                   for p in imgs[:5]]
        with torch.no_grad():   # Disable gradient tracking to reduce memory consumption.
            dists = [loss_fn(tensors[i],tensors[j]).item()
                     for i in range(len(tensors)) for j in range(i+1,len(tensors))]
        r["LPIPS"] = round(float(np.mean(dists)), 4)
    print("  ✅ LPIPS evaluation completed")
except ImportError:
    print("  ⚠️  Install lpips with `pip install lpips` to enable LPIPS evaluation.")

csv_path = os.path.join(eval_dir, "diversity.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader(); writer.writerows(results)
print(f"  📄 CSV → {csv_path}")

from collections import defaultdict
summary = defaultdict(list)
for r in results: summary[r["task"]].append(r["MAD"])
print(f"\n  {'Task':<35} {'avg MAD':>8}")
for t, v in summary.items():
    print(f"  {t:<35} {np.mean(v):>8.4f}")
print(f"\n  Output directory: {eval_dir}")
