"""Throughput benchmark: fp32 vs fp16 autocast on Hopfield GPT-2 + DREADDIT."""
import sys, time, gc
sys.path.insert(0, "/home/rrobin711/projects/lora-hopfield-clinical")

import torch
import torch.nn as nn
from src.model import load_gpt2
from src.hopfield_gpt2 import inject_hopfield
from src.train import extract_last_hidden_state, seed_everything
from src.data import DataConfig, load_dreaddit
from transformers import GPT2Tokenizer

device = torch.device("cuda")
NUM_STEPS = 20

def run_benchmark(label, batch_size, use_fp16):
    seed_everything(42)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = DataConfig(seed=42, batch_size=batch_size)
    train_loader, _, _ = load_dreaddit(config, tokenizer)
    
    model = load_gpt2()
    inject_hopfield(model)
    model = model.to(device).train()
    head = nn.Linear(768, 2).to(device)
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Warmup (2 steps, not timed)
    loader_iter = iter(train_loader)
    for _ in range(2):
        batch = next(loader_iter)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = extract_last_hidden_state(outputs.last_hidden_state, attention_mask)
            logits = head(pooled)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Timed steps
    start = time.perf_counter()
    total_samples = 0
    inf_count = 0
    
    for step in range(NUM_STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = extract_last_hidden_state(outputs.last_hidden_state, attention_mask)
            logits = head(pooled)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        
        # Check for inf/nan in gradients
        for p in trainable:
            if p.grad is not None and (p.grad.isinf().any() or p.grad.isnan().any()):
                inf_count += 1
                break
        
        scaler.step(optimizer)
        scaler.update()
        total_samples += labels.size(0)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    steps_per_sec = NUM_STEPS / elapsed
    samples_per_sec = total_samples / elapsed
    final_scale = scaler.get_scale() if use_fp16 else "N/A"
    
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  batch_size: {batch_size}, fp16: {use_fp16}")
    print(f"  Steps: {NUM_STEPS}, Total samples: {total_samples}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Steps/sec: {steps_per_sec:.2f}")
    print(f"  Samples/sec: {samples_per_sec:.1f}")
    print(f"  Peak VRAM: {peak_vram_mb:.0f} MB")
    print(f"  Inf/NaN gradient steps: {inf_count}/{NUM_STEPS}")
    print(f"  Final GradScaler scale: {final_scale}")
    print(f"{'='*60}")
    
    # Cleanup
    del model, head, optimizer, scaler, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "label": label, "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec, "peak_vram_mb": peak_vram_mb,
        "inf_count": inf_count, "final_scale": final_scale,
    }

print("=== Throughput Benchmark: Hopfield GPT-2 on DREADDIT ===")

results = []
results.append(run_benchmark("A) fp32, bs=16", batch_size=16, use_fp16=False))
results.append(run_benchmark("B) fp16, bs=16", batch_size=16, use_fp16=True))
results.append(run_benchmark("C) fp16, bs=32", batch_size=32, use_fp16=True))

print("\n\n=== SUMMARY TABLE ===")
print(f"{'Config':<22} {'Steps/s':>8} {'Samples/s':>10} {'VRAM MB':>8} {'Inf steps':>10} {'Scale':>8}")
print("-" * 70)
for r in results:
    print(f"{r['label']:<22} {r['steps_per_sec']:>8.2f} {r['samples_per_sec']:>10.1f} {r['peak_vram_mb']:>8.0f} {r['inf_count']:>10} {str(r['final_scale']):>8}")
