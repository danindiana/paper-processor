# GPU Thermal Management Session
**Date:** 2026-05-10T00:49:22-05:00  
**Host:** worlock (Ubuntu/Debian, Linux 6.8.12)  
**User:** jeb

---

## Objective
Investigate and implement temperature capping for the dual NVIDIA GPUs on this machine.

---

## Hardware Inventory

| Slot | GPU | VRAM | Default TDP | Power Range |
|------|-----|------|-------------|-------------|
| GPU 0 | RTX 5080 (Blackwell) | ~16 GB | 360 W | 250–400 W |
| GPU 1 | RTX 3080 (Ampere) | ~10 GB | 340 W | 100–375 W |

---

## Findings

### RTX 3080 (GPU 1) — Full thermal target support
- `nvidia-smi -gtt` (GPU Target Temperature) is supported.
- Supported range: **65°C – 91°C**.
- Default target: **83°C**.
- The driver dynamically adjusts fan speed and clock scaling to keep the GPU at the target.
- **Set to 70°C** for this machine (revised down from initial 80°C).

### RTX 5080 (GPU 0) — Target temp NOT supported
- `GPU Target Temperature: N/A` — Blackwell architecture does not expose this control via nvidia-smi.
- The GPU uses newer internal firmware-managed thermal control.
- **Workaround: power cap.** Reducing the power limit indirectly caps heat output.
- **Set to 300 W** (from default 360 W) — approximately 17% power reduction.

---

## Changes Made

### 1. `gpu_power_toggle.sh` updated
**File:** `/home/jeb/programs/gemini_cli_workspace/ampere-vision-archival/gpu_power_toggle.sh`

Added to both `eco` and `perf` modes:
- `nvidia-smi -i 1 -gtt 70` — RTX 3080 target temp 70°C
- `nvidia-smi -pm 1 -i 0` + `nvidia-smi -pl 300 -i 0` (eco) / `pl 360` (perf) — RTX 5080 power cap

### 2. Live settings applied
Settings were applied immediately without a reboot:
```
RTX 5080: power limit 360 W → 300 W
RTX 3080: GPU Target Temperature → 70°C (revised from 80°C)
```

### 3. Persistence mechanism
`gpu-eco-mode.service` (systemd, runs at boot after `nvidia-persistenced`) calls `gpu_power_toggle.sh eco`, so these settings will be restored on every boot automatically.

---

## Verified State (post-change)

```
GPU 0  RTX 5080   65°C   158 W / 300 W limit   Target Temp: N/A (firmware-managed)
GPU 1  RTX 3080   59°C   124 W / 340 W limit   Target Temp: 70°C
```

---

## Commands Reference

```bash
# Set RTX 3080 target temp (65–91°C range)
sudo nvidia-smi -i 1 -gtt <temp>

# Set RTX 5080 power limit (250–400W range)
sudo nvidia-smi -i 0 -pl <watts>

# Query supported target temp range (GPU 1 only)
sudo nvidia-smi -i 1 -q -d SUPPORTED_GPU_TARGET_TEMP

# Apply eco/perf mode (persisted via gpu-eco-mode.service)
/home/jeb/programs/gemini_cli_workspace/ampere-vision-archival/gpu_power_toggle.sh [eco|perf]

# Reload gpu-eco-mode service (e.g. after script edits)
sudo systemctl restart gpu-eco-mode.service
```

---

## Notes
- The `-gtt` setting resets on driver reload; the `gpu-eco-mode.service` handles re-application at boot.
- The RTX 5080's power cap (300 W) is conservative for inference workloads. Raise to 360 W (perf mode) if throughput bottlenecks appear.
- `nvidia-persistenced` must be running for persistence mode to hold across power state changes; it is enabled and active on this machine.
