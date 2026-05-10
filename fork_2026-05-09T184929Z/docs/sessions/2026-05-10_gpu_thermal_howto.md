# GPU Thermal Management — How-To Guide
**Date:** 2026-05-10  
**Host:** worlock  
**GPUs:** RTX 5080 (GPU 0, Blackwell) · RTX 3080 (GPU 1, Ampere)

---

## Check what thermal controls are available

```bash
# Query target temp support (Ampere supports it; Blackwell does not)
sudo nvidia-smi -i <gpu_id> -q -d SUPPORTED_GPU_TARGET_TEMP

# Check current temps, power draw, and limits
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,fan.speed \
  --format=csv,noheader

# Check thermal thresholds (slowdown, shutdown)
nvidia-smi -q | grep -E "Slowdown Temp|Shutdown Temp|Max Operating Temp|T\.Limit Temp"
```

---

## Set GPU Target Temperature (RTX 3080 only)

```bash
# Supported range: 65–91°C. Current setting: 70°C.
sudo nvidia-smi -i 1 -gtt 70

# Verify
nvidia-smi -i 1 -q | grep "GPU Target Temperature"
```

---

## Cap GPU power limit

```bash
# RTX 5080 — range 250–400W, capped at 300W
sudo nvidia-smi -i 0 -pl 300

# RTX 3080 — range 100–375W, capped at 300W
sudo nvidia-smi -i 1 -pl 300

# Verify both
nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
```

---

## Switch between eco and perf modes

```bash
# ECO: locked low clocks, 300W cap, 70°C target (3080), 300W cap (5080)
sudo /home/jeb/programs/gemini_cli_workspace/ampere-vision-archival/gpu_power_toggle.sh eco

# PERF: auto clocks, 300W cap, 70°C target (3080), 300W cap (5080)
sudo /home/jeb/programs/gemini_cli_workspace/ampere-vision-archival/gpu_power_toggle.sh perf
```

---

## Make settings persist across reboots

Settings applied via `nvidia-smi` reset on driver reload. The `gpu-eco-mode.service` re-applies them at boot:

```bash
# Restart the service to apply updated script settings immediately
sudo systemctl restart gpu-eco-mode.service

# Check service ran cleanly
systemctl status gpu-eco-mode.service --no-pager

# View recent service output
journalctl -u gpu-eco-mode.service -n 30 --no-pager
```

---

## Monitor GPU temps live

```bash
# One-shot snapshot
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,fan.speed \
  --format=csv,noheader

# Continuous watch (every 2s)
watch -n 2 "nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,fan.speed \
  --format=csv,noheader"
```

---

## Inspect the gpu-watchdog

The watchdog (`gpu-watchdog.service`) monitors GPU 1 utilization and automatically toggles eco/perf mode:

```bash
# View recent watchdog activity
journalctl -u gpu-watchdog.service -n 50 --no-pager

# Check watchdog is running
systemctl status gpu-watchdog.service --no-pager
```

---

## Key files

| File | Purpose |
|------|---------|
| `/home/jeb/programs/gemini_cli_workspace/ampere-vision-archival/gpu_power_toggle.sh` | Eco/perf toggle script — edit power limits and target temps here |
| `/etc/systemd/system/gpu-eco-mode.service` | Calls toggle script at boot |
| `/etc/systemd/system/gpu-watchdog.service` | Runs watchdog daemon |
| `/home/jeb/programs/gemini_cli_workspace/ampere-vision-archival/tools/gpu_watchdog.sh` | Watchdog logic — load detection thresholds here |
