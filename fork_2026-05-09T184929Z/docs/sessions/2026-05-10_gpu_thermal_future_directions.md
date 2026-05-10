# GPU Thermal Management — Future Directions
**Date:** 2026-05-10  
**Host:** worlock

---

## 1. Fan curve control for the RTX 5080

The 5080 lacks `-gtt` support, so its thermal management is indirect (power cap). A more precise approach would be a custom fan curve — forcing higher fan speeds at lower temperatures to keep it in the 60–65°C range rather than 66–70°C. This requires either:

- `nvidia-settings` with an X display or virtual framebuffer (e.g. `Xvfb :1 &`)
- A fan control loop using `nvidia-settings -a [gpu:0]/GPUFanControlState=1` and `GPUTargetFanSpeed`

Worth exploring if the 5080 starts approaching 70°C+ under heavier workloads.

---

## 2. Per-workload power profiles

Currently eco/perf is binary. A more granular approach:

- **Inference-only:** 300W cap both GPUs, locked clocks on 3080
- **Training/fine-tuning:** lift 3080 to 340W, keep 5080 at 300W
- **Idle:** drop 5080 to 250W

The watchdog could be extended to read GPU utilization thresholds and select from multiple profiles rather than just eco/perf.

---

## 3. Temperature-triggered alerts

Add alerting when either GPU approaches its thermal ceiling. Options:
- Extend `gpu_watchdog.sh` to emit a notification (e.g. `wall`, `notify-send`, or a curl to a webhook) if temp exceeds a threshold (e.g. 80°C on 5080, 75°C on 3080)
- Hook into Netdata (already running on port 19999) — it has NVIDIA GPU plugin support and can alert on temp thresholds via its health engine

---

## 4. Persistent power limits via nvidia-persistenced tuning

Currently persistence depends on the `gpu-eco-mode.service` oneshot. An alternative is configuring `nvidia-persistenced` with a power limit baked into the driver's persistence state, removing the dependency on the toggle script at boot. Worth investigating if the service ordering ever causes issues.

---

## 5. Blackwell thermal API tracking

NVIDIA may expose more thermal controls for Blackwell in future driver releases (580.x → 6xx). The `-gtt` feature's absence is likely a firmware/driver decision, not a hardware limitation. Worth re-testing `SUPPORTED_GPU_TARGET_TEMP` after major driver updates.

```bash
# Re-test after driver upgrade
sudo nvidia-smi -i 0 -q -d SUPPORTED_GPU_TARGET_TEMP
```

---

## 6. Unified GPU management dashboard

The `gpu_watchdog.sh` already has a colorized dashboard mode. Consider extending it to display:
- Current mode (eco/perf)
- Both GPU temps + power draws side by side
- Time-in-mode counters
- Throttle event history

This would give a single-pane view of dual-GPU thermal state without needing to run `nvidia-smi` manually.

---

## 7. Validate thermal settings survive a full reboot

A `systemctl restart gpu-eco-mode.service` was verified, but a full cold reboot has not been tested in this session. After the next scheduled reboot, confirm:

```bash
nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
nvidia-smi -i 1 -q | grep "GPU Target Temperature"
```

Expected: both GPUs at 300W, 3080 target temp 70°C.
