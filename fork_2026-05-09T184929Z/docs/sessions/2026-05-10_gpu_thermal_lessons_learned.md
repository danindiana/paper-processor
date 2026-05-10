# GPU Thermal Management — Lessons Learned
**Date:** 2026-05-10  
**Host:** worlock

---

## 1. Blackwell and Ampere expose different thermal control surfaces

The RTX 5080 (Blackwell) and RTX 3080 (Ampere) look identical in `nvidia-smi` output until you probe thermal control. Blackwell removed the user-facing `GPU Target Temperature` feature (`-gtt`) entirely — it reports `N/A` and `Min/Max: N/A` in `SUPPORTED_GPU_TARGET_TEMP`. Never assume a feature present on one NVIDIA architecture carries forward to the next.

## 2. Power capping is the universal fallback for thermal control

When `-gtt` is unavailable, reducing the power limit (`-pl`) is the most reliable indirect lever. Heat output scales with power draw — capping the 5080 at 300W (from 360W default) reliably held it in the 66–70°C range under sustained inference load.

## 3. The gpu-watchdog was silently lifting power limits under load

The existing `gpu-watchdog.service` monitors GPU 1 (3080) utilization and calls `gpu_power_toggle.sh perf` when load is detected. The perf mode previously hardcoded `nvidia-smi -pl 340 -i 1` and `nvidia-smi -pl 360 -i 0`, bypassing any manually applied caps. This was invisible until the watchdog logs were inspected. **Always audit watchdog/automation scripts before assuming manual settings will persist.**

## 4. Eco mode's locked clocks conflict with -gtt's dynamic scaling

The eco mode script locks the 3080's clocks (`-lgc 210`, `-lmc 405`). The `-gtt` target temperature mechanism works by having the driver dynamically scale clocks to maintain the target — but locked clocks remove that headroom. In practice, with clocks locked at 210 MHz the 3080 runs so cool (54°C at idle) that the conflict is moot, but under full perf mode (auto clocks), `-gtt 70` has full authority to throttle as needed.

## 5. Systemd oneshot services are the right persistence layer for nvidia-smi settings

nvidia-smi settings (`-pl`, `-gtt`, `-pm`) do not survive driver reloads or reboots. A `Type=oneshot` systemd service with `RemainAfterExit=yes` and `After=nvidia-persistenced.service` is the correct pattern — it re-applies settings at every boot in the right order without requiring a long-running process.

## 6. Always verify with a service restart, not just manual application

Manual `nvidia-smi` calls confirm a setting is accepted by the driver. Only a `systemctl restart` of the persistence service confirms the setting survives as part of the automated boot sequence. These are different things and both matter.
