# nvidia-sharing-k8s

Reference guide and implementation patterns for sharing NVIDIA GPUs efficiently on Kubernetes.

Covers time-slicing, MIG partitioning, MPS, scheduling optimizations, dynamic MIG reconfiguration, and a VM trade-off analysis.

## Files

| File | Description |
|---|---|
| `gpu-sharing-k8s.md` | Full reference in Markdown |
| `index.html` | Rendered interactive reference — also served via GitHub Pages |

## Approaches Covered

- **Time-Slicing** — NVIDIA device plugin replicas; any GPU; no memory isolation
- **MIG Partitioning** — hardware-level isolation on Ampere+ (A100, H100, A30); dedicated SMs and memory per slice
- **MPS** — concurrent CUDA contexts; lower latency than time-slicing; no isolation
- **Scheduling** — bin-packing via `MostAllocated`, GPU Feature Discovery node selectors, PriorityClass preemption
- **Dynamic MIG Reconfiguration** — CronJob-driven profile shifts (e.g. training by day, inference by night)
- **VM Trade-offs** — when VMs (KubeVirt) help vs. when bare-metal K8s is sufficient

## Quick Start

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set mig.strategy=mixed \
  --set devicePlugin.config.name=time-slicing-config
```

## Source

Based on: [Sharing NVIDIA GPUs at the System Level: Time-Sliced and MIG-Backed vGPUs](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/)
