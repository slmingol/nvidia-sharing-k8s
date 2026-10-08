<p align="center">
  <img src="logo.svg" alt="NVIDIA GPU Sharing on Kubernetes" width="860">
</p>

Reference guide and implementation patterns for sharing NVIDIA GPUs efficiently on Kubernetes.

**Live site:** https://slmingol.github.io/nvidia-sharing-k8s/

Covers GPU sharing mechanisms, scheduling, VRAM isolation, hybrid approaches, Triton vs. vLLM, and a tailored recommendation for OCP + Proxmox environments.

## Files

| File | Description |
|---|---|
| `gpu-sharing-k8s.md` | Full reference in Markdown |
| `index.html` | Rendered interactive reference — also served via GitHub Pages |
| `flow.html` | Interactive live simulation — 10 concurrent clients, day/night MIG reconfiguration, QoS preemption |

**[→ Launch interactive simulation](https://slmingol.github.io/nvidia-sharing-k8s/flow.html)**

## Sections

| # | Section |
|---|---|
| TL;DR | Recommended path for OCP + Proxmox + audio/LLM workloads (bare-metal node, ArgoCD GPU Operator, MIG, Triton) |
| 00 | Hardware sharing mechanisms — Time-Slicing, MIG, MPS comparison |
| — | Architecture diagrams — vLLM bare-metal, K8s+GPU Operator, MIG, VM+SR-IOV |
| 01 | Time-Slicing — device plugin replicas, no VRAM isolation caveat |
| 02 | MIG Partitioning — hardware isolation on Ampere+ (A100, H100, A30) |
| 03 | MPS — concurrent CUDA contexts, shared fault domain |
| 04 | Scheduling — bin-packing, GPU Feature Discovery, PriorityClass preemption |
| 05 | Dynamic MIG Reconfiguration — CronJob-driven profile shifts |
| — | Hybrid Approaches — vLLM+MIG, MIG+MPS |
| — | VRAM Isolation Landscape — MIG, MPS pinned limits, AMD NPS4, dmem cgroup, HAMi |
| — | Triton vs. vLLM — when to use each, Triton+MIG RAG pipeline |
| 06 | Recommended Stack — decision guide by workload type |
| 07 | VMs: Boon or Bust? — KubeVirt, SR-IOV, decision matrix |
| 08 | GPU Operator install reference — Helm (generic) + OCP/ArgoCD notes |
| 09 | Next Steps |
| — | References — 14 cited sources |

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
