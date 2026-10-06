# NVIDIA GPU Sharing on Kubernetes

Reference: [Sharing NVIDIA GPUs at the System Level: Time-Sliced and MIG-Backed vGPUs](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/)

---

## Hardware Sharing Mechanisms

| Mechanism | Isolation | Hardware | Best For |
|---|---|---|---|
| Time-Slicing | None (memory shared) | Any NVIDIA GPU | Dev, low-utilization inference |
| MIG | Hard (dedicated SMs, memory) | Ampere+ (A100, H100, A30) | Production inference, multi-tenant SLA |
| MPS | None (concurrent contexts) | Any NVIDIA GPU | Batch decode, embedding gen, idle-SM workloads |

---

## Architecture Diagrams (described)

Four deployment patterns for sharing GPUs:

**1 — vLLM bare metal.** One vLLM process claims all 8 GPUs with tensor parallelism (TP=8). Every inference request enters a single HTTP endpoint, is continuously batched, and all 8 GPUs collaborate on every request simultaneously over NVLink (600 GB/s all-reduce). No isolation between users; throughput maximized by keeping all SMs busy.

**2 — OCP / K8s with GPU Operator.** The node carries a taint (`nvidia.com/gpu=present:NoSchedule`). GPU Operator deploys the device plugin, which advertises 8 schedulable GPU resources to the API server. Pods declare a toleration and request `nvidia.com/gpu: N`; the device plugin allocates specific GPU device files exclusively to each pod.

**3 — MIG partitioning.** GPU Operator's MIG Manager applies a profile (e.g. `all-1g.10gb`), splitting one A100 into 7 hardware-isolated slices. Each pod requests `nvidia.com/mig-1g.10gb: 1` and gets a dedicated portion of SMs, L2 cache, and HBM.

**4 — VM + SR-IOV.** Each physical GPU exposes a PCIe Physical Function (PF). SR-IOV creates Virtual Functions (VFs); the hypervisor assigns VFs to VMs via IOMMU. After assignment, GPU DMA goes directly between VM memory and GPU — no hypervisor in the data path. Requires NVIDIA vGPU host driver license.

---

## Approach 1: Time-Slicing via NVIDIA Device Plugin

Simplest option. Device plugin advertises N virtual GPUs per physical GPU.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        replicas: 4   # 4 pods share 1 physical GPU
```

Pods request `nvidia.com/gpu: 1` normally. ~20% ctx-switch overhead.

> **Operational reality — no enforcement.** Time-slicing is a scheduler fiction. The device plugin creates N virtual GPU entries, but there are *zero* enforcement mechanisms between them. Any pod can allocate all 80 GB of VRAM and all SMs simultaneously — no budget, no cgroup equivalent. A greedy pod exhausts VRAM and all co-tenants OOM. Reserve time-slicing for trusted dev/test workloads where VRAM consumption per pod is known and bounded.

---

## Approach 2: MIG Partitioning (Ampere+)

GPU Operator's MIG Manager reconfigures profiles via node label. Handles drain/reconfigure/rejoin automatically — no manual `nvidia-smi mig` calls.

```yaml
# Apply to node to trigger reconfiguration
nvidia.com/mig.config: "all-3g.40gb"   # 2x MIG instances on A100-80GB
```

Pods request specific MIG slices:

```yaml
resources:
  limits:
    nvidia.com/mig-3g.40gb: 1
```

### Common MIG Profiles — A100-80GB

| Profile | SMs | Memory | Instances / GPU |
|---|---|---|---|
| `1g.10gb` | 1/7 | 10 GB | 7 |
| `2g.20gb` | 2/7 | 20 GB | 3 |
| `3g.40gb` | 3/7 | 40 GB | 2 |
| `7g.80gb` | 7/7 | 80 GB | 1 |

> **What MIG actually enforces — in hardware.** The A100/H100 firmware partitions the die into independent GPC slices and dedicated HBM regions. Each MIG instance gets a fixed fraction of SMs, L2 cache, and memory bandwidth — all enforced by the GPU's MMU and SM scheduler in silicon. One MIG tenant *cannot* access or exhaust another's VRAM or SMs regardless of what its workload does. This is the only NVIDIA sharing mode with genuine multi-tenant isolation.

---

## Approach 3: MPS — Multi-Process Service

Multiple CUDA contexts run simultaneously (not time-sliced). Lower latency, no ctx-switch overhead. Better throughput for concurrent small kernels; same lack of memory isolation as time-slicing.

```yaml
sharing:
  mps:
    replicas: 4
```

> **Shared memory space — no isolation.** MPS runs all CUDA clients through a single context in a shared address space. A process that over-allocates VRAM starves all other MPS clients on that GPU. If one client segfaults, the MPS server crashes and all co-tenant processes die simultaneously. Use MPS where workloads are trusted and the goal is to eliminate context-switch latency, not to provide resource guardrails.

---

## Approach 4: Scheduling Optimizations

### Bin-Packing

Default kube-scheduler spreads workloads across nodes. Override with `MostAllocated` to pack GPUs full before using next node.

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: MostAllocated
        resources:
        - name: nvidia.com/gpu
          weight: 10
        - name: cpu
          weight: 1
        - name: memory
          weight: 1
```

### GPU Feature Discovery Node Selectors

```yaml
nodeSelector:
  nvidia.com/gpu.product: "A100-SXM4-80GB"
  nvidia.com/gpu.memory: "81920"
  nvidia.com/mig.capable: "true"
```

### Mixed GPU types on one node — assignment is non-deterministic

> **Danger.** The GPU Operator device plugin exposes all GPUs on a node as a fungible pool of `nvidia.com/gpu` units. It does *not* distinguish between individual GPU slots by VRAM size or model. A pod requesting `nvidia.com/gpu: 1` on a mixed node (e.g. 8 GB and 16 GB cards) may land on either card. Node selectors using GFD labels target the *node*, not a specific slot. **Operational rule: GPU nodes must be homogeneous.** Use separate node groups per GPU type and let taints + node selectors steer pods to the right pool.

Three projects solve per-GPU-slot selection within a heterogeneous node:

**1 — Kubernetes DRA + NVIDIA DRA Driver (K8s ≥ 1.34, GA)**

Dynamic Resource Allocation (KEP-3063) replaces count-based device plugin slots with a claim-based API. NVIDIA's DRA driver publishes a `ResourceSlice` per node listing every GPU with structured per-device attributes: product name, memory capacity, architecture, NVLink topology, UUID.

```yaml
spec:
  devices:
    requests:
    - name: gpu
      selectors:
      - cel:
          expression: |
            device.attributes["gpu.nvidia.com"].productName.startsWith("A100") &&
            device.capacity["gpu.nvidia.com"].memory.isGreaterThan(quantity("40Gi"))
```

Requires GPU Operator with DRA mode enabled (opt-in). Hardware-enforced, no shim needed.

**2 — HAMi — CNCF Incubating (K8s ≥ 1.26)**

HAMi intercepts CUDA calls via an injected `libvgpu.so` shim and exposes virtual GPU resources with per-pod VRAM limits. Supports GPU-type filtering within a heterogeneous node via pod annotations, and UUID-level targeting. Covers NVIDIA, AMD, and Ascend. Isolation is software-enforced (CUDA interception), not hardware.

**3 — Custom device plugin (any K8s version, DIY)**

A patched device plugin can advertise `nvidia.com/gpu-a100` and `nvidia.com/gpu-a10` as separate resource types on the same node. Works without DRA or HAMi but requires maintaining a plugin fork.

| Solution | K8s version | Isolation | Status |
|---|---|---|---|
| DRA + NVIDIA DRA driver | ≥ 1.34 | Hardware | GA |
| HAMi | ≥ 1.26 | Software (CUDA shim) | CNCF Incubating |
| Custom device plugin | Any | Hardware | DIY / operator fork |

#### DRA vs. HAMi — when to use which

| | DRA + NVIDIA driver | HAMi |
|---|---|---|
| **What it is** | K8s core API (KEP-3063) | External project — CUDA shim + scheduler extender |
| **Mechanism** | Scheduler assigns specific GPU UUID via `ResourceClaim` | Mutating webhook injects `libvgpu.so` into every container |
| **Per-slot GPU selection** | Yes — CEL on productName, memory, arch, UUID | Yes — type filter + UUID annotation |
| **Sub-GPU VRAM limits** | No — pod gets the whole assigned device | Yes — per-pod VRAM cap enforced at CUDA call level |
| **SM utilization limits** | No | Yes — per-pod SM % cap |
| **Isolation strength** | Hardware — device file binding, no bypass possible | Software — CUDA interception; bypassed if code avoids CUDA |
| **Multi-vendor** | NVIDIA only | NVIDIA, AMD, Ascend, Cambricon, Hygon |
| **K8s version** | ≥ 1.26 alpha → GA ~1.34 | ≥ 1.16 (essentially any) |
| **Overhead** | None | CUDA interception latency (small) |

**Core split:** DRA answers *which GPU slot does this pod get* — hardware binding, no sub-GPU VRAM enforcement beyond MIG. HAMi answers *how much of a GPU slot can this pod consume* — soft quotas, multiple pods per physical device. Practical today: HAMi if K8s < 1.34, multi-vendor, or VRAM quotas needed. DRA if K8s ≥ 1.34 and clean K8s-native semantics matter.

### Priority Classes + Preemption

High-priority inference pods preempt low-priority training jobs when GPU capacity is needed.

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: inference-high
value: 1000
preemptionPolicy: PreemptLowerPriority
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: batch-train-low
value: 100
preemptionPolicy: Never
```

---

## Approach 5: Dynamic MIG Reconfiguration

Shift MIG profiles by time of day to match workload patterns. GPU Operator handles node drain/reconfigure/rejoin — no manual intervention.

| Time | Profile | Instances | Workload |
|---|---|---|---|
| Day | `all-7g.80gb` | 1× full A100 | Distributed training |
| Night | `all-1g.10gb` | 7× small slices | Batch inference |

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mig-reconfigure-night
spec:
  schedule: "0 20 * * *"   # 8pm
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: reconfigure
            image: bitnami/kubectl
            command:
            - kubectl
            - label
            - node
            - <gpu-node>
            - nvidia.com/mig.config=all-1g.10gb
```

---

## Hybrid Approaches

The mechanisms above compose. The most useful production hybrid is **vLLM + MIG**: partition each GPU into hardware-isolated slices, then run an independent vLLM instance per slice. You get K8s scheduling flexibility, MIG's hard resource guarantees, and vLLM's continuous-batching throughput simultaneously.

### vLLM + MIG — per-model isolation at scale

Pick a MIG profile that fits your model's VRAM footprint, schedule one vLLM pod per MIG slice. Each pod sees an isolated GPU device with dedicated SMs and VRAM — noisy-neighbor exhaustion is impossible.

#### Profile sizing guide (A100-80GB)

| Model size (FP16) | MIG profile | Instances per GPU | Deployments on 8× A100 |
|---|---|---|---|
| Embedding / <3B | `1g.10gb` | 7 | 56 |
| 7B (Mistral, LLaMA-3-8B) | `2g.20gb` | 3 | 24 |
| 13B | `3g.40gb` | 2 | 16 |
| 70B (needs all 80 GB) | `7g.80gb` | 1 | 8 (no MIG benefit) |
| 70B+ (multi-GPU TP) | whole GPUs, no MIG | — | depends on TP width |

Diagram: each A100 splits into 2× `3g.40gb` slices = 16 isolated vLLM deployments across an 8-GPU host. Each vLLM pod runs TP=1 (one MIG slice only) and gets guaranteed 40 GB VRAM + 3/7 of the SMs. Different models share the same physical node with zero cross-model interference.

> **NVLink boundary — TP=1 limit.** MIG instances cannot communicate over NVLink. Each vLLM runs `tensor_parallel_size=1`. A model that requires tensor parallelism across multiple GPUs — typically 70B+ at FP16 — cannot be served by a MIG-sliced pod. Use whole GPUs with `tensor_parallel_size=N` for those models. Both modes can run on different node pools in the same cluster.

### MIG + MPS within a slice

Once a MIG slice is assigned to a pod, that pod can itself start an MPS daemon to multiplex multiple CUDA processes *within* the slice — running concurrent kernels without serialization, without surrendering the slice's hard VRAM floor. This stacks two levels: hardware boundary between slices, MPS concurrency within a slice.

> **Isolation hierarchy.** MIG enforces the outer boundary (slice cannot be exceeded by any workload). MPS adds concurrent execution within that boundary. The MIG guarantee holds regardless of MPS behavior: a runaway MPS client cannot consume another slice's VRAM.

---

## VRAM Isolation Landscape

Not all GPU sharing approaches provide VRAM isolation. Without it, a greedy or buggy workload can exhaust all GPU memory and take down every co-tenant on that device.

| Approach | VRAM isolation | Strength | Hardware requirement |
|---|---|---|---|
| MIG | Hardware | Hard — GPU MMU enforces in silicon | NVIDIA Ampere+ (A100, H100, A30) |
| AMD NPS4 (MI300X) | Hardware | Hard — HBM physically partitioned | AMD MI300X / MI325X only |
| SR-IOV / VM passthrough | Hardware (IOMMU) | Hard — VMs only, not pods | Any GPU with SR-IOV + hypervisor |
| K8s exclusive GPU per pod | Hardware | Hard — pod owns entire device | Any GPU; not sub-GPU sharing |
| DRA (whole GPU) | Hardware | Hard — specific device assigned to pod | Any GPU; K8s ≥ 1.34; not sub-GPU splitting |
| NVIDIA MPS pinned limits | Daemon (userspace) | Soft-medium — MPS server enforces caps; shared fault domain | Volta+, CUDA 11.5+, NVIDIA only |
| HAMi | CUDA shim | Soft — bypassed if workload avoids CUDA calls | NVIDIA, AMD, Ascend (any K8s ≥ 1.16) |
| `dmem` cgroup controller | Kernel | Hard — but no K8s/runc wiring yet | Intel xe (Linux ≥ 6.14), AMD (≥ 6.15); no NVIDIA proprietary |
| Time-slicing | None | — | — |
| MPS (no limits configured) | None | — | — |
| Triton concurrent models | None | — | — |

### Non-MIG options in detail

#### NVIDIA MPS pinned memory limits (Volta+)

Set per-client VRAM caps inside the MPS daemon. Already wired into the k8s-device-plugin MPS sharing mode (splits evenly across replicas) and the NVIDIA DRA driver (`MpsConfig.DefaultPinnedDeviceMemoryLimit`). GKE supports it natively.

```bash
# per-client cap set in MPS daemon
echo "set_default_device_pinned_mem_limit 0 4096m" | nvidia-cuda-mps-control

# or via environment variable before launching client
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=4096m" python inference.py
```

> **Shared fault domain.** MPS clients share a CUDA context. A fatal fault (illegal memory access, driver crash) in one client kills all co-located MPS clients simultaneously. The VRAM cap is enforced by the MPS daemon process — if it crashes, caps are gone. Not equivalent to MIG's hardware boundary.

#### AMD MI300X — NPS4 memory partitioning

The MI300X and MI325X support Non-Uniform Memory Access Partitioning (NPS4), which physically splits the 192 GB HBM stack into 4 × 48 GB regions. Combined with CPX (Compute Partitioning), up to 64 hardware partitions per 8-GPU node are possible. The AMD GPU Operator's Device Config Manager applies profiles declaratively; supported on OpenShift. This is the AMD hardware equivalent of MIG, limited to MI300-class GPUs.

> **Production-ready on OCP.** AMD GPU Operator + Device Config Manager handles partition setup and exposes slices as schedulable K8s resources. Consumer and Radeon cards have no equivalent. AMD MxGPU SR-IOV exists but is VM-only.

#### `dmem` cgroup controller (Linux ≥ 6.14)

A kernel-level VRAM cgroup with min/low/max semantics identical to the memory cgroup. Merged for Intel `xe` in Linux 6.14; AMD `amdgpu` support targeting 6.15/6.16; nouveau patch on mailing list. Does **not** work with the NVIDIA proprietary driver. No runc, containerd, or K8s wiring exists yet — cgroup files must be set manually via an NRI plugin or privileged DaemonSet. Known gaps: `hipMallocAsync` pools spill uncapped into system RAM; lowering `dmem.max` does not reclaim already-allocated memory. HAMi is evaluating `dmem` as its AMD backend.

> **Experimental — no K8s integration.** Not usable in production today without significant operator work. Watch HAMi and the amdgpu driver for when this lands as a first-class K8s resource.

---

## Inference Server Layer — Triton vs. vLLM

Triton Inference Server and vLLM sit at the *application layer*, above K8s GPU sharing. They don't replace MIG, device plugin, or DRA — they run inside whatever GPU resource(s) K8s assigns to their pod.

```
Hardware: 8× A100
  ↓ K8s GPU sharing layer  # MIG / time-slicing / DRA / device plugin
  ↓ pod gets GPU resource(s)
  ↓ Application / inference server layer  # Triton and vLLM live here
    ↓ Triton's own internal sharing  # instance_group, concurrent execution, dynamic batching
```

### What Triton adds at the application layer

- **`instance_group`** — load N copies of a model spread across specific GPU indices the pod owns.
- **Concurrent model execution** — multiple different models share one GPU simultaneously through Triton's own scheduler. No VRAM isolation between them (same noisy-neighbor exposure as MPS).
- **Dynamic batching** — coalesces requests before GPU invocation; complements K8s-level sharing.
- **Rate limiter** — caps simultaneous requests per model, prevents one model starving others on the same GPU.
- **Multi-backend + ensembles** — TensorRT, PyTorch, ONNX, TF, OpenVINO, Python, and a vLLM backend. One Triton instance can serve a CNN, reranker, and LLM together in a single ensemble pipeline.

### Triton vs. vLLM

| | vLLM | Triton |
|---|---|---|
| **Target** | LLMs only | Any model type |
| **LLM batching** | Continuous batching + PagedAttention — best-in-class LLM throughput | Dynamic batching — good, but LLM-specific optimizations weaker without TRT-LLM backend |
| **LLM backend** | Built-in | TRT-LLM backend or vLLM backend |
| **Multi-model serving** | No — one model per instance | Yes — model repository, ensembles, multiple models concurrently |
| **RAG pipelines** | No | Yes — ensemble: embed → rerank → LLM in one server |
| **Tensor parallelism** | Native, first-class | Via TRT-LLM backend |
| **K8s integration** | Direct pod | Direct pod, or via KServe |

### Triton + MIG — same pattern as vLLM + MIG

One Triton pod per MIG slice, each slice running different models, isolated at hardware level. For mixed-model workloads this beats running separate vLLM instances:

| MIG slice | Triton model(s) | Use |
|---|---|---|
| `1g.10gb` | embedding model | RAG retrieval |
| `1g.10gb` | reranker | RAG reranking |
| `3g.40gb` | LLM (TRT-LLM or vLLM backend) | generation |
| `2g.20gb` | vision / classification | multimodal pre-processing |

Each slice is hardware-isolated — the embedding model cannot OOM the LLM slice. Triton's ensemble API wires the pipeline together over gRPC without leaving the node.

> **Triton does not solve K8s-level isolation.** Concurrent model execution within a Triton instance has no VRAM isolation between models. A misbehaving model inside Triton can OOM the entire server and kill all co-located models — same noisy-neighbor problem as MPS. For hard per-tenant isolation, MIG is still required at the K8s layer beneath Triton.

---

## Recommended Stack

```
NVIDIA GPU Operator          # manages drivers, device plugin, MIG manager, MPS daemon
  └── GPU Feature Discovery  # labels nodes with GPU model, VRAM, MIG capability
  └── DCGM Exporter          # Prometheus metrics: utilization, memory, errors, NVLink

Scheduling:
  kube-scheduler             # bin-packing config for GPU workloads
  Volcano / JobSet           # gang scheduling for distributed training

Sharing strategy by workload:
  Dev / experimentation      → time-slicing (4–8× replicas, trusted workloads only)
  Production inference       → MIG (A100/H100) — hardware-isolated, no noisy-neighbor
  Multiple models, one host  → vLLM + MIG (one vLLM pod per MIG slice)
  Latency-sensitive, trusted → MPS (concurrent kernels, no ctx-switch)
  Distributed training       → full GPU + gang scheduling (no sharing)
  70B+ inference             → whole GPUs, TP=8, no MIG
```

### K8s Value-Add vs Bare-Metal / Hypervisor vGPU

| Concern | K8s Solution |
|---|---|
| Dynamic MIG reconfiguration | GPU Operator + node label |
| Workload isolation | Namespaces + ResourceQuota per team |
| GPU bin-packing | Scheduler `MostAllocated` strategy |
| Observability | DCGM Exporter + Prometheus/Grafana |
| Autoscaling GPU nodes | Karpenter with GPU instance type constraints |
| Mixed workload priority | PriorityClass + preemption |
| License cost | No NVIDIA vGPU license needed for MIG/time-slicing in pods |

---

## VMs: Boon or Bust?

Mostly bust. The VM layer adds overhead and cost that K8s + MIG already solves for most workloads.

**Where VMs hurt:**

- **Performance** — ~20% overhead per CUDA API call through vGPU driver stack; VM memory eats RAM that could be VRAM headroom
- **License cost** — NVIDIA vGPU requires per-concurrent-user licensing; MIG + time-slicing in pods is free
- **Complexity** — hypervisor + VM images + K8s + GPU Operator + vGPU driver versions must match host exactly
- **Startup latency** — pods spin up in seconds; VMs take 30–90s, killing autoscaling responsiveness

**Where VMs earn their cost:**

- **Hard multi-tenant isolation** — different orgs, compliance (HIPAA, FedRAMP), untrusted code; container escape doesn't become tenant escape
- **Mixed OS** — Windows workloads needing GPU access (DirectML, DirectX) — no other path
- **Already on hypervisor** — EKS, GKE, AKS GPU nodes are VMs; overhead is baked in

### Middle Ground: KubeVirt

Run VMs as K8s workloads with GPU passthrough. Gets K8s-native lifecycle management + strong VM isolation + MIG hardware partitioning. Loses pod startup speed; adds KubeVirt operator complexity.

```yaml
apiVersion: kubevirt.io/v1
kind: VirtualMachine
spec:
  template:
    spec:
      domain:
        devices:
          gpus:
          - deviceName: nvidia.com/mig-3g.40gb
            name: gpu0
```

### Decision Matrix

| Scenario | Recommendation |
|---|---|
| Internal teams, trusted workloads | Bare metal K8s + MIG/time-slicing. Skip VMs. |
| Multi-org or compliance boundary | VMs (KubeVirt or hypervisor) + MIG passthrough |
| Cloud (EKS/GKE/AKS) | Already on VMs — GPU passthrough node pools, accept it |
| Windows GPU workloads | VMs required, no alternative |
| Mixed Linux multi-tenant inference | MIG + ResourceQuota sufficient, no VMs needed |

**Default**: bare metal K8s nodes with GPU Operator. VMs only when compliance or OS requirements force it.

---

## GPU Operator Helm Install (baseline)

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set mig.strategy=mixed \
  --set devicePlugin.config.name=time-slicing-config
```

---

## References

### Primary source

- [Sharing NVIDIA GPUs at the System Level: Time-Sliced and MIG-Backed vGPUs](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/) — Colfax Research. Basis of this guide.

### NVIDIA official documentation

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) — GPU Operator overview, installation, MIG manager, MPS daemon configuration.
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — MIG architecture, profiles, partitioning, and instance management on A100/H100/A30.
- [NVIDIA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) — MPS architecture, client/server model, use cases.
- [MPS environment variables](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html) — `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` and related per-client VRAM cap configuration.
- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) — model repository, instance groups, dynamic batching, ensemble pipelines, backend support.
- [GPU Operator + KubeVirt](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-kubevirt.html) — vGPU manager integration; VMs on K8s; DRA-mode mixed container/VM nodes.

### Kubernetes GPU projects

- [HAMi — Heterogeneous AI Computing Virtualization Middleware](https://github.com/Project-HAMi/HAMi) (CNCF Incubating) — CUDA shim, per-pod VRAM limits, GPU-type filtering, multi-vendor support.
- [NVIDIA DRA Driver API reference](https://pkg.go.dev/github.com/NVIDIA/k8s-dra-driver-gpu/api/nvidia.com/resource/v1beta1) — `ResourceClaim` structured attributes; `MpsConfig.DefaultPinnedDeviceMemoryLimit`.
- [k8s-device-plugin issue #764](https://github.com/NVIDIA/k8s-device-plugin/issues/764) — MPS pinned memory limits: `cudaMemGetInfo` still reports full card; tracking issue.
- [Intel GPU Plugin for Kubernetes](https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/gpu_plugin/README.html) — `shared-dev-num`, SR-IOV VF support, Intel dGPU scheduling.
- [HAMi AMD core — dmem cgroup evaluation](https://github.com/Project-HAMi/amd-hami-core/issues/6) — tracking issue for using the Linux `dmem` cgroup as the AMD VRAM enforcement backend.

### AMD

- [AMD ROCm — Compute and Memory Partition Modes](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html) — SPX/CPX compute partitioning; NPS1/NPS4 memory partitioning on MI300X/MI325X.
- [AMD GPU Operator — Applying Partition Profiles](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/dcm/applying-partition-profiles.html) — Device Config Manager; NPS4 setup on OpenShift.

### Kernel / hardware

- [dmem cgroup controller — VRAM control in Linux](https://www.phoronix.com/news/DMEM-cgroup-vRAM-Control) — Phoronix coverage of the kernel-level GPU memory cgroup merged in Linux 6.14 for Intel xe.
- [Linux 6.15 DRM fixes](https://www.phoronix.com/news/Linux-6.15-rc2-DRM-Fixes) — amdgpu dmem cgroup support timeline.
