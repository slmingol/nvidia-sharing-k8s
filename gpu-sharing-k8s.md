# NVIDIA GPU Sharing on Kubernetes

Reference: [Sharing NVIDIA GPUs at the System Level: Time-Sliced and MIG-Backed vGPUs](https://research.colfax-intl.com/sharing-nvidia-gpus-at-the-system-level-time-sliced-and-mig-backed-vgpus/)

## Hardware Sharing Mechanisms

| Mechanism | Isolation | Hardware | Best For |
|---|---|---|---|
| Time-Slicing | None (memory shared) | Any NVIDIA GPU | Dev, low-utilization inference |
| MIG | Hard (dedicated SMs, memory) | Ampere+ (A100, H100, A30) | Production inference, multi-tenant SLA |
| MPS | None (concurrent contexts) | Any NVIDIA GPU | Batch decode, embedding gen, idle-SM workloads |

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

Pods request `nvidia.com/gpu: 1` normally. ~20% ctx-switch overhead. No memory isolation -- one pod OOMs, all pods on that GPU feel it.

---

## Approach 2: MIG Partitioning (Ampere+)

GPU Operator's MIG Manager reconfigures profiles via node label:

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

Hard isolation: dedicated SMs, memory bandwidth, and engines per instance. GPU Operator handles drain/reconfigure/rejoin automatically when label changes -- no manual `nvidia-smi mig` calls.

### Common MIG Profiles (A100-80GB)

| Profile | SMs | Memory | Instances/GPU |
|---|---|---|---|
| `1g.10gb` | 1/7 | 10 GB | 7 |
| `2g.20gb` | 2/7 | 20 GB | 3 |
| `3g.40gb` | 3/7 | 40 GB | 2 |
| `7g.80gb` | 7/7 | 80 GB | 1 |

---

## Approach 3: MPS (Multi-Process Service)

Multiple CUDA contexts run simultaneously (not time-sliced). Lower latency, no ctx-switch overhead.

```yaml
sharing:
  mps:
    replicas: 4
```

Trade-off vs time-slicing: better throughput for concurrent small kernels, same lack of memory isolation.

---

## Approach 4: Scheduling Optimizations

### Bin-Packing (pack GPUs full before using next node)

Default kube-scheduler spreads workloads. Override to pack:

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

### Priority Classes + Preemption

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

High-priority inference pods preempt low-priority training jobs when GPU capacity is needed.

---

## Approach 5: Dynamic MIG Reconfiguration

Shift MIG profiles by time of day to match workload patterns:

```
Day:   all-7g.80gb  → 1 large instance per A100, training jobs
Night: all-1g.10gb  → 7 small instances per A100, batch inference
```

Automate via CronJob patching the node MIG config label. GPU Operator drains the node, reconfigures, rejoins -- no manual intervention.

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

## Recommended Stack

```
NVIDIA GPU Operator          # manages drivers, device plugin, MIG manager, MPS daemon
  └── GPU Feature Discovery  # labels nodes with GPU model, VRAM, MIG capability
  └── DCGM Exporter          # Prometheus metrics: utilization, memory, errors, NVLink

Scheduling:
  kube-scheduler             # bin-packing config for GPU workloads
  Volcano / JobSet           # gang scheduling for distributed training

Sharing strategy by workload:
  Dev / experimentation      → time-slicing (4-8x replicas)
  Production inference       → MIG (A100/H100) or MPS
  Distributed training       → full GPU + gang scheduling
```

---

## K8s Value-Add vs Bare-Metal / Hypervisor vGPU

| Concern | K8s Solution |
|---|---|
| Dynamic MIG reconfiguration | GPU Operator + node label |
| Workload isolation | Namespaces + ResourceQuota per team |
| GPU bin-packing | Scheduler `MostAllocated` strategy |
| Observability | DCGM Exporter + Prometheus/Grafana |
| Autoscaling GPU nodes | Karpenter with GPU instance type constraints |
| Mixed workload priority | PriorityClass + preemption |
| License cost | No NVIDIA vGPU license needed for MIG/time-slicing in pods |

MIG and time-slicing are exposed directly to pods via the device plugin -- no VM hypervisor layer, no vGPU license for most workloads.

---

## VMs: Boon or Bust?

### Mostly bust. Here's the breakdown.

**Where VMs hurt:**

- **Performance overhead** -- hypervisor adds CPU cycles for every CUDA API call routed through the vGPU driver stack. ~20% for time-sliced vGPU, on top of VM memory overhead eating RAM that could be GPU VRAM headroom.
- **License cost** -- NVIDIA vGPU requires per-concurrent-user licensing. Time-slicing and MIG in bare-metal K8s pods: free.
- **Operational complexity** -- hypervisor + VM images + K8s + GPU Operator + vGPU driver versions that must match host exactly. Driver mismatch = silent failures or crashes.
- **Pod startup vs VM startup** -- pods spin up in seconds, VMs in 30-90s. Kills autoscaling responsiveness.

**Where VMs earn their cost:**

- **Hard multi-tenant isolation** -- different orgs, compliance boundaries (HIPAA, FedRAMP), untrusted code. VM gives OS-level separation that K8s namespaces + MIG alone don't provide. Container escape doesn't become tenant escape.
- **Mixed OS requirements** -- Windows workloads needing GPU access (DirectML, DirectX). No other path.
- **Already on a hypervisor** -- cloud GPU nodes (EKS, GKE, AKS) are VMs. GPU passthrough via SR-IOV to those VMs running K8s is the standard cloud path; overhead is already baked in.

### Middle Ground: KubeVirt

Run VMs as K8s workloads with GPU passthrough:

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

Gets: K8s-native lifecycle management + strong VM isolation + MIG hardware partitioning.
Loses: pod startup speed, adds KubeVirt operator complexity.

### Decision Matrix

| Scenario | Recommendation |
|---|---|
| Internal teams, trusted workloads | Bare metal K8s + MIG/time-slicing. Skip VMs. |
| Multi-org or compliance boundary | VMs (KubeVirt or hypervisor) + MIG passthrough |
| Cloud (EKS/GKE/AKS) | Already on VMs -- GPU passthrough node pools, accept it |
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
