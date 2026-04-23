
# Day 6: GPU & Datacenter Security

Today's content is a lot more theory than labs, so you should make a file called w2d4\_answers.md and write your solution to each exercise there. Ask a TA to verify after you've completed each exercise.

## Table of Contents

- [Exercise 1: Nvidia container toolkit exploit](#exercise-1-nvidia-container-toolkit-exploit)
- [Exercise 2.1: GPU Security Issues (Leaky DNN)](#exercise-21-gpu-security-issues-leaky-dnn)
- [Exercise 2.2: GPU Multi-Tenancy Issues](#exercise-22-gpu-multi-tenancy-issues)
- [Exercise 3.1: GPU Interconnect Architecture](#exercise-31-gpu-interconnect-architecture)
- [Exercise 3.2: Kubernetes and Slurm](#exercise-32-kubernetes-and-slurm)
- [Exercise 3.3 (optional): Designing your own ML workstation](#exercise-33-optional-designing-your-own-ml-workstation)
- [Exercise 4: Confidential Computing](#exercise-4-confidential-computing)
- [Exercise 5: What’s in a data center?](#exercise-5-what’s-in-a-data-center)
- [Exercise 6: Securing a Data Center](#exercise-6-securing-a-data-center)
- [Exercise 99a: Super technical track](#exercise-99a-super-technical-track)
- [Exercise 99b: Even more technical track](#exercise-99b-even-more-technical-track)

## Exercise 1: Nvidia container toolkit exploit

This is a hacking exercise. We will be replicating the following exploit of CVE-2025-23266: [https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape)

You have a ready-made machine to log into.

**Login Info:**

<!-- FIXME: HOW MACHINE IS AVAILABLE ? -->

TO BE DISCOSED BY THE TAs TO PARTICIPANTS

**Instructions:**

* Please copy \~/test into a new folder with your pair's name (eg, `cp -r test w2d4-pranav`) and ignore \~/solution unless stuck
* You have sudo access, but the exploit should run as the regular user
* A vulnerable version of the nvidia stack is already installed, please do not install any nvidia packages
* Write a POC file in C (in poc.c), and use the Makefile to build it
  * For example: update the file to write your username into the file /tmp/output-pranav
  * **PLEASE USE A UNIQUE OUTPUT FILENAME FOR YOUR TEAM** to avoid confusing others
* Write a Dockerfile, and use the Makefile to build it
* Try the exploit with `docker run --rm --runtime=nvidia --gpus=all nvidia-ctk-image`

You will exploit nvidia-ctk on the host, becoming root.

## Exercise 2.1: GPU Security Issues (Leaky DNN)

**Background:** GPUs are increasingly used in containerized environments and virtual machines for AI workloads, but this introduces unique security considerations beyond traditional CPU-based computing.

**Reading Materials:**

* Paper slides: [https://yichez.site/papers/2020/dsn2020slides.pdf](https://yichez.site/papers/2020/dsn2020slides.pdf)
* Video presentation: [https://youtu.be/M\_F7khOLZ7A](https://youtu.be/M_F7khOLZ7A)

**Your Task:** Analyze the security implications of GPU usage in virtualized environments by addressing:

* **Container GPU Issues**: What security vulnerabilities might arise when GPUs are shared between containers or when containers have direct GPU access?
* **VM GPU Passthrough**: What attack vectors are introduced when GPUs are passed through directly to virtual machines?
* **Isolation Concerns**: How might GPU memory persistence, side-channel attacks, or privilege escalation differ between containerized and VM environments?

**Deliverable:** Write down two specific security risks for each deployment model (containers vs. VMs) and suggest a mitigation strategy for each risk.
<details>
<summary>Claude's solution</summary><blockquote>

Container GPU Security Risks
----------------------------

Risk 1: GPU Memory Persistence and Data Leakage

**Issue**: When containers share GPU resources, GPU memory isn't automatically cleared between container executions. Residual data from previous workloads can persist in GPU memory (including global memory, shared memory, and registers), potentially exposing sensitive information like model weights, intermediate computations, or user data to subsequent containers.

**Mitigation Strategy**:

-   Implement GPU memory sanitization policies that force memory clearing between container context switches
-   Use NVIDIA Multi-Instance GPU (MIG) technology to create isolated GPU partitions with hardware-enforced memory boundaries
-   Deploy container runtime hooks that execute `cudaMemset()` or equivalent operations to zero out GPU memory before container termination

Risk 2: Kernel Driver Exploitation and Container Escape

**Issue**: Containers typically share the host kernel and GPU drivers. A vulnerability in GPU drivers (NVIDIA, AMD, Intel) could allow malicious containers to escape containment through driver exploitation, potentially gaining host-level privileges since GPU drivers often run with elevated permissions.

**Mitigation Strategy**:

-   Use gVisor or Kata Containers to provide an additional isolation layer between containers and host GPU drivers
-   Implement strict seccomp profiles that limit GPU-related system calls to the minimum required set
-   Regular driver updates and consider using LTS driver branches with security backports
-   Deploy GPU device plugins with restricted device permissions and capability dropping

VM GPU Passthrough Security Risks
---------------------------------

Risk 1: DMA Attack Vectors via IOMMU Bypass

**Issue**: GPU passthrough typically grants VMs direct memory access (DMA) capabilities. If IOMMU configuration is incorrect or if the GPU firmware is compromised, malicious VMs could potentially read/write arbitrary host memory locations, bypassing hypervisor protections and accessing other VMs' memory spaces.

**Mitigation Strategy**:

-   Ensure proper IOMMU grouping and enforce strict VFIO isolation policies
-   Enable and verify Intel VT-d or AMD-Vi with interrupt remapping enabled
-   Implement PCI Access Control Services (ACS) to prevent peer-to-peer DMA between devices
-   Use SR-IOV where possible instead of full device passthrough to limit the attack surface

Risk 2: GPU Firmware Persistence Attacks

**Issue**: When GPUs are passed through to VMs, malicious guests can potentially flash compromised firmware to the GPU's EEPROM/NVRAM. This persistent firmware modification survives VM destruction and host reboots, potentially compromising future VMs that use the same GPU.

**Mitigation Strategy**:

-   Implement GPU firmware attestation and verification before VM assignment
-   Use vendor-specific protections like NVIDIA GPU firmware signing and secure boot
-   Deploy physical GPU reset mechanisms between VM assignments (power cycling via BMC)
-   Maintain separate GPU pools for different trust levels and implement GPU quarantine procedures for suspected compromised devices

Key Isolation Differences
-------------------------

**Container Environments** tend to have:

-   Faster GPU context switching but weaker isolation boundaries
-   Shared kernel attack surface
-   More susceptible to timing-based side-channel attacks due to fine-grained GPU sharing

**VM Environments** typically provide:

-   Stronger isolation through hardware virtualization
-   Higher overhead but better protection against GPU memory side-channels
-   More resilient against kernel exploits but vulnerable to lower-level hardware attacks
</blockquote></details>
<details>
<summary>A longer list of risks</summary><blockquote>

Container-Specific Risks

-   **Memory persistence**: Leftover data in GPU memory between container runs
-   **Driver exploitation**: Shared kernel drivers enabling container escape
-   **Resource exhaustion**: One container monopolizing GPU resources (DoS)
-   **Side-channel timing**: Fine-grained GPU sharing enabling timing attacks
-   **Namespace bypass**: GPU devices bypassing Linux namespace isolation
-   **Library injection**: Shared CUDA/ROCm libraries being tampered
-   **Device node access**: Direct `/dev/dri` or `/dev/nvidia*` access bypassing controls

VM-Specific Risks

-   **DMA attacks**: Direct memory access to host via misconfigured IOMMU
-   **Firmware persistence**: Malicious firmware flashing surviving reboots
-   **PCI passthrough leaks**: Information disclosure via PCI configuration space
-   **GPU reset failures**: Incomplete GPU state clearing between VM assignments
-   **MMIO exploitation**: Memory-mapped I/O region manipulation
-   **Interrupt injection**: Malicious interrupt generation affecting host
-   **SR-IOV vulnerabilities**: Virtual function isolation bypass
-   **Power management attacks**: GPU power state manipulation affecting other VMs

Common Risks (Both Environments)

-   **Rowhammer-style attacks**: GPU-accelerated bit flipping in adjacent memory
-   **Cache timing attacks**: GPU cache-based side channels
-   **Performance counters**: Information leakage through GPU metrics
-   **Thermal monitoring**: Temperature-based side channels
-   **Memory allocation patterns**: Inferring workloads from allocation behavior
-   **Speculative execution**: GPU-specific speculative vulnerabilities
-   **Cross-VM/container correlation**: Power analysis attacks
-   **GPU interconnect**: NVLink/Infinity Fabric exploitation
</blockquote></details>

## Exercise 2.2: GPU Multi-Tenancy Issues

**Background:** GPUs may be used by multiple parties, which leads to more issues.

**Reading Materials:**

* Multi-tenancy: [https://kubernetes.io/docs/concepts/security/multi-tenancy/](https://kubernetes.io/docs/concepts/security/multi-tenancy/)
* Special hardware support for GPU partitioning: [https://www.scaleway.com/en/docs/gpu/how-to/use-nvidia-mig-technology/](https://www.scaleway.com/en/docs/gpu/how-to/use-nvidia-mig-technology/)

**Your Task:** After reviewing the materials above, analyze advanced GPU security considerations by addressing:

* **Software vs. Hardware Isolation**: How does Kubernetes namespace-based multi-tenancy compare to NVIDIA MIG's hardware partitioning for GPU security?
* **Attack Surface Analysis**: What new vulnerabilities emerge when multiple tenants share GPU resources, and how do different isolation approaches mitigate these risks?
* **Resource Guarantees**: How do the quality-of-service guarantees provided by MIG differ from traditional container resource limits for GPU workloads?

**Deliverable:** Compare the security trade-offs between software-only multi-tenancy and hardware-assisted GPU partitioning, identifying which approach would be more suitable for a high-security AI training environment and why.

## Exercise 3.1: GPU Interconnect Architecture

**Background:** ML training/inference requires GPUs to communicate with the host system and with each other to optimize performance and store large models.

**Reading Materials:**

* PCIe lanes: [https://www.cgdirector.com/guide-to-pcie-lanes/](https://www.cgdirector.com/guide-to-pcie-lanes/)
* NVLink: [https://www.fibermall.com/blog/nvidia-nvlink.htm](https://www.fibermall.com/blog/nvidia-nvlink.htm)

**Your Task:** After reading the materials above, explain how PCIe and NVLink enable GPU computing by answering:

* **PCIe Lanes**: How do PCIe lanes affect GPU performance? What happens when a GPU has fewer lanes than expected?
* **NVLink Benefits**: Why use NVLink instead of PCIe for GPU-to-GPU communication? What performance advantages does it provide?
* **System Architecture**: In a multi-GPU training setup, when would data flow through PCIe vs. NVLink?

**Deliverable:** Draw a diagram describing how data flows between CPU, system memory, and 4 GPUs connected via both PCIe and NVLink, labeling the bandwidth differences.

**Stretch:** How does this extend to multi-node clusters? How does infiniband fit in?

## Exercise 3.2: Kubernetes and Slurm

Kubernetes and Slurm are the most popular ML cluster management software suites.

Read [https://nebius.com/blog/posts/model-pre-training/slurm-vs-k8s](https://nebius.com/blog/posts/model-pre-training/slurm-vs-k8s)

**Example scenario:** NeuralFlow serves 15 specialized AI models to 10+ enterprise clients with strict SLA requirements. They handle 10,000+ daily inference requests while performing continuous online learning from client feedback. Recent issues include service outages during peak hours, unpredictable latency when batch learning jobs interfere with real-time inference, and resource conflicts between different client workloads requiring data isolation.

**Your Task:** Recommend Kubernetes or Slurm for NeuralFlow's mixed real-time inference and online learning workloads.

**Deliverable:** Think about and describe which software would be most appropriate.

## Exercise 3.3 (optional): Designing your own ML workstation

Haven’t you always wanted a sweet ML server in your apartment? Now’s your chance to plan one out. Reading: [https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)

Here are some examples:

- [https://pcpartpicker.com/user/vacuo/saved/t8CnGX](https://pcpartpicker.com/user/vacuo/saved/t8CnGX)
- [https://pcpartpicker.com/user/vacuo/saved/9gkGLk](https://pcpartpicker.com/user/vacuo/saved/9gkGLk)

Make yours here, with (say) a 3000 GBP budget: [https://pcpartpicker.com/](https://pcpartpicker.com/). What are the potential bottlenecks, and how would you address them?

## Exercise 4: Confidential Computing

**Background:** Confidential computing protects data during processing by combining memory encryption with attestation mechanisms, enabling secure computation on untrusted infrastructure.

**Reading Materials:**

* Video overview: [https://www.youtube.com/watch?v=cbzyGCO6y-s](https://www.youtube.com/watch?v=cbzyGCO6y-s)
* Google Cloud implementation: [https://cloud.google.com/blog/products/identity-security/confidential-computing-data-encryption-during-processing/](https://cloud.google.com/blog/products/identity-security/confidential-computing-data-encryption-during-processing/)
* Red Hat platform details: [https://www.redhat.com/en/blog/confidential-computing-platform-specific-details](https://www.redhat.com/en/blog/confidential-computing-platform-specific-details)

**Your Task:** After reviewing the materials above, analyze confidential computing by addressing:

* **Memory Encryption Trade-offs**: Given the \~15% performance overhead on GCP, when would the security benefits justify this cost for AI workloads?
* **Attestation Process**: How do Trusted Execution Environments (TEEs) verify that code is running in a secure environment before processing sensitive data?
* **Use Cases**: What types of AI/ML scenarios would most benefit from confidential computing versus traditional security measures?

**Deliverable:** Design a brief scenario where confidential computing would be essential for an AI application, explaining which component (memory encryption, attestation, or both) addresses the primary threat and why standard encryption-at-rest wouldn't suffice.

## Exercise 5: What’s in a data center?

**Background:** Building large-scale AI infrastructure requires understanding different deployment models, from cloud-based solutions to on-premises supercomputers.

**Reading Materials:**

* Lambda Labs one-click: [https://finance.yahoo.com/news/lambda-announces-general-availability-multi-200000916.html?guccounter=1](https://finance.yahoo.com/news/lambda-announces-general-availability-multi-200000916.html?guccounter=1)
* Nvidia DGX (optional): [https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)
* Detailed analysis of 100,000 H100 clusters (optional): [https://semianalysis.com/2024/06/17/100000-h100-clusters-power-network/](https://semianalysis.com/2024/06/17/100000-h100-clusters-power-network/)
* Read section 3.3.1, 3.3.4 and 3.3.4 of the llama 3 paper: https://arxiv.org/pdf/2407.21783 and (optinally) section 3 of the deepseek v3 report: https://arxiv.org/pdf/2412.19437

**Your Task:** After reviewing the materials above, compare different approaches to AI infrastructure by analyzing:

* **Cloud vs. On-Premises**: What are the trade-offs between Lambda's cloud approach and building your own data center?
* **System Architecture**: How does the DGX B200's design address the networking and power challenges described in the H100 cluster analysis?
* **Scale Considerations**: What infrastructure challenges emerge when scaling from a single DGX system to a 100,000 GPU cluster?

**Deliverable:** Design a brief proposal for an on-premises 512 GPU training cluster, specifying what hardware you’d use including number of nodes, and power consumption. You can use datasheets (like https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf) and consult documentation to figure out the numbers. Ask a TA if you need help finding some documentation.

<details>
<summary>Solution: Hardware Level 1 (basic)</summary><blockquote>

Components

-   64x compute nodes, each with 256GB memory, 2TB disk, 50+ CPU cores, 8x H100's
-   1x storage node, 100TB disk
-   2x head nodes
-   Infiniband networking

How many GPUs can you put in each node? Why?
What is the purpose of Infiniband networking?
Why use a separate storage node?
</blockquote></details>

<details>
<summary>Solution: Hardware Level 2 (advanced)</summary><blockquote>

Components
-   64x compute nodes
-   2x Intel Xeon or AMD Epyc processors with 50+ cores
-   256GB memory, quad- or six-channel
-   2TB disk as 2x 2TB in RAID 1
-   8x H100's with ConnectX
-   1x storage node
-   256GB memory, quad- or six-channel
-   100TB disk, many 8TB drives in RAID6 or RAID10
-   2x head nodes
-   256GB memory, quad- or six-channel
-   2TB disk as 2x 2TB in RAID 1
-   Infiniband networking (400Gb/s+)
-   Dedicated 100Gb/s internet link (possibly dual link for redundancy)
-   Hardware firewall

Example: <https://bizon-tech.com/bizon-g9000.html#4654:46133;4658:46420>
</blockquote></details>


## Exercise 6: Securing a Data Center

**Reference Material:** Use your proposal from the previous exercise, or look at the solution, as the basis for this exercise.

**Your Task:** Based on your data center design, analyze its security implications by addressing:

* Physical Security
* Network Attack Surface
* Operational Security
* Anything else you can think of

**Deliverable:** Create a security threat model, identifying the top 5 unique security risks and proposing specific mitigation strategies for each.

## Exercise 99a: Super technical track

Replicate this:
- [https://docs.nvidia.com/attestation/overview-attestation-suite/latest/introduction.html](https://docs.nvidia.com/attestation/overview-attestation-suite/latest/introduction.html)

## Exercise 99b: Even more technical track
- set up confidential computing: https://docs.nvidia.com/cc-deployment-guide-tdx.pdf

May be suitable for a course project next week\!

