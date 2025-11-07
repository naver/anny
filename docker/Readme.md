# 🧠 Anny — Dockerized Human Body Mesh Model

**Anny** is a differentiable human body mesh model written in **PyTorch**. It models a broad spectrum of human body shapes—from infants to elders—using a shared topology and parameter space.

This Docker setup provides a clean, production-ready environment for running **Anny** locally, supporting both **CPU-only** and **GPU (CUDA + Warp)** builds.

---

## 🚀 Overview

| Build Target       | GPU Support   | Base Image                               | Approx. Size | Recommended For                          |
| :----------------- | :------------ | :--------------------------------------- | :----------- | :--------------------------------------- |
| **`anny-minimal`** | ❌ CPU-only    | `python:3.11-slim`                       | ~1.8 GB      | Systems without NVIDIA GPU               |
| **`anny-full`**    | ✅ CUDA + Warp | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | ~2.3 GB      | Workstations or servers with NVIDIA GPUs |

---

## 🧩 Prerequisites

### For All Builds

* **Docker** ≥ 24.0
* **Docker Compose** v2.3 or newer
* Internet access for image and dependency downloads

### For GPU (Full) Build

* **NVIDIA GPU** with driver ≥ 525
* **NVIDIA Container Toolkit** installed
* Test GPU access:

  ```bash
  nvidia-smi
  ```

If this shows your GPU and driver info, you’re ready to build the GPU version.

---

## ⚙️ Build Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/naver/anny.git
cd anny
```

### 2. Enable BuildKit for faster builds

```bash
export DOCKER_BUILDKIT=1
```

### 3. Build the Docker Images

**CPU-only (minimal)**:

```bash
docker compose build anny-minimal
```

**GPU-enabled (full)**:

```bash
docker compose build anny-full
```

---

## ▶️ Run the Demo

### CPU-only Demo

```bash
docker compose up anny-minimal
```

Access the demo at: [http://localhost:7860](http://localhost:7860)

### GPU + Warp Demo

```bash
docker compose up anny-full
```

Access the demo at: [http://localhost:7860](http://localhost:7860)

> ⚠️ Ensure NVIDIA runtime is properly configured (`docker run --gpus all ...` should work).

---

## 🧠 Validation

Once the container is running, validate Torch and Warp inside:

**Check PyTorch GPU availability:**

```bash
docker exec -it anny-full python3 -c "import torch; print(torch.__version__, 'CUDA?', torch.cuda.is_available())"
```

**Expected output:**

```
2.4.1 CUDA? True
```

**Check Warp installation:**

```bash
docker exec -it anny-full python3 -c "import warp; print('Warp OK:', warp.__version__)"
```

---

## 🧾 Helpful Commands

| Action          | Command                             |
| :-------------- | :---------------------------------- |
| Build CPU image | `docker compose build anny-minimal` |
| Build GPU image | `docker compose build anny-full`    |
| Run CPU demo    | `docker compose up anny-minimal`    |
| Run GPU demo    | `docker compose up anny-full`       |
| Stop containers | `docker compose down`               |
| Rebuild clean   | `docker compose build --no-cache`   |

---

## 🧰 Troubleshooting

| Issue                          | Cause                     | Solution                                  |
| ------------------------------ | ------------------------- | ----------------------------------------- |
| `ModuleNotFoundError: trimesh` | Missing demo deps         | Fixed via `[examples]` extras in build    |
| `warp` import fails            | No NVIDIA runtime         | Install NVIDIA Container Toolkit          |
| Port not reachable             | Firewall or port conflict | Ensure port 7860 is free                  |
| Slow build                     | No pip cache              | Use BuildKit (`export DOCKER_BUILDKIT=1`) |

---

## 🧩 Tech Stack

* **Python** 3.11
* **PyTorch** 2.4.1 (CPU / CUDA 12.1)
* **Gradio** ≥ 4.0
* **Warp-lang** (for GPU builds)
* **Anny** (with `examples` extras)

---

## 🪪 License

* **Anny** © 2025 NAVER Corp. — Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
* **MakeHuman assets** (MPFB2) — Licensed under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

---

## 🧡 Credits

* [NAVER AI Lab — Anny](https://github.com/naver/anny)
* [MakeHuman Community](http://www.makehumancommunity.org/)
* [NVIDIA Container Toolkit](https://developer.nvidia.com/container-runtime)

---

### ✨ Maintained Dockerized Setup

Created for easy local deployment, demonstration, and experimentation with Anny using modern container practices.

