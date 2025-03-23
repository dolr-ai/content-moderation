import sys
import subprocess


def check_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        print("nvidia-smi output:")
        print(output.decode("utf-8"))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi failed or not found")
        return False


def check_cuda_libraries():
    try:
        output = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        print("Checking for CUDA libraries:")

        cuda_libs = ["libcuda.so", "libcudart.so", "libnvidia-ml.so"]
        for lib in cuda_libs:
            if lib in output:
                print(f"✅ {lib} found")
            else:
                print(f"❌ {lib} NOT found")

        return all(lib in output for lib in cuda_libs)
    except subprocess.CalledProcessError:
        print("Failed to check CUDA libraries")
        return False


def check_torch_cuda():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                )
        return cuda_available
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {e}")
        return False


if __name__ == "__main__":
    print("Running GPU checks...")

    nvidia_smi_ok = check_nvidia_smi()
    cuda_libs_ok = check_cuda_libraries()
    torch_cuda_ok = check_torch_cuda()

    if nvidia_smi_ok and cuda_libs_ok and torch_cuda_ok:
        print("✅ All GPU checks passed!")
        sys.exit(0)
    else:
        print("❌ GPU checks failed! Container might not be able to access the GPU.")
        sys.exit(1)
