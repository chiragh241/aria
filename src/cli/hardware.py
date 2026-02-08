"""Hardware detection for Ollama model recommendations.

Checks RAM, CPU, and GPU to suggest the best local model for the user's machine.

Model selection rationale:
- Models are chosen from Ollama library (ollama.com/library) based on:
  - VRAM/RAM requirements (Ollama FAQ, community benchmarks)
  - Tool/function calling support (required for Aria skills)
  - Size tiers: 1-3B (tiny), 7-9B (small), 12-14B (medium), 70B+ (large)
- Only models that support tool calling are recommended so skills work.
- When the user has models already downloaded, we compare them to hardware
  and suggest the best-fitting downloaded model if it beats our tier recommendation.
"""

import platform
import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class HardwareSpecs:
    """Detected hardware capabilities."""

    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    cpu_cores: int = 0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_type: str = ""  # "nvidia" | "apple" | "amd" | ""
    platform: str = ""


def detect_hardware() -> HardwareSpecs:
    """Detect system hardware for model recommendations."""
    specs = HardwareSpecs(platform=platform.system())

    try:
        import psutil

        vm = psutil.virtual_memory()
        specs.ram_total_gb = vm.total / (1024**3)
        specs.ram_available_gb = vm.available / (1024**3)
        specs.cpu_cores = psutil.cpu_count() or 0
    except ImportError:
        pass

    # GPU detection
    if platform.system() == "Darwin":
        _detect_apple_gpu(specs)
    elif platform.system() == "Linux":
        _detect_nvidia_gpu(specs)
        if not specs.gpu_name:
            _detect_amd_gpu(specs)
    elif platform.system() == "Windows":
        _detect_nvidia_gpu(specs)
        if not specs.gpu_name:
            _detect_amd_gpu(specs)

    return specs


def _detect_nvidia_gpu(specs: HardwareSpecs) -> None:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            if len(parts) >= 2:
                specs.gpu_name = parts[0].strip()
                try:
                    specs.gpu_vram_gb = float(parts[1].strip().split()[0]) / 1024  # MiB -> GB
                except (ValueError, IndexError):
                    pass
            specs.gpu_type = "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def _detect_apple_gpu(specs: HardwareSpecs) -> None:
    """Detect Apple Silicon GPU - use unified memory as proxy."""
    specs.gpu_type = "apple"
    specs.gpu_name = "Apple Silicon (unified memory)"
    if specs.ram_total_gb >= 8:
        # Apple Silicon: unified memory, use ~60% as effective for model loading
        specs.gpu_vram_gb = specs.ram_total_gb * 0.6


def _detect_amd_gpu(specs: HardwareSpecs) -> None:
    """Detect AMD GPU via rocm-smi (Linux only)."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "VRAM" in result.stdout:
            # Parse "VRAM Total Size (B): 17179869184" style output
            for line in result.stdout.split("\n"):
                if "Total" in line and "VRAM" in line:
                    try:
                        val = int(line.split(":")[-1].strip())
                        specs.gpu_vram_gb = val / (1024**3)
                        break
                    except (ValueError, IndexError):
                        pass
            if specs.gpu_vram_gb > 0:
                specs.gpu_name = "AMD GPU"
                specs.gpu_type = "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


# Regex to extract size from model tags (e.g. 8b, 70b, 13b)
_MODEL_SIZE_PATTERN = re.compile(r"(?:^|[:-])(\d+)b", re.IGNORECASE)

# Approximate RAM/VRAM (GB) by param size - Q4 quantized typical usage
# Sources: Ollama docs, community benchmarks
_PARAM_TO_GB = {
    1: (2, 1.5),
    2: (3, 2),
    3: (5, 4),
    4: (6, 5),
    7: (8, 6),
    8: (10, 8),
    9: (10, 8),
    12: (14, 10),
    13: (16, 10),
    14: (16, 12),
    22: (24, 16),
    32: (36, 24),
    34: (40, 24),
    70: (48, 40),
    72: (48, 40),
    405: (256, 231),
}


def _estimate_model_reqs_gb(model_name: str) -> tuple[float, float]:
    """Estimate (min_ram_gb, min_vram_gb) for a model from its name. Returns (0,0) if unknown."""
    match = _MODEL_SIZE_PATTERN.search(model_name)
    if not match:
        return (8, 6)  # Default for :latest or unknown
    try:
        params = int(match.group(1))
        return _PARAM_TO_GB.get(params, (8, 6))
    except (ValueError, IndexError):
        return (8, 6)


# Headroom: require ~25% above minimum for recommended performance (avoids swapping/sluggishness)
_PERFORMANCE_HEADROOM = 1.25


def _model_fits_hardware(model_name: str, specs: HardwareSpecs) -> bool:
    """Check if a model can load on the given hardware."""
    min_ram, min_vram = _estimate_model_reqs_gb(model_name)
    if specs.gpu_type and specs.gpu_vram_gb >= min_vram:
        return True
    if specs.ram_total_gb >= min_ram:
        return True
    return False


def _model_performs_well(model_name: str, specs: HardwareSpecs) -> bool:
    """Check if a model will perform well (has headroom for smooth inference)."""
    min_ram, min_vram = _estimate_model_reqs_gb(model_name)
    if specs.gpu_type:
        return specs.gpu_vram_gb >= min_vram * _PERFORMANCE_HEADROOM
    return specs.ram_total_gb >= min_ram * _PERFORMANCE_HEADROOM


def _model_quality_score(model_name: str) -> int:
    """Larger params = higher score. Used to rank models."""
    match = _MODEL_SIZE_PATTERN.search(model_name)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except (ValueError, IndexError):
        return 0


def best_downloaded_model(
    available_models: list[str],
    specs: HardwareSpecs,
) -> tuple[str | None, str]:
    """
    Find the best-performing model among already-downloaded ones.
    Only recommends models that support tool calling (skills) and perform well.
    Picks the largest model that performs well; uses it if it beats our tier recommendation.

    Returns:
        (model_name, reason) or (None, "") if no downloaded model performs well.
    """
    if not available_models:
        return None, ""

    # Only consider models that support tools and will perform well
    performing_well = [
        (m, _model_quality_score(m))
        for m in available_models
        if model_supports_tools(m) and _model_performs_well(m, specs)
    ]
    if not performing_well:
        return None, ""

    best = max(performing_well, key=lambda x: x[1])
    tier_rec, _ = recommend_ollama_model(specs)
    tier_score = _model_quality_score(tier_rec)

    # Prefer downloaded model if it's as good or larger than tier recommendation
    if best[1] >= tier_score:
        reason = f"Best fit among your {len(available_models)} downloaded model(s)—will perform well"
        return best[0], reason

    return None, ""


# Model base names that support tool/function calling (required for skills)
# See: https://ollama.com/search?c=tools and https://docs.ollama.com/capabilities/tool-calling
OLLAMA_TOOL_SUPPORTED_PREFIXES = frozenset({
    "llama3.1", "llama3.2", "llama4",
    "qwen2", "qwen2.5", "qwen3",
    "mistral-nemo", "ministral",
    "command-r-plus", "firefunction", "devstral",
    "granite4", "functiongemma",
    "olmo-3.1", "rnj-1", "gpt-oss", "deepseek-v3.1",
    "nemotron-3-nano", "lfm2.5-thinking", "glm-4.7-flash", "glm-ocr",
    "qwen3-coder", "qwen3-coder-next", "qwen3-vl",
    "mistral-small3.2",
})


def model_supports_tools(model_name: str) -> bool:
    """Check if an Ollama model supports tool/function calling (required for skills)."""
    if not model_name:
        return False
    base = model_name.split(":")[0].lower()
    if base in OLLAMA_TOOL_SUPPORTED_PREFIXES:
        return True
    return any(base.startswith(p) or base == p for p in OLLAMA_TOOL_SUPPORTED_PREFIXES)


# Model tiers: (model_name, min_ram_gb, min_vram_gb, description)
# Only tool-capable models—skills require function calling
OLLAMA_MODEL_TIERS = [
    # Tiny - 1-3B params
    ("llama3.2:1b", 4, 2, "Tiny, fast—tool calling supported"),
    ("llama3.2:3b", 6, 4, "Small, efficient—tool calling supported"),
    ("qwen2.5:3b", 6, 4, "Qwen 3B—tool calling supported"),
    # Small - 7-8B params
    ("llama3.1:8b", 8, 6, "Balanced—recommended for most users"),
    ("llama3.2:8b", 8, 6, "Llama 3.2 8B—strong generalist"),
    ("qwen2.5:7b", 8, 6, "Qwen 7B—tool calling supported"),
    ("qwen3:8b", 8, 6, "Qwen3 8B—tool calling supported"),
    # Medium - 12-14B params
    ("ministral:8b", 10, 8, "Mistral Nemo family—tool calling"),
    ("llama3.1:70b-viewer", 48, 40, "70B viewer—needs 48GB+ RAM or 40GB+ VRAM"),
    # Large - 70B
    ("llama3.1:70b", 64, 48, "70B—best quality, needs high-end hardware"),
]


def recommend_ollama_model(specs: HardwareSpecs) -> tuple[str, str]:
    """
    Recommend the best Ollama model for the given hardware.

    Returns:
        (model_name, reason) e.g. ("llama3.1:8b", "Best fit for 16GB RAM + 8GB VRAM")
    """
    effective_ram = specs.ram_available_gb
    effective_vram = specs.gpu_vram_gb if specs.gpu_type else 0

    # Prefer VRAM for GPU; otherwise use RAM
    if effective_vram >= 40:
        return "llama3.1:70b", f"Your GPU has {effective_vram:.0f}GB VRAM—70B model recommended"
    if effective_vram >= 16:
        return "llama3.1:8b", f"Your GPU has {effective_vram:.0f}GB VRAM—8B model recommended"
    if effective_vram >= 8:
        return "llama3.2:3b", f"Your GPU has {effective_vram:.0f}GB VRAM—3B runs well"
    if effective_vram >= 4:
        return "llama3.2:1b", f"Your GPU has {effective_vram:.0f}GB VRAM—1B for best speed"

    # CPU/Apple Silicon - use RAM
    if effective_ram >= 48:
        return "llama3.1:70b", f"Your system has {effective_ram:.0f}GB RAM—70B possible"
    if effective_ram >= 16:
        return "llama3.1:8b", f"Your system has {effective_ram:.0f}GB RAM—8B recommended"
    if effective_ram >= 10:
        return "llama3.2:3b", f"Your system has {effective_ram:.0f}GB RAM—3B recommended"
    if effective_ram >= 6:
        return "llama3.2:1b", f"Your system has {effective_ram:.0f}GB RAM—1B for low memory"

    return "llama3.2:1b", "Minimum viable model for your hardware"


def get_suggested_models(specs: HardwareSpecs) -> list[tuple[str, bool]]:
    """
    Get list of suggested models for the user's hardware.
    Each item is (model_name, is_recommended).
    """
    recommended, _ = recommend_ollama_model(specs)
    suggested: list[tuple[str, bool]] = []

    for model, min_ram, min_vram, _ in OLLAMA_MODEL_TIERS:
        fits = False
        if specs.gpu_type and specs.gpu_vram_gb >= min_vram:
            fits = True
        if specs.ram_total_gb >= min_ram:
            fits = True
        if fits:
            suggested.append((model, model == recommended))

    # Deduplicate by model name, put recommended first
    seen: set[str] = set()
    result: list[tuple[str, bool]] = []
    for m, rec in suggested:
        if m not in seen:
            seen.add(m)
            result.append((m, rec))

    if not result:
        result = [("llama3.2:1b", True)]

    return result


def format_hardware_summary(specs: HardwareSpecs) -> str:
    """Format hardware specs for display."""
    parts = []
    parts.append(f"{specs.ram_total_gb:.1f}GB RAM")
    if specs.gpu_type:
        parts.append(f"{specs.gpu_name}")
        if specs.gpu_vram_gb > 0:
            parts.append(f"{specs.gpu_vram_gb:.1f}GB VRAM")
    parts.append(f"{specs.cpu_cores} CPU cores")
    return " · ".join(parts)
