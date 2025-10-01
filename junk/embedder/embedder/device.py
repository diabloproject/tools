import torch
import logging
from typing import Optional, List

def get_available_devices() -> List[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    return devices

def auto_select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_device_info(device: str) -> dict:
    info = {"device": device, "type": device.split(":")[0]}
    
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(device_idx)
            info.update({
                "name": props.name,
                "memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}"
            })
    elif device == "mps":
        info["name"] = "Apple Silicon"
    else:
        info["name"] = "CPU"
    
    return info

def validate_device(device: Optional[str]) -> str:
    if device is None:
        return auto_select_device()
    
    available = get_available_devices()
    if device not in available:
        logging.warning(f"Device {device} not available, using auto-selected device")
        return auto_select_device()
    
    return device