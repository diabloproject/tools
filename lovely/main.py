import datetime
import tomllib

import paramiko
from pydantic import field_validator, BaseModel, model_validator


class Node(BaseModel):
    name: str
    user: str
    password: str | None = None
    host: str | None = None
    setup: str | None = None


class Inventory(BaseModel):
    compute: dict[str, Node]
    data: dict[str, Node]

    # noinspection PyNestedDecorators
    @field_validator("compute", "data", mode="before")
    @classmethod
    def inject_key_into_item(cls, v: dict[str, dict]) -> dict[str, dict]:
        return {key: {**value, "name": key} for key, value in v.items()}


from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class CPUInfo(BaseModel):
    model: Optional[str] = None
    vendor: Optional[str] = None
    family: Optional[int] = None
    model_id: Optional[int] = None
    stepping: Optional[int] = None
    cores: Optional[int] = None
    threads_per_core: Optional[int] = None
    sockets: Optional[int] = None
    max_mhz: Optional[float] = None
    min_mhz: Optional[float] = None
    flags: Optional[str] = None
    architecture: Optional[str] = None

    @field_validator('cores', 'threads_per_core', 'sockets', mode='before')
    @classmethod
    def parse_int_fields(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @field_validator('max_mhz', 'min_mhz', mode='before')
    @classmethod
    def parse_float_fields(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


class MemoryInfo(BaseModel):
    total_kb: Optional[int] = None
    available_kb: Optional[int] = None
    free_kb: Optional[int] = None
    swap_total_kb: Optional[int] = None
    swap_free_kb: Optional[int] = None

    @field_validator('*', mode='before')
    @classmethod
    def parse_memory_fields(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @property
    def total_gb(self) -> Optional[float]:
        return round(self.total_kb / 1024 / 1024, 2) if self.total_kb else None

    @property
    def available_gb(self) -> Optional[float]:
        return round(self.available_kb / 1024 / 1024, 2) if self.available_kb else None

    @property
    def free_gb(self) -> Optional[float]:
        return round(self.free_kb / 1024 / 1024, 2) if self.free_kb else None


class OSInfo(BaseModel):
    kernel_name: Optional[str] = None
    kernel_release: Optional[str] = None
    kernel_version: Optional[str] = None
    machine: Optional[str] = None
    processor: Optional[str] = None
    hardware_platform: Optional[str] = None
    operating_system: Optional[str] = None
    distro_id: Optional[str] = None
    distro_name: Optional[str] = None
    distro_version: Optional[str] = None
    distro_codename: Optional[str] = None
    distro_pretty: Optional[str] = None
    arch: Optional[str] = None
    byte_order: Optional[str] = None
    address_sizes: Optional[str] = None

    @field_validator('*', mode='before')
    @classmethod
    def parse_os_fields(cls, v):
        if v is None or v == 'unknown':
            return None
        if isinstance(v, int):
            return str(v)
        return None


class LibcInfo(BaseModel):
    implementation: Optional[str] = None
    version: Optional[float] = None
    full_info: Optional[str] = None
    musl_detected: bool = False
    musl_version: Optional[str] = None
    glibc_version: Optional[str] = None

    @field_validator('musl_detected', mode='before')
    @classmethod
    def parse_musl_detected(cls, v):
        if isinstance(v, str):
            return v.lower() == 'yes'
        return bool(v) if v is not None else False


class StorageInfo(BaseModel):
    total: Optional[str] = None
    used: Optional[str] = None
    available: Optional[str] = None
    usage_percent: Optional[int] = None
    filesystem_root: Optional[str] = None

    @field_validator('usage_percent', mode='before')
    @classmethod
    def parse_usage_percent(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return int(str(v).replace('%', ''))
        except (ValueError, TypeError):
            return None


class VirtualizationInfo(BaseModel):
    type: Optional[str] = None
    container: Optional[str] = None

    @property
    def is_virtualized(self) -> bool:
        return self.type not in [None, 'none', 'unknown']

    @property
    def is_container(self) -> bool:
        return self.container not in [None, 'none', 'unknown']


class PackageManagerInfo(BaseModel):
    dpkg: Optional[str] = None
    rpm: Optional[str] = None
    pacman: Optional[str] = None
    apk: Optional[str] = None
    portage: Optional[str] = None

    @field_validator('*', mode='before')
    @classmethod
    def parse_package_managers(cls, v):
        return None if v == 'not_found' else v

    @property
    def available_managers(self) -> List[str]:
        managers = []
        for field_name, value in self.__dict__.items():
            if value is not None:
                managers.append(field_name)
        return managers


class SecurityInfo(BaseModel):
    selinux: Optional[str] = None
    apparmor: Optional[str] = None
    seccomp: Optional[str] = None

    @field_validator('*', mode='before')
    @classmethod
    def parse_security_fields(cls, v):
        return None if v in ['not_found', 'unknown'] else v


class NetworkInfo(BaseModel):
    interfaces: List[str] = Field(default_factory=list)
    default_route: Optional[str] = None

    @field_validator('interfaces', mode='before')
    @classmethod
    def parse_interfaces(cls, v):
        if isinstance(v, str) and v != 'unknown':
            return [iface.strip() for iface in v.split(',') if iface.strip()]
        return []


class SystemLoadInfo(BaseModel):
    average_1min: Optional[float] = None
    average_5min: Optional[float] = None
    average_15min: Optional[float] = None
    processes: Optional[int] = None

    @field_validator('average_1min', 'average_5min', 'average_15min', mode='before')
    @classmethod
    def parse_load_average(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @field_validator('processes', mode='before')
    @classmethod
    def parse_processes(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class SystemInfo(BaseModel):
    """Complete system information model"""

    # Basic system info
    hostname: Optional[str] = None
    uptime_seconds: Optional[int] = None
    current_user: Optional[str] = None
    shell: Optional[str] = None
    timezone: Optional[str] = None
    timestamp: Optional[str] = None

    # Structured components
    os: OSInfo = Field(default_factory=OSInfo)
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    storage: StorageInfo = Field(default_factory=StorageInfo)
    libc: LibcInfo = Field(default_factory=LibcInfo)
    virtualization: VirtualizationInfo = Field(default_factory=VirtualizationInfo)
    package_managers: PackageManagerInfo = Field(default_factory=PackageManagerInfo)
    security: SecurityInfo = Field(default_factory=SecurityInfo)
    network: NetworkInfo = Field(default_factory=NetworkInfo)
    system_load: SystemLoadInfo = Field(default_factory=SystemLoadInfo)

    # Additional fields
    init_system: Optional[str] = None
    systemd_version: Optional[str] = None
    python_version: Optional[str] = None
    gcc_version: Optional[str] = None

    # Raw data for anything not explicitly modeled
    raw_data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('uptime_seconds', mode='before')
    @classmethod
    def parse_uptime(cls, v):
        if v is None or v == 'unknown':
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    @field_validator('systemd_version', mode='before')
    @classmethod
    def parse_systemd_version(cls, v):
        if isinstance(v, int) or isinstance(v, float):
            return str(v)
        return v

    @model_validator(mode='before')
    @classmethod
    def parse_raw_data(cls, data):
        """Parse flat key-value pairs into structured components"""

        if not isinstance(data, dict):
            return data

        # Create component dictionaries
        os_data = {}
        cpu_data = {}
        memory_data = {}
        storage_data = {}
        libc_data = {}
        virt_data = {}
        pkg_data = {}
        security_data = {}
        network_data = {}
        load_data = {}

        # Keep track of all keys we've processed
        processed_keys = set()

        for key, value in data.items():
            # Clean up the value
            if value == 'unknown' or value == 'not_found':
                value = None

            # Route to appropriate component
            if key.startswith('kernel_') or key.startswith('distro_') or key in [
                'machine', 'processor', 'hardware_platform', 'operating_system',
                'arch', 'byte_order', 'address_sizes'
            ]:
                os_data[key] = value
                processed_keys.add(key)

            elif key.startswith('cpu_'):
                cpu_key = key.replace('cpu_', '')
                cpu_data[cpu_key] = value
                processed_keys.add(key)

            elif key.startswith('mem_') or key.startswith('swap_'):
                memory_data[key] = value
                processed_keys.add(key)

            elif key.startswith('disk_') or key == 'filesystem_root':
                storage_key = key.replace('disk_', '') if key.startswith('disk_') else key
                storage_data[storage_key] = value
                processed_keys.add(key)

            elif key.startswith('libc_') or key.startswith('musl_') or key == 'glibc_version':
                libc_key = key.replace('libc_', '').replace('musl_', 'musl_')
                libc_data[libc_key] = value
                processed_keys.add(key)

            elif key in ['virt_type', 'container']:
                virt_key = 'type' if key == 'virt_type' else key
                virt_data[virt_key] = value
                processed_keys.add(key)

            elif key in ['dpkg', 'rpm', 'pacman', 'apk', 'portage']:
                pkg_data[key] = value
                processed_keys.add(key)

            elif key in ['selinux', 'apparmor', 'seccomp']:
                security_data[key] = value
                processed_keys.add(key)

            elif key in ['interfaces', 'default_route']:
                network_data[key] = value
                processed_keys.add(key)

            elif key.startswith('load_average_') or key == 'processes':
                load_key = key.replace('load_average_', 'average_')
                load_data[load_key] = value
                processed_keys.add(key)

        # Update data with structured components
        data['os'] = os_data
        data['cpu'] = cpu_data
        data['memory'] = memory_data
        data['storage'] = storage_data
        data['libc'] = libc_data
        data['virtualization'] = virt_data
        data['package_managers'] = pkg_data
        data['security'] = security_data
        data['network'] = network_data
        data['system_load'] = load_data

        # Store remaining unprocessed data in raw_data
        raw_data = {}
        for key, value in data.items():
            if key not in processed_keys and key not in [
                'os', 'cpu', 'memory', 'storage', 'libc', 'virtualization',
                'package_managers', 'security', 'network', 'system_load', 'raw_data'
            ]:
                if key not in ['hostname', 'uptime_seconds', 'current_user', 'shell',
                               'timezone', 'timestamp', 'init_system', 'systemd_version',
                               'python_version', 'gcc_version']:
                    raw_data[key] = value

        data['raw_data'] = raw_data

        return data

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a concise summary of the system"""
        return {
            'hostname': self.hostname,
            'os': {
                'name': self.os.distro_pretty or self.os.distro_name,
                'kernel': f"{self.os.kernel_name} {self.os.kernel_release}",
                'architecture': self.os.arch
            },
            'cpu': {
                'model': self.cpu.model,
                'cores': self.cpu.cores,
                'vendor': self.cpu.vendor
            },
            'memory': {
                'total_gb': self.memory.total_gb,
                'available_gb': self.memory.available_gb
            },
            'virtualization': {
                'is_virtual': self.virtualization.is_virtualized,
                'type': self.virtualization.type,
                'is_container': self.virtualization.is_container
            }
        }


def parse_output(output: str) -> SystemInfo:
    """Parse raw script output into SystemInfo model"""
    data = {}

    for line in output.strip().split('\n'):
        line = line.strip()

        # Skip empty lines, comments, and report markers
        if not line or line.startswith('#') or line.startswith('REPORT_'):
            continue

        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Basic type conversion
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'

            data[key] = value

    return SystemInfo(**data)


schema = tomllib.load(open('inventory.toml', 'rb'))
schema.setdefault('compute', {})
schema.setdefault('data', {})

inventory = Inventory.model_validate(schema)


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def eval_iostream(ssh_client, node, command, do_print=False):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    left = ""
    total = ""
    while current := stdout.read().decode():
        total += current
        parts = (left + current).split('\n')
        if len(parts) == 1:  # No newline yet
            left = parts[0]
        for item in parts[:-1]:
            if do_print:
                print(f"[{node.user}@{node.name} {now()}] {item}")
    return total


for _, node in inventory.compute.items():
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(node.host, username=node.user, password=node.password)
    out = eval_iostream(ssh_client, node, open('sys_check.sh').read())
    print(parse_output(out))
    if node.setup:
        eval_iostream(ssh_client, node, node.setup, do_print=False)
        # eval_iostream(ssh_client, node, node.setup)
    ssh_client.close()

"""
import paramiko

# SSH connection details
hostname = 'your.remote.host'
port = 22
username = 'your_username'
password = 'your_password'  # Or use a private key for better security

# Create SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Connect to the server
    client.connect(hostname, port=port, username=username, password=password)

    # Execute command
    stdin, stdout, stderr = client.exec_command('ls -l /home/your_username')

    # Read output
    print("STDOUT:")
    print(stdout.read().decode())

    print("STDERR:")
    print(stderr.read().decode())

finally:
    # Close connection
    client.close()
"""