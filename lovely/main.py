import datetime
import tomllib

import paramiko
from pydantic import field_validator, BaseModel

from system_info import parse_output
from setup_scripts import *


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
    system_config = parse_output(eval_iostream(ssh_client, node, open('sys_check.sh').read()))
    print(f"""
Validated connection to {node.user}@{node.host} (node {node.name}):
Detected OS: {system_config.os.operating_system}
Detected Kernel: {system_config.os.kernel_name} {system_config.os.kernel_release}
Detected CPU: {system_config.cpu.model} ({system_config.cpu.cores} cores)
Detected Memory: {system_config.memory.total_gb}GB total, {system_config.memory.available_gb}GB available
Detected Storage: {system_config.storage.total} total, {system_config.storage.available} available
Detected Architecture: {system_config.os.arch}
Detected Virtualization: {system_config.virtualization.type if system_config.virtualization.is_virtualized else "None"}
Detected Container: {system_config.virtualization.container if system_config.virtualization.is_container else "None"}
Detected Network Interfaces: {', '.join(system_config.network.interfaces)}
Detected System Load: {system_config.system_load.average_1min} (1min), {system_config.system_load.average_5min} (5min)
Detected Container Tools:
- Docker: {system_config.container.docker_version or "Not installed"}
- Podman: {system_config.container.podman_version or "Not installed"}
- Kubernetes:
  * Client: {system_config.container.kubernetes_client_version or "Not installed"}
  * Kustomize: {system_config.container.kubernetes_kustomize_version or "Not installed"}
  * Server: {system_config.container.kubernetes_server_version or "Not installed"}
""")
    print("Trying to run data client...")
    eval_iostream(ssh_client, node, DATA_CLIENT_RUN, do_print=True)
    ssh_client.close()
