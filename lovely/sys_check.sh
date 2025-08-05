#!/bin/bash

# System Information Reporter - Robust Version
# Outputs comprehensive system information in key=value format for easy parsing

set -euo pipefail

# Force English locale to avoid localization issues
export LC_ALL=C
export LANG=C

# Function to safely get command output and sanitize it
safe_cmd() {
    local cmd="$1"
    local default="${2:-unknown}"
    if command -v "${cmd%% *}" >/dev/null 2>&1; then
        # Execute command, remove newlines, trim whitespace
        local result
        result=$(eval "$cmd" 2>/dev/null | tr '\n' ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "$default")
        # If result is empty, return default
        [[ -n "$result" ]] && echo "$result" || echo "$default"
    else
        echo "$default"
    fi
}

# Function to safely read file and sanitize
safe_read() {
    local file="$1"
    local default="${2:-unknown}"
    if [[ -r "$file" ]]; then
        local result
        result=$(cat "$file" 2>/dev/null | tr '\n' ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "$default")
        [[ -n "$result" ]] && echo "$result" || echo "$default"
    else
        echo "$default"
    fi
}

# Function to get first line of file and sanitize
safe_read_line() {
    local file="$1"
    local default="${2:-unknown}"
    if [[ -r "$file" ]]; then
        local result
        result=$(head -n1 "$file" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "$default")
        [[ -n "$result" ]] && echo "$result" || echo "$default"
    else
        echo "$default"
    fi
}

# Function to extract value after colon and clean it
extract_after_colon() {
    local input="$1"
    local default="${2:-unknown}"
    if [[ -n "$input" && "$input" != "unknown" ]]; then
        local result
        result=$(echo "$input" | cut -d: -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -n "$result" ]] && echo "$result" || echo "$default"
    else
        echo "$default"
    fi
}

echo "REPORT_START=true"
echo "TIMESTAMP=$(date -u '+%Y-%m-%d_%H:%M:%S_UTC')"

# Basic System Info
echo "hostname=$(hostname 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo 'unknown')"
echo "uptime_seconds=$(safe_cmd 'cat /proc/uptime | cut -d. -f1')"
echo "current_user=$(whoami 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo 'unknown')"
echo "shell=${SHELL:-unknown}"
echo "timezone=$(safe_cmd 'timedatectl show --property=Timezone --value 2>/dev/null' "$(date +%Z 2>/dev/null || echo 'unknown')")"

# OS and Distribution Info
echo "kernel_name=$(uname -s 2>/dev/null || echo 'unknown')"
echo "kernel_release=$(uname -r 2>/dev/null || echo 'unknown')"
echo "kernel_version=$(safe_cmd 'uname -v' | sed 's/#[0-9]*[[:space:]]*//')"
echo "machine=$(uname -m 2>/dev/null || echo 'unknown')"
echo "processor=$(uname -p 2>/dev/null || echo 'unknown')"
echo "hardware_platform=$(uname -i 2>/dev/null || echo 'unknown')"
echo "operating_system=$(uname -o 2>/dev/null || echo 'unknown')"

# Distribution Detection
if [[ -f /etc/os-release ]]; then
    # Source the file safely
    eval "$(grep -E '^(ID|NAME|VERSION|VERSION_CODENAME|UBUNTU_CODENAME|PRETTY_NAME)=' /etc/os-release 2>/dev/null || true)"
    echo "distro_id=${ID:-unknown}"
    echo "distro_name=${NAME:-unknown}"
    echo "distro_version=${VERSION:-unknown}"
    echo "distro_codename=${VERSION_CODENAME:-${UBUNTU_CODENAME:-unknown}}"
    echo "distro_pretty=${PRETTY_NAME:-unknown}"
elif [[ -f /etc/redhat-release ]]; then
    REDHAT_RELEASE=$(safe_read_line /etc/redhat-release)
    echo "distro_id=rhel"
    echo "distro_name=$REDHAT_RELEASE"
    echo "distro_version=unknown"
    echo "distro_codename=unknown"
    echo "distro_pretty=$REDHAT_RELEASE"
else
    echo "distro_id=unknown"
    echo "distro_name=unknown"
    echo "distro_version=unknown"
    echo "distro_codename=unknown"
    echo "distro_pretty=unknown"
fi

# Architecture Details
echo "arch=$(uname -m 2>/dev/null || echo 'unknown')"

# CPU Information using multiple methods
if command -v lscpu >/dev/null 2>&1; then
    # Use lscpu with forced English locale
    CPU_MODEL=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "model name"')")
    CPU_VENDOR=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "vendor"')")
    CPU_FAMILY=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "cpu family"')")
    CPU_MODEL_ID=$(extract_after_colon "$(safe_cmd 'lscpu | grep -E "^Model:"')")
    CPU_STEPPING=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "stepping"')")
    CPU_THREADS_PER_CORE=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "thread.*per core"')")
    CPU_SOCKETS=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "socket"')")
    CPU_MAX_MHZ=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "cpu max mhz"')")
    CPU_MIN_MHZ=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "cpu min mhz"')")
    CPU_FLAGS=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "flags"')")
    BYTE_ORDER=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "byte order"')")
    ADDRESS_SIZES=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "address sizes"')")
    CPU_ARCH=$(extract_after_colon "$(safe_cmd 'lscpu | grep -i "architecture"')")
else
    # Fallback to /proc/cpuinfo
    CPU_MODEL=$(safe_cmd 'grep -i "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed "s/^[[:space:]]*//"')
    CPU_VENDOR=$(safe_cmd 'grep -i "vendor_id" /proc/cpuinfo | head -1 | cut -d: -f2 | sed "s/^[[:space:]]*//"')
    CPU_FAMILY=$(safe_cmd 'grep -i "cpu family" /proc/cpuinfo | head -1 | cut -d: -f2 | sed "s/^[[:space:]]*//"')
    CPU_MODEL_ID=$(safe_cmd 'grep -i "^model" /proc/cpuinfo | head -1 | cut -d: -f2 | sed "s/^[[:space:]]*//"')
    CPU_STEPPING=$(safe_cmd 'grep -i "stepping" /proc/cpuinfo | head -1 | cut -d: -f2 | sed "s/^[[:space:]]*//"')
    CPU_FLAGS=$(safe_cmd 'grep -i "flags" /proc/cpuinfo | head -1 | cut -d: -f2 | sed "s/^[[:space:]]*//"')
    CPU_THREADS_PER_CORE="unknown"
    CPU_SOCKETS="unknown"
    CPU_MAX_MHZ="unknown"
    CPU_MIN_MHZ="unknown"
    BYTE_ORDER="unknown"
    ADDRESS_SIZES="unknown"
    CPU_ARCH="unknown"
fi

echo "cpu_model=$CPU_MODEL"
echo "cpu_vendor=$CPU_VENDOR"
echo "cpu_family=$CPU_FAMILY"
echo "cpu_model_id=$CPU_MODEL_ID"
echo "cpu_stepping=$CPU_STEPPING"
echo "cpu_cores=$(safe_cmd 'nproc')"
echo "cpu_threads_per_core=$CPU_THREADS_PER_CORE"
echo "cpu_sockets=$CPU_SOCKETS"
echo "cpu_max_mhz=$CPU_MAX_MHZ"
echo "cpu_min_mhz=$CPU_MIN_MHZ"
echo "cpu_flags=$CPU_FLAGS"
echo "cpu_arch=$CPU_ARCH"
echo "byte_order=$BYTE_ORDER"
echo "address_sizes=$ADDRESS_SIZES"

# Memory Information
if [[ -f /proc/meminfo ]]; then
    echo "mem_total_kb=$(safe_cmd 'grep -i "MemTotal" /proc/meminfo | awk "{print \$2}"')"
    echo "mem_available_kb=$(safe_cmd 'grep -i "MemAvailable" /proc/meminfo | awk "{print \$2}"')"
    echo "mem_free_kb=$(safe_cmd 'grep -i "MemFree" /proc/meminfo | awk "{print \$2}"')"
    echo "swap_total_kb=$(safe_cmd 'grep -i "SwapTotal" /proc/meminfo | awk "{print \$2}"')"
    echo "swap_free_kb=$(safe_cmd 'grep -i "SwapFree" /proc/meminfo | awk "{print \$2}"')"
else
    echo "mem_total_kb=unknown"
    echo "mem_available_kb=unknown"
    echo "mem_free_kb=unknown"
    echo "swap_total_kb=unknown"
    echo "swap_free_kb=unknown"
fi

# Storage Information
ROOT_FS_INFO=$(safe_cmd 'df -h / | tail -1')
if [[ "$ROOT_FS_INFO" != "unknown" ]]; then
    echo "disk_total=$(echo "$ROOT_FS_INFO" | awk '{print $2}')"
    echo "disk_used=$(echo "$ROOT_FS_INFO" | awk '{print $3}')"
    echo "disk_available=$(echo "$ROOT_FS_INFO" | awk '{print $4}')"
    echo "disk_usage_percent=$(echo "$ROOT_FS_INFO" | awk '{print $5}' | tr -d '%')"
    echo "filesystem_root=$(safe_cmd 'df -T / | tail -1 | awk "{print \$2}"')"
else
    echo "disk_total=unknown"
    echo "disk_used=unknown"
    echo "disk_available=unknown"
    echo "disk_usage_percent=unknown"
    echo "filesystem_root=unknown"
fi

# libc Implementation
if command -v ldd >/dev/null 2>&1; then
    LIBC_INFO=$(safe_cmd 'ldd --version | head -1')
    if [[ "$LIBC_INFO" != "unknown" ]]; then
        echo "libc_implementation=$(echo "$LIBC_INFO" | awk '{print $1}')"
        echo "libc_version=$(echo "$LIBC_INFO" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || echo 'unknown')"
        echo "libc_full_info=$LIBC_INFO"
    else
        echo "libc_implementation=unknown"
        echo "libc_version=unknown"
        echo "libc_full_info=unknown"
    fi
else
    echo "libc_implementation=unknown"
    echo "libc_version=unknown"
    echo "libc_full_info=unknown"
fi

# Check for musl specifically
MUSL_PATH=$(ls /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1 2>/dev/null | head -1 || echo "")
if [[ -n "$MUSL_PATH" ]]; then
    echo "musl_detected=yes"
    echo "musl_version=$(safe_cmd '$MUSL_PATH --version | head -1 | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?"')"
else
    echo "musl_detected=no"
    echo "musl_version=unknown"
fi

# Virtualization Detection
echo "virt_type=$(safe_cmd 'systemd-detect-virt 2>/dev/null' "$(safe_cmd 'dmesg 2>/dev/null | grep -i hypervisor | head -1 | cut -d] -f2 | sed "s/^[[:space:]]*//"' 'unknown')")"
echo "container=$(safe_cmd 'systemd-detect-virt --container 2>/dev/null')"

# Network Information
echo "interfaces=$(safe_cmd 'ip -o link show 2>/dev/null | awk -F: "{print \$2}" | tr "\n" "," | sed "s/,$//" | sed "s/^[[:space:]]*//"')"
echo "default_route=$(safe_cmd 'ip route show default 2>/dev/null | head -1')"

# Package Managers
echo "dpkg=$(safe_cmd 'dpkg --version 2>/dev/null | head -1' 'not_found')"
echo "rpm=$(safe_cmd 'rpm --version 2>/dev/null' 'not_found')"
echo "pacman=$(safe_cmd 'pacman --version 2>/dev/null | head -1' 'not_found')"
echo "apk=$(safe_cmd 'apk --version 2>/dev/null' 'not_found')"
echo "portage=$(safe_cmd 'emerge --version 2>/dev/null | head -1' 'not_found')"

# Init System
if [[ -d /run/systemd/system ]]; then
    echo "init_system=systemd"
    echo "systemd_version=$(safe_cmd 'systemctl --version 2>/dev/null | head -1 | awk "{print \$2}"')"
elif [[ -f /sbin/openrc ]]; then
    echo "init_system=openrc"
    echo "systemd_version=not_applicable"
elif [[ -f /etc/inittab ]]; then
    echo "init_system=sysv"
    echo "systemd_version=not_applicable"
else
    echo "init_system=unknown"
    echo "systemd_version=unknown"
fi

# Additional System Details
echo "python_version=$(safe_cmd 'python3 --version 2>&1 | awk "{print \$2}"' "$(safe_cmd 'python --version 2>&1 | awk "{print \$2}"')")"
echo "gcc_version=$(safe_cmd 'gcc --version 2>/dev/null | head -1 | awk "{print \$3}"')"
echo "glibc_version=$(safe_cmd 'getconf GNU_LIBC_VERSION 2>/dev/null | awk "{print \$2}"')"

LOAD_AVG=$(safe_read_line /proc/loadavg)
if [[ "$LOAD_AVG" != "unknown" ]]; then
    echo "load_average_1min=$(echo "$LOAD_AVG" | cut -d' ' -f1)"
    echo "load_average_5min=$(echo "$LOAD_AVG" | cut -d' ' -f2)"
    echo "load_average_15min=$(echo "$LOAD_AVG" | cut -d' ' -f3)"
else
    echo "load_average_1min=unknown"
    echo "load_average_5min=unknown"
    echo "load_average_15min=unknown"
fi

echo "processes=$(safe_cmd 'ps aux 2>/dev/null | wc -l')"

# Security Features
echo "selinux=$(safe_cmd 'getenforce 2>/dev/null')"
echo "apparmor=$(safe_cmd 'aa-status --enabled 2>/dev/null && echo enabled || echo disabled' 'not_found')"
echo "seccomp=$(safe_cmd 'grep Seccomp /proc/self/status 2>/dev/null | awk "{print \$2}"')"

echo "REPORT_END=true"