#!/usr/bin/env python3
"""
DKMS CLI Examples - Demonstration script showing various CLI usage patterns.

This script provides examples of how to use the DKMS CLI client in different scenarios.
"""

import subprocess
import sys
import time
import os
from typing import List

def run_command(cmd: List[str], description: str = "", check_output: bool = False):
    """Run a CLI command and display the results."""
    print(f"\n{'='*60}")
    if description:
        print(f"EXAMPLE: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        if check_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return result.stdout
        else:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"STDOUT: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"STDERR: {e.stderr}")
    except FileNotFoundError:
        print("Error: Make sure you're running this from the correct directory and the CLI is available")

def main():
    """Demonstrate DKMS CLI usage with examples."""
    
    print("DKMS CLI Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use the DKMS CLI client.")
    print("Make sure your DKMS server is running before executing these examples.")
    print("\nTo start the server, run: uv run python start_server.py")
    
    input("\nPress Enter to continue with the examples...")
    
    # Configuration examples
    run_command(
        ["python", "cli.py", "config"],
        "Show current CLI configuration"
    )
    
    # Basic key operations
    run_command(
        ["python", "cli.py", "set", "demo-key", "--value", "my-secret-password"],
        "Store a simple key with inline value"
    )
    
    run_command(
        ["python", "cli.py", "get", "demo-key"],
        "Retrieve the stored key"
    )
    
    run_command(
        ["python", "cli.py", "get", "demo-key", "--hide-value"],
        "Check if key exists without showing value"
    )
    
    # Interactive prompt example
    print("\n" + "="*60)
    print("EXAMPLE: Store a key using secure prompt")
    print("COMMAND: python cli.py set secret-key --prompt")
    print("="*60)
    print("This would prompt you securely for the value. Skipping in demo...")
    
    # Piped input example
    run_command(
        ["python", "cli.py", "set", "piped-key", "--value", "value-from-pipe"],
        "Store a key (simulating piped input)"
    )
    
    # List keys
    run_command(
        ["python", "cli.py", "list"],
        "List all keys (table format)"
    )
    
    run_command(
        ["python", "cli.py", "list", "--format", "simple"],
        "List all keys (simple format)"
    )
    
    run_command(
        ["python", "cli.py", "list", "--format", "json"],
        "List all keys (JSON format)"
    )
    
    # Advanced scenarios
    print("\n" + "="*60)
    print("EXAMPLE: Using environment variables for configuration")
    print("="*60)
    
    # Set environment variables for demo
    env = os.environ.copy()
    env["API_KEY"] = "custom-api-key"
    env["DKMS_URL"] = "http://127.0.0.1:8000"
    
    try:
        result = subprocess.run(
            ["python", "cli.py", "config"],
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print("With API_KEY=custom-api-key and DKMS_URL=http://127.0.0.1:8000:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run with custom environment: {e}")
    
    # Real-world usage patterns
    print("\n" + "="*60)
    print("REAL-WORLD USAGE PATTERNS")
    print("="*60)
    
    scenarios = [
        {
            "title": "Database Credentials Management",
            "keys": [
                ("db-host", "postgres.example.com"),
                ("db-user", "myapp_user"),
                ("db-password", "super-secure-password-123"),
                ("db-name", "myapp_production")
            ]
        },
        {
            "title": "API Keys and Tokens",
            "keys": [
                ("stripe-api-key", "sk_live_1234567890abcdef"),
                ("jwt-secret", "your-256-bit-secret"),
                ("oauth-client-secret", "oauth_secret_xyz789")
            ]
        },
        {
            "title": "Application Configuration",
            "keys": [
                ("app-secret-key", "flask-secret-key-for-sessions"),
                ("encryption-key", "aes-256-encryption-key"),
                ("redis-url", "redis://localhost:6379/0")
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['title']}:")
        print("-" * len(scenario['title']))
        
        for key_name, key_value in scenario['keys']:
            run_command(
                ["python", "cli.py", "set", key_name, "--value", key_value],
                f"Store {key_name}"
            )
    
    # Show all stored keys
    run_command(
        ["python", "cli.py", "list"],
        "List all stored keys"
    )
    
    # Cleanup demonstration
    print("\n" + "="*60)
    print("CLEANUP: Deleting demo keys")
    print("="*60)
    
    # Get list of keys to delete
    try:
        result = subprocess.run(
            ["python", "cli.py", "list", "--format", "simple"],
            capture_output=True,
            text=True,
            check=True
        )
        keys_to_delete = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        for key_name in keys_to_delete:
            run_command(
                ["python", "cli.py", "delete", key_name, "--yes"],
                f"Delete {key_name}"
            )
    except subprocess.CalledProcessError:
        print("Could not retrieve keys for cleanup")
    
    # Final list to confirm cleanup
    run_command(
        ["python", "cli.py", "list"],
        "Verify cleanup (should show no keys)"
    )
    
    print("\n" + "="*60)
    print("ADDITIONAL CLI USAGE TIPS")
    print("="*60)
    print("""
1. Configuration Priority:
   Command line options > Environment variables > Defaults

2. Environment Variables:
   - DKMS_URL: Set the server URL (default: http://127.0.0.1:8000)
   - API_KEY: Set the authentication key (default: default-dev-key)

3. Input Methods for Values:
   - --value "secret"           # Inline (visible in command history)
   - --prompt                   # Secure prompt (recommended)
   - echo "secret" | cli.py set # Pipe from stdin

4. Output Formats:
   - --format table    # Rich table (default)
   - --format simple   # Plain text, one per line
   - --format json     # JSON output (for scripting)

5. Security Best Practices:
   - Use --prompt for sensitive values
   - Set API_KEY via environment variable
   - Use HTTPS in production (set DKMS_URL to https://)
   - Rotate keys regularly

6. Scripting Examples:
   # Store multiple keys from a file
   cat secrets.txt | while read line; do
       name=$(echo $line | cut -d= -f1)
       value=$(echo $line | cut -d= -f2-)
       python cli.py set "$name" --value "$value"
   done
   
   # Backup all keys
   for key in $(python cli.py list --format simple); do
       value=$(python cli.py get "$key" | grep "Value:" | cut -d: -f2- | xargs)
       echo "$key=$value" >> backup.txt
   done
""")

if __name__ == "__main__":
    main()