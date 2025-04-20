"""
This module implements a binary protocol for cluster commands.

The protocol header is fixed at 56 bytes:
  * Magic Number (4 bytes)   : 0x12233445
  * Version (4 bytes)        : 0x00000001
  * Command ID (4 bytes)
  * Payload Length (4 bytes)
  * 5 field pairs (each 8 bytes): (field_id, field_value)
"""
from io import BytesIO
import struct
import socket
from typing import IO, BinaryIO, Literal, overload
from dataclasses import dataclass, field

# -------------------------
# Protocol Constants
# -------------------------
MAGIC_NUMBER = 0x12233445
VERSION = 0x00000001
FIXED_FIELD_COUNT = 5
HEADER_SIZE = 4 + 4 + 4 + 4 + (FIXED_FIELD_COUNT * 8)  # 56 bytes

# -------------------------
# Command Definitions
# -------------------------
STORE_BLOCK_COMMAND = 0x00001001
FETCH_BLOCK_COMMAND = 0x00001002
COMMANDS = {
    "STORE_BLOCK_COMMAND": {
        "command_id": STORE_BLOCK_COMMAND,
        "encoder_arguments": {
            "block_id": int,
            "block": {
                "type": bytes,
                "content_size": 4 * 1024 * 1024,  # 4MB
            }
        },
        "fields": {
            # It is assumed that only block content is stored in the payload.
            "block_id": 0x00000001,
        },
        "response_id": 0x2001,  # BASIC_STATUS_RESPONSE
    },
    "FETCH_BLOCK_COMMAND": {
        "command_id": FETCH_BLOCK_COMMAND,
        "encoder_arguments": {
            "block_id": int,
        },
        "fields": {
            "block_id": 0x00000001,
        },
        "response_id": 0x2002,  # FETCH_BLOCK_RESPONSE
    }
}

# -------------------------
# Response Definitions
# -------------------------
RESPONSES = {
    "BASIC_STATUS_RESPONSE": {
        "response_id": 0x2001,
        "fields": {
            "status": 0x00000001,
        }
    },
    "FETCH_BLOCK_RESPONSE": {
        "response_id": 0x2002,
        "encoder_arguments": {
            "block_id": int,
            "block": {
                "type": bytes,
                "content_size": 4 * 1024 * 1024,  # 4MB
            }
        },
        "fields": {
            # It is assumed that only block content is stored in the payload.
            "status": 0x00000001,
        }
    }
}

# Build reverse lookup dictionary for command IDs.
COMMAND_ID_TO_NAME = {spec["command_id"]: name for name, spec in COMMANDS.items()}


@dataclass
class Message:
    action_code: int  # Command or response ID
    payload: bytes
    fields: list[tuple[int, int]] = field(default_factory=list)


def encode_message(
    message: Message
) -> bytes:
    io = BytesIO()
    encode_message_stream(message, io)
    return io.getvalue()

def encode_message_stream(
    message: Message,
    stream: BinaryIO
) -> None:
    stream.write(MAGIC_NUMBER.to_bytes(4, 'big'))
    stream.write(VERSION.to_bytes(4, 'big'))
    stream.write(len(message.payload).to_bytes(4, 'big'))
    stream.write(message.action_code.to_bytes(4, 'big'))
    for field_id, field_value in message.fields:
        stream.write(field_id.to_bytes(4, 'big'))
        stream.write(field_value.to_bytes(4, 'big'))
    stream.write(message.payload)


def decode_message_stream(
    stream: BinaryIO
) -> Message:
    ...

def decode_message(
    data: bytes
) -> Message:
    ...
