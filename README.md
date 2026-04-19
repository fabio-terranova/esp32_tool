# ESP32 Binary Image Tool

Small utility to inspect and fix ESP32 firmware footer (image checksum and
SHA256 hash) after binary patching.

## Features

- Show image header, segments, checksum, and optional SHA256 status
- Fix invalid checksum and appended SHA256 hash in-place

## Usage

- Show help:
  `esp32_tool.py -h`
- Show image info:
  `esp32_tool.py info firmware.bin`
- Fix checksum/hash:
  `esp32_tool.py fix firmware.bin`
