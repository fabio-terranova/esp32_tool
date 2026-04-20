#!/bin/python

import argparse
import hashlib
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from operator import xor

VERSION = "0.1.0"

ESP_MAGIC = 0xE9
FILE_HEADER_SIZE = 8
EXTENDED_FILE_HEADER_SIZE = 16
SEGMENT_HEADER_SIZE = 8
CHECKSUM_SIZE = 1
SHA256_SIZE = 32

FILE_HEADER_FMT = "<BBBB I"
EXT_HEADER_FMT = "<B 3s H B H H 4x B"
SEGMENT_HEADER_FMT = "<II"


class HumanEnum(IntEnum):
    @classmethod
    def to_str(cls, value: int) -> str:
        try:
            name = cls(value).name.replace("FREQ_", "").replace("SIZE_", "")
            return name
        except ValueError:
            return f"Unknown ({value:#02x})"


class FlashMode(HumanEnum):
    QIO, QOUT, DIO, DOUT = 0x00, 0x01, 0x02, 0x03


class FlashSize(HumanEnum):
    SIZE_1MB, SIZE_2MB, SIZE_4MB, SIZE_8MB, SIZE_16MB = 0x00, 0x01, 0x02, 0x03, 0x04


class FlashFreq(HumanEnum):
    FREQ_40MHz, FREQ_26MHz, FREQ_20MHz, FREQ_80MHz = 0x00, 0x01, 0x02, 0x0F


@dataclass(frozen=True)
class SegmentInfo:
    load_address: int
    size: int
    file_offset: int
    data: memoryview


@dataclass(frozen=True)
class ValidationResult:
    calculated_checksum: int
    checksum_is_valid: bool
    calculated_sha256: bytes | None
    sha256_is_valid: bool


@dataclass(frozen=True)
class ParsedImage:
    magic_number: int
    num_segments: int
    spi_flash_mode: FlashMode | int
    spi_flash_size: int
    spi_flash_freq: int
    entry_point: int
    wp_pin: int
    drive_settings: bytes
    chip_id: int
    min_chip_rev: int
    max_chip_rev: int
    hash_appended: bool
    segments: list[SegmentInfo]
    checksum_offset: int
    sha256_offset: int | None


def get_segment_data(data: bytearray, offset: int, size: int) -> memoryview:
    """
    Extract segment data from the binary image as a memoryview (zero-copy).

    Args:
        data (bytearray): The entire binary image data.
        offset (int): The byte offset into data where the segment starts.
        size (int): The size of the segment in bytes.
    Returns:
        memoryview: A view into the segment data.
    """
    if offset + size > len(data):
        raise ValueError("Invalid binary file: segment extends beyond end of file")
    return memoryview(data)[offset : offset + size]


def _get_memory_types(addr: int) -> str:
    MEM_MAP = [
        (0x3F400000, 0x3F800000, "DROM"),
        (0x400D0000, 0x40400000, "IROM"),
        (0x40080000, 0x400A0000, "IRAM"),
        (0x3FFB0000, 0x3FFC0000, "BYTE_ACCESSIBLE, DRAM"),
        (0x3FFC0000, 0x40000000, "DRAM"),
        (0x50000000, 0x50000001, "RTC_DATA"),
    ]
    types = [name for start, end, name in MEM_MAP if start <= addr < end]
    return ", ".join(types) if types else "Unknown"


def calculate_checksum(segments: list[memoryview]) -> int:
    """
    Calculate the checksum of the segments.

    Args:
        segments (list of memoryview): A list of segment data views.
    Returns:
        int: The calculated checksum byte.
    """
    checksum = 0xEF
    for segment in segments:
        if segment:
            checksum = reduce(xor, segment, checksum)
    return checksum


def calculate_sha256(data: bytes) -> bytes:
    """
    Calculate the SHA256 hash of the given data.

    Args:
        data (bytes): The binary data.
    Returns:
        bytes: The calculated SHA256 hash.
    """
    return hashlib.sha256(data).digest()


def calculate_sha256_payload(data: bytearray, hash_appended: bool) -> memoryview:
    """
    Return a view of the data that should be covered by the SHA256 hash.
    When hash_appended is True, excludes the trailing 32-byte hash itself.
    When False, excludes only the 1-byte checksum trailer.

    Note: this must be called on the *final* state of data (after any checksum
    fix) so the hash covers the correct bytes.
    """
    view = memoryview(data)
    return view[:-SHA256_SIZE] if hash_appended else view[:-CHECKSUM_SIZE]


def _require(condition: bool, message: str = "Invalid binary file") -> None:
    if not condition:
        raise ValueError(message)


def _parse_segment(data: bytearray, offset: int) -> tuple[SegmentInfo, int]:
    _require(
        offset + SEGMENT_HEADER_SIZE <= len(data),
        f"Invalid binary file: segment header at offset {offset:#010x} exceeds file size",
    )
    load_address, size = struct.unpack_from(SEGMENT_HEADER_FMT, data, offset)

    segment_data_offset = offset + SEGMENT_HEADER_SIZE
    segment_data = get_segment_data(data, segment_data_offset, size)
    return (
        SegmentInfo(load_address, size, segment_data_offset, segment_data),
        segment_data_offset + size,
    )


def _parse_footer_offsets(
    data_length: int, hash_appended: bool
) -> tuple[int, int | None]:
    checksum_offset = data_length - (
        SHA256_SIZE + CHECKSUM_SIZE if hash_appended else CHECKSUM_SIZE
    )
    _require(
        checksum_offset >= 0, "Invalid binary file: file too small to contain a footer"
    )
    if hash_appended:
        sha256_offset = data_length - SHA256_SIZE
        _require(
            sha256_offset >= SHA256_SIZE,
            "Invalid binary file: file truncated, missing SHA256 trailer",
        )
    else:
        sha256_offset = None
    return checksum_offset, sha256_offset


def parse_image(data: bytearray) -> ParsedImage:
    _require(len(data) >= FILE_HEADER_SIZE + EXTENDED_FILE_HEADER_SIZE)

    view = memoryview(data)
    header = view[:FILE_HEADER_SIZE]
    extended_header = view[
        FILE_HEADER_SIZE : FILE_HEADER_SIZE + EXTENDED_FILE_HEADER_SIZE
    ]

    magic_number, num_segments, spi_flash_mode, spi_flash_size_freq, entry_point = (
        struct.unpack(FILE_HEADER_FMT, header)
    )
    _require(magic_number == ESP_MAGIC, f"Invalid magic number: {magic_number:#02x}")

    (
        wp_pin,
        drive_settings,
        chip_id,
        _,  # reserved byte
        min_chip_rev,
        max_chip_rev,
        hash_appended,
    ) = struct.unpack(EXT_HEADER_FMT, extended_header)

    spi_flash_mode_parsed: FlashMode | int
    try:
        spi_flash_mode_parsed = FlashMode(spi_flash_mode)
    except ValueError:
        spi_flash_mode_parsed = spi_flash_mode

    offset = FILE_HEADER_SIZE + EXTENDED_FILE_HEADER_SIZE
    segments = []
    for _ in range(num_segments):
        segment, offset = _parse_segment(data, offset)
        segments.append(segment)

    checksum_offset, sha256_offset = _parse_footer_offsets(len(data), hash_appended)

    return ParsedImage(
        magic_number,
        num_segments,
        spi_flash_mode_parsed,
        spi_flash_size_freq >> 4,
        spi_flash_size_freq & 0xF,
        entry_point,
        wp_pin,
        drive_settings,
        chip_id,
        min_chip_rev,
        max_chip_rev,
        hash_appended,
        segments,
        checksum_offset,
        sha256_offset,
    )


def _validate_footer(data: bytearray, parsed: ParsedImage) -> ValidationResult:
    calculated_checksum = calculate_checksum(
        [segment.data for segment in parsed.segments]
    )
    checksum_is_valid = data[parsed.checksum_offset] == calculated_checksum

    calculated_sha256: bytes | None = None
    sha256_is_valid = True
    if parsed.hash_appended and parsed.sha256_offset is not None:
        calculated_sha256 = calculate_sha256(
            bytes(calculate_sha256_payload(data, parsed.hash_appended))
        )
        sha256_is_valid = (
            bytes(data[parsed.sha256_offset : parsed.sha256_offset + SHA256_SIZE])
            == calculated_sha256
        )

    return ValidationResult(
        calculated_checksum, checksum_is_valid, calculated_sha256, sha256_is_valid
    )


def print_validation(label: str, value: int, calculated: int) -> None:
    print(f"{label}: {value:#04x}", end=" ")
    if value == calculated:
        print("(valid)")
    else:
        print(f"(invalid, calculated {calculated:#04x})")


def print_section_header(label: str) -> None:
    print()
    print(f"{label}:")
    print("=" * (len(label) + 1))


def print_title(label: str) -> None:
    print("*" * (len(label) + 4))
    print(f"* {label} *")
    print("*" * (len(label) + 4))


def print_info(parsed: ParsedImage, data: bytearray) -> None:
    print_section_header("ESP32 Binary Image Header")
    print(f"Magic Number: {parsed.magic_number:#04x}")
    print(f"Number of Segments: {parsed.num_segments}")
    print(
        f"SPI Flash: {FlashMode.to_str(parsed.spi_flash_mode)}, "
        f"{FlashSize.to_str(parsed.spi_flash_size)}, "
        f"{FlashFreq.to_str(parsed.spi_flash_freq)}"
    )
    print(f"Entry: {parsed.entry_point:#010x}")

    print_section_header("Extended File Header")
    print(f"WP Pin: {parsed.wp_pin:#04x}")
    print(f"Drive Settings: {parsed.drive_settings.hex()}")
    print(f"Chip ID: {parsed.chip_id:#06x}")
    print(f"Minimal Chip Revision: {parsed.min_chip_rev}")
    print(f"Maximal Chip Revision: {parsed.max_chip_rev}")
    print(f"Hash Appended: {parsed.hash_appended}")

    print_section_header("Segments")
    print("Segment   Length   Load addr   File offs  Memory types")
    print("-------  -------  ----------  ----------  ------------")
    for index, segment in enumerate(parsed.segments):
        memory_types = _get_memory_types(segment.load_address)
        print(
            f"{index:7d}  {segment.size:#07x}  {segment.load_address:#010x}"
            f"  {segment.file_offset:#010x}  {memory_types}"
        )

    validation = _validate_footer(data, parsed)

    print_section_header("Footer")
    print_validation(
        "Checksum", data[parsed.checksum_offset], validation.calculated_checksum
    )
    if (
        parsed.hash_appended
        and parsed.sha256_offset is not None
        and validation.calculated_sha256 is not None
    ):
        sha256_hash = bytes(
            data[parsed.sha256_offset : parsed.sha256_offset + SHA256_SIZE]
        )
        print(f"SHA256 Hash: {sha256_hash.hex()}", end=" ")
        if sha256_hash == validation.calculated_sha256:
            print("(valid)")
        else:
            print(f"(invalid, calculated {validation.calculated_sha256.hex()})")


def fix_image(parsed: ParsedImage, data: bytearray, binary_file: str) -> int:
    validation = _validate_footer(data, parsed)

    if validation.checksum_is_valid and validation.sha256_is_valid:
        print(
            f"Checksum is already valid, no need to fix ({validation.calculated_checksum:#04x})."
        )
        if (
            parsed.hash_appended
            and parsed.sha256_offset is not None
            and validation.calculated_sha256 is not None
        ):
            print(
                f"SHA256 hash is already valid, no need to fix ({validation.calculated_sha256.hex()})."
            )
        return 0

    if not validation.checksum_is_valid:
        old_checksum = data[parsed.checksum_offset]
        data[parsed.checksum_offset] = validation.calculated_checksum
        print(
            f"Checksum fixed: {old_checksum:#04x} -> {validation.calculated_checksum:#04x}."
        )

    if (
        parsed.hash_appended
        and not validation.sha256_is_valid
        and parsed.sha256_offset is not None
        and validation.calculated_sha256 is not None
    ):
        # Recalculate SHA256 over the final state of data (checksum already patched above).
        new_sha256 = calculate_sha256(bytes(calculate_sha256_payload(data, True)))
        old_sha256 = bytes(
            data[parsed.sha256_offset : parsed.sha256_offset + SHA256_SIZE]
        )
        data[parsed.sha256_offset : parsed.sha256_offset + SHA256_SIZE] = new_sha256
        print(f"SHA256 hash fixed: {old_sha256.hex()} -> {new_sha256.hex()}.")

    with open(binary_file, "r+b") as f:
        f.write(data)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=f"ESP32 Tool v{VERSION}")
    p.add_argument(
        "command",
        choices=["info", "fix"],
        help="Command to execute: 'info' to display image information, 'fix' to "
        "validate and fix checksum and hash if needed",
    )
    p.add_argument("file", help="Path to the ESP32 binary image file")
    args = p.parse_args()

    try:
        print_title(f"ESP32 Tool v{VERSION}")
        with open(args.file, "rb") as f:
            data = bytearray(f.read())

        print(f"Processing file: {args.file}")
        parsed = parse_image(data)
        print("File parsed successfully.")

        if args.command == "info":
            print_info(parsed, data)
        elif args.command == "fix":
            print_section_header("Fixing Image")
            sys.exit(fix_image(parsed, data, args.file))
    except (ValueError, IOError) as e:
        print(f"Error: {e}")
        sys.exit(1)
