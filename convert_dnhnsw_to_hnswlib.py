#!/usr/bin/env python3
"""
Convert dnhnsw-fork binary index file to hnswlib-fork format

Key differences:
- dnhnsw: tableint = size_t, hnswlib: tableint = unsigned int (4 bytes)
- dnhnsw: LinkData.size = size_t, hnswlib: link count = unsigned short (2 bytes)
- dnhnsw: entry_idx (size_t), hnswlib: enterpoint_node_ (unsigned int)
- dnhnsw stores deleted_elements, hnswlib doesn't in saveIndex
"""

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple


UNSIGNED_INT_MAX = 0xFFFFFFFF
UNSIGNED_SHORT_MAX = 0xFFFF
UINT32_BYTES = 4
INT32_BYTES = 4
DOUBLE_BYTES = 8
FLOAT_BYTES = 4
USHORT_BYTES = 2
SIZE_T_BYTES = 8

TABLEINT_SIZE_HNSWLIB = 4
LINKLISTSIZEINT_SIZE = 4
LABELTYPE_SIZE = 8
PROGRESS_LOG_INTERVAL = 100000


@dataclass
class HNSWConfig:
    """HNSW configuration parameters"""
    M: int
    dim: int
    maxM0: int
    max_elements: int
    ef_construct: int
    ef_search: int
    seed: int
    max_multi_candidate: int


@dataclass
class HNSWMetadata:
    """HNSW index metadata"""
    cur_element_count: int
    maxlevel: int
    entry_idx: int


@dataclass
class ConversionContext:
    """Context for conversion from dnhnsw to hnswlib"""
    config: HNSWConfig
    metadata: HNSWMetadata
    vector_data: List[List[float]]
    labels: List[int]
    element_levels: List[int]
    level0_links: List[Tuple[int, List[int]]]
    higher_links: Dict[int, Dict[int, Tuple[int, List[int]]]]


class BinaryReader:
    """Binary reader for dnhnsw/hnswlib format"""

    def __init__(self, file: BinaryIO):
        self._file = file

    def read_size_t(self) -> int:
        return struct.unpack('<Q', self._file.read(SIZE_T_BYTES))[0]

    def read_uint32(self) -> int:
        return struct.unpack('<I', self._file.read(UINT32_BYTES))[0]

    def read_int32(self) -> int:
        return struct.unpack('<i', self._file.read(INT32_BYTES))[0]

    def read_double(self) -> float:
        return struct.unpack('<d', self._file.read(DOUBLE_BYTES))[0]

    def read_float(self) -> float:
        return struct.unpack('<f', self._file.read(FLOAT_BYTES))[0]

    def tell(self) -> int:
        return self._file.tell()


class BinaryWriter:
    """Binary writer for hnswlib format"""

    def __init__(self, file: BinaryIO):
        self._file = file

    def write_size_t(self, value: int) -> None:
        self._file.write(struct.pack('<Q', value))

    def write_uint32(self, value: int) -> None:
        self._file.write(struct.pack('<I', value))

    def write_int32(self, value: int) -> None:
        self._file.write(struct.pack('<i', value))

    def write_double(self, value: float) -> None:
        self._file.write(struct.pack('<d', value))

    def write_ushort(self, value: int) -> None:
        self._file.write(struct.pack('<H', value))

    def write_float(self, value: float) -> None:
        self._file.write(struct.pack('<f', value))

    def write_padding(self, num_bytes: int) -> None:
        self._file.write(b'\x00' * num_bytes)

    def tell(self) -> int:
        return self._file.tell()


class DNHNSWReader:
    """Reader for dnhnsw binary format"""

    def __init__(self, file_path: Path):
        self._file_path = file_path

    def read(self) -> ConversionContext:
        with open(self._file_path, 'rb') as f:
            reader = BinaryReader(f)

            print("\nReading dnhnsw file header...")
            config = self._read_config(reader)
            self._print_config(config)

            metadata = self._read_metadata(reader)
            self._print_metadata(metadata)

            vector_data = self._read_vectors(reader, metadata.cur_element_count, config.dim)
            labels = self._read_labels(reader, metadata.cur_element_count)
            element_levels = self._read_element_levels(reader, metadata.cur_element_count)

            self._skip_deleted_elements(reader)

            level0_links = self._read_level0_links(reader, metadata.cur_element_count)
            higher_links = self._read_higher_level_links(reader, metadata.cur_element_count, element_levels)

            print(f"Finished reading dnhnsw file at position: {reader.tell()}")

            return ConversionContext(
                config=config,
                metadata=metadata,
                vector_data=vector_data,
                labels=labels,
                element_levels=element_levels,
                level0_links=level0_links,
                higher_links=higher_links
            )

    @staticmethod
    def _read_config(reader: BinaryReader) -> HNSWConfig:
        return HNSWConfig(
            M=reader.read_size_t(),
            dim=reader.read_size_t(),
            maxM0=reader.read_size_t(),
            max_elements=reader.read_size_t(),
            ef_construct=reader.read_size_t(),
            ef_search=reader.read_size_t(),
            seed=reader.read_size_t(),
            max_multi_candidate=reader.read_size_t()
        )

    @staticmethod
    def _read_metadata(reader: BinaryReader) -> HNSWMetadata:
        return HNSWMetadata(
            cur_element_count=reader.read_size_t(),
            maxlevel=reader.read_int32(),
            entry_idx=reader.read_size_t()
        )

    @staticmethod
    def _read_vectors(reader: BinaryReader, count: int, dim: int) -> List[List[float]]:
        print("Reading vector data...")
        vectors = []
        for i in range(count):
            vector = [reader.read_float() for _ in range(dim)]
            vectors.append(vector)
            if (i + 1) % PROGRESS_LOG_INTERVAL == 0:
                print(f"  Read {i + 1}/{count} vectors...")
        return vectors

    @staticmethod
    def _read_labels(reader: BinaryReader, count: int) -> List[int]:
        print("Reading labels...")
        return [reader.read_size_t() for _ in range(count)]

    @staticmethod
    def _read_element_levels(reader: BinaryReader, count: int) -> List[int]:
        print("Reading element levels...")
        return [reader.read_int32() for _ in range(count)]

    @staticmethod
    def _skip_deleted_elements(reader: BinaryReader) -> None:
        print("Reading deleted elements...")
        deleted_size = reader.read_size_t()
        for _ in range(deleted_size):
            reader.read_size_t()
        print(f"Skipped {deleted_size} deleted elements")

    @staticmethod
    def _read_level0_links(reader: BinaryReader, count: int) -> List[Tuple[int, List[int]]]:
        print("Reading Level 0 links...")
        links_list = []
        for i in range(count):
            link_size = reader.read_size_t()
            links = [DNHNSWReader._convert_to_uint32(reader.read_size_t(), i) for _ in range(link_size)]
            links_list.append((link_size, links))
            if (i + 1) % PROGRESS_LOG_INTERVAL == 0:
                print(f"  Read {i + 1}/{count} level0 links...")
        return links_list

    @staticmethod
    def _read_higher_level_links(reader: BinaryReader, count: int, element_levels: List[int]) -> Dict[int, Dict[int, Tuple[int, List[int]]]]:
        print("Reading higher level links...")
        higher_links = {}
        for i in range(count):
            if element_levels[i] > 0:
                higher_links[i] = {}
                for level in range(1, element_levels[i] + 1):
                    link_size = reader.read_size_t()
                    links = [DNHNSWReader._convert_to_uint32(reader.read_size_t(), i, level) for _ in range(link_size)]
                    higher_links[i][level] = (link_size, links)
        return higher_links

    @staticmethod
    def _convert_to_uint32(value: int, element_idx: int, level: int = 0) -> int:
        return DNHNSWReader._check_overflow(
            value, UNSIGNED_INT_MAX, element_idx, level, "link", "unsigned int"
        )

    @staticmethod
    def _check_overflow(value: int, max_value: int, element_idx: int, level: int,
                       value_type: str, range_type: str) -> int:
        if value > max_value:
            level_str = f" level {level}" if level > 0 else ""
            print(f"WARNING: {value_type} {value} at element {element_idx}{level_str} exceeds {range_type} range")
            return value & max_value
        return value

    @staticmethod
    def _print_config(config: HNSWConfig) -> None:
        print(f"Config: M={config.M}, dim={config.dim}, maxM0={config.maxM0}, "
              f"max_elements={config.max_elements}, ef_construct={config.ef_construct}")

    @staticmethod
    def _print_metadata(metadata: HNSWMetadata) -> None:
        print(f"Elements: {metadata.cur_element_count:,}, maxlevel: {metadata.maxlevel}, "
              f"entry_idx: {metadata.entry_idx:,}")


class HNSWLibWriter:
    """Writer for hnswlib binary format"""

    def __init__(self, file_path: Path):
        self._file_path = file_path

    def write(self, context: ConversionContext) -> None:
        print("\nWriting hnswlib file...")

        params = self._calculate_parameters(context.config)

        with open(self._file_path, 'wb') as f:
            writer = BinaryWriter(f)

            self._write_header(writer, context, params)
            self._write_level0_data(writer, context, params)
            self._write_link_lists(writer, context, params)

            print(f"Finished writing hnswlib file at position: {writer.tell()}")

    @staticmethod
    def _calculate_parameters(config: HNSWConfig) -> Dict[str, float]:
        data_size = config.dim * FLOAT_BYTES
        size_links_level0 = config.maxM0 * TABLEINT_SIZE_HNSWLIB + LINKLISTSIZEINT_SIZE
        size_data_per_element = size_links_level0 + data_size + LABELTYPE_SIZE
        size_links_per_element = config.M * TABLEINT_SIZE_HNSWLIB + LINKLISTSIZEINT_SIZE
        mult = 1.0 / math.log(float(config.M))

        params = {
            'size_data_per_element': size_data_per_element,
            'size_links_per_element': size_links_per_element,
            'size_links_level0': size_links_level0,
            'data_size': data_size,
            'mult': mult,
            'offsetLevel0': 0,
            'offsetData': size_links_level0,
            'label_offset': size_links_level0 + data_size
        }

        print(f"size_data_per_element: {params['size_data_per_element']}")
        print(f"size_links_per_element: {params['size_links_per_element']}")
        print(f"mult: {params['mult']}")

        return params

    @staticmethod
    def _write_header(writer: BinaryWriter, context: ConversionContext, params: Dict[str, float]) -> None:
        print("Writing header...")
        config = context.config
        metadata = context.metadata

        enterpoint_node = HNSWLibWriter._convert_entry_idx(metadata.entry_idx)

        writer.write_size_t(int(params['offsetLevel0']))
        writer.write_size_t(config.max_elements)
        writer.write_size_t(metadata.cur_element_count)
        writer.write_size_t(int(params['size_data_per_element']))
        writer.write_size_t(int(params['label_offset']))
        writer.write_size_t(int(params['offsetData']))
        writer.write_int32(metadata.maxlevel)
        writer.write_uint32(enterpoint_node)
        writer.write_size_t(config.M)
        writer.write_size_t(config.maxM0)
        writer.write_size_t(config.M)
        writer.write_double(params['mult'])
        writer.write_size_t(config.ef_construct)

    @staticmethod
    def _write_level0_data(writer: BinaryWriter, context: ConversionContext, params: Dict[str, float]) -> None:
        print("Writing data_level0_memory_...")

        for i in range(context.metadata.cur_element_count):
            link_count, links = context.level0_links[i]

            link_count = HNSWLibWriter._convert_to_ushort(link_count, i)
            writer.write_ushort(link_count)
            writer.write_padding(2)

            HNSWLibWriter._write_padded_links(writer, links, context.config.maxM0)

            for value in context.vector_data[i]:
                writer.write_float(value)

            writer.write_size_t(context.labels[i])

    @staticmethod
    def _write_link_lists(writer: BinaryWriter, context: ConversionContext, params: Dict[str, float]) -> None:
        print("Writing linkLists...")

        for i in range(context.metadata.cur_element_count):
            if context.element_levels[i] > 0:
                link_list_size = int(params['size_links_per_element'] * context.element_levels[i])
                writer.write_uint32(link_list_size)

                for level in range(1, context.element_levels[i] + 1):
                    link_count, links = context.higher_links[i][level]

                    link_count = HNSWLibWriter._convert_to_ushort(link_count, i, level)
                    writer.write_ushort(link_count)
                    writer.write_padding(2)

                    HNSWLibWriter._write_padded_links(writer, links, context.config.M)
            else:
                writer.write_uint32(0)

    @staticmethod
    def _write_padded_links(writer: BinaryWriter, links: List[int], max_count: int) -> None:
        for j in range(max_count):
            value = links[j] if j < len(links) else 0
            writer.write_uint32(value)

    @staticmethod
    def _convert_entry_idx(entry_idx: int) -> int:
        return HNSWLibWriter._check_overflow(entry_idx, UNSIGNED_INT_MAX, 0, 0, "entry_idx", "unsigned int")

    @staticmethod
    def _convert_to_ushort(value: int, element_idx: int, level: int = 0) -> int:
        return HNSWLibWriter._check_overflow(value, UNSIGNED_SHORT_MAX, element_idx, level, "link_count", "unsigned short")

    @staticmethod
    def _check_overflow(value: int, max_value: int, element_idx: int, level: int,
                       value_type: str, range_type: str) -> int:
        if value > max_value:
            level_str = f" level {level}" if level > 0 else ""
            element_str = f" at element {element_idx}" if element_idx > 0 else ""
            print(f"WARNING: {value_type} {value}{element_str}{level_str} exceeds {range_type} range")
            return max_value
        return value


class DNHNSWToHNSWLibConverter:
    """Converter from dnhnsw to hnswlib format"""

    def __init__(self, input_path: Path, output_path: Path):
        self._input_path = input_path
        self._output_path = output_path

    def convert(self) -> None:
        reader = DNHNSWReader(self._input_path)
        context = reader.read()

        writer = HNSWLibWriter(self._output_path)
        writer.write(context)

        self._print_summary()

    def _print_summary(self) -> None:
        print("\nConversion complete!")
        print(f"Input: {self._input_path}")
        print(f"Output: {self._output_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_dnhnsw_to_hnswlib.py <input_dnhnsw.bin> <output_hnswlib.bin>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        converter = DNHNSWToHNSWLibConverter(input_path, output_path)
        converter.convert()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
