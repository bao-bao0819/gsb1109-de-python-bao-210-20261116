from collections import UserDict
from dataclasses import dataclass
from typing import Iterable, NewType, cast

import numpy as np

from ..vlrs import VLR
from ..vlrs.known import WaveformPacketVlr

WavePacketDescriptorIndex = NewType("WavePacketDescriptorIndex", int)


class WavePacketDescriptorRecordId(int):
    RECORD_ID_OFFSET = 99

    @staticmethod
    def from_index(index: WavePacketDescriptorIndex) -> "WavePacketDescriptorRecordId":
        return WavePacketDescriptorRecordId(
            index + WavePacketDescriptorRecordId.RECORD_ID_OFFSET
        )


@dataclass(frozen=True)
class WaveformPacketDescriptor:
    bits_per_sample: int
    waveform_compression_type: int
    number_of_samples: int
    temporal_sample_spacing: int
    digitizer_gain: float
    digitizer_offset: float

    @property
    def sample_size_bytes(self) -> int | None:
        """Number of bytes used by one raw waveform sample, if byte-aligned."""
        if self.bits_per_sample % 8 != 0:
            return None
        return self.bits_per_sample // 8

    @property
    def packet_size_bytes(self) -> int | None:
        """Number of bytes a complete waveform packet occupies."""
        sample_size = self.sample_size_bytes
        if sample_size is None:
            return None
        return sample_size * self.number_of_samples

    def ensure_supported(self) -> None:
        if self.waveform_compression_type != 0:
            raise NotImplementedError(
                "Only uncompressed waveform packets (compression_type=0) are supported"
            )
        if self.bits_per_sample % 8 != 0:
            raise NotImplementedError(
                "Waveform sample size must be a multiple of 8 bits"
            )

    def dtype(self) -> np.dtype:
        if self.bits_per_sample % 8 != 0:
            raise NotImplementedError(
                "Waveform sample size must be a multiple of 8 bits"
            )
        match self.bits_per_sample:
            case 8:
                base_dtype = np.dtype(np.uint8)
            case 16:
                base_dtype = np.dtype(np.uint16)
            case 32:
                base_dtype = np.dtype(np.uint32)
            case 64:
                base_dtype = np.dtype(np.uint64)
            case _:
                raise NotImplementedError(
                    f"Unsupported waveform sample width: {self.bits_per_sample} bits"
                )
        return np.dtype([("waveform", (base_dtype, (self.number_of_samples,)))])


class WaveformPacketDescriptorRegistry(
    UserDict[WavePacketDescriptorRecordId, WaveformPacketDescriptor]
):
    """Registry of waveform packet descriptors by index."""

    bits_per_sample: int
    number_of_samples: int
    temporal_sample_spacing: int

    @classmethod
    def from_vlrs(
        cls,
        vlrs: Iterable[VLR],
        *,
        require_supported: bool = True,
    ) -> "WaveformPacketDescriptorRegistry":
        """Build a registry from waveform packet VLRs.

        With the default ``require_supported=True`` the historical strict
        behaviour is kept: the first unsupported descriptor or incompatible
        layout raises.

        Pass ``require_supported=False`` to keep descriptors laspy cannot
        decode (unknown compression, non byte-aligned samples) and mixed
        layouts, so callers can report per-point problems instead of failing
        the whole file.
        """
        registry = cls()
        waveform_vlrs: list[WaveformPacketVlr] = [
            cast(WaveformPacketVlr, vlr)
            for vlr in vlrs
            if vlr.record_id in WaveformPacketVlr.official_record_ids()
        ]
        if not waveform_vlrs:
            return registry

        waveform_vlrs.sort(key=lambda vlr: vlr.record_id)

        for vlr in waveform_vlrs:
            raw = vlr.parsed_record
            if raw is None:
                raise ValueError(
                    f"Waveform packet VLR with record ID {vlr.record_id} is not parsed"
                )
            descriptor = WaveformPacketDescriptor(
                bits_per_sample=raw.bits_per_sample,
                waveform_compression_type=raw.waveform_compression_type,
                number_of_samples=raw.number_of_samples,
                temporal_sample_spacing=raw.temporal_sample_spacing,
                digitizer_gain=raw.digitizer_gain,
                digitizer_offset=raw.digitizer_offset,
            )
            if require_supported:
                descriptor.ensure_supported()
            record_id = WavePacketDescriptorRecordId(vlr.record_id)
            registry.data[record_id] = descriptor

        if registry.data:
            first = next(iter(registry.data.values()))
            if require_supported:
                (
                    registry.bits_per_sample,
                    registry.number_of_samples,
                    registry.temporal_sample_spacing,
                ) = registry.ensure_supported()
            else:
                registry.bits_per_sample = first.bits_per_sample
                registry.number_of_samples = first.number_of_samples
                registry.temporal_sample_spacing = first.temporal_sample_spacing
        return registry

    def ensure_supported(self) -> tuple[int, int, int]:
        """Ensure that all descriptors in the registry are compatible.

        Currently, this means:
        - same bits_per_sample
        - same number_of_samples
        - same temporal_sample_spacing
        """
        bits_per_sample_set = {desc.bits_per_sample for desc in self.data.values()}
        if len(bits_per_sample_set) != 1:
            raise NotImplementedError(
                "All waveform packet descriptors must have the same bits_per_sample"
            )
        number_of_samples_set = {desc.number_of_samples for desc in self.data.values()}
        if len(number_of_samples_set) != 1:
            raise NotImplementedError(
                "All waveform packet descriptors must have the same number_of_samples"
            )
        temporal_sample_spacing_set = {
            desc.temporal_sample_spacing for desc in self.data.values()
        }
        if len(temporal_sample_spacing_set) != 1:
            raise NotImplementedError(
                "All waveform packet descriptors must have the same temporal_sample_spacing"
            )
        return (
            bits_per_sample_set.pop(),
            number_of_samples_set.pop(),
            temporal_sample_spacing_set.pop(),
        )

    def descriptor_for_index(
        self, descriptor_index: int
    ) -> WaveformPacketDescriptor | None:
        """Return the descriptor referenced by a point's ``wavepacket_index``.

        ``wavepacket_index`` 0 means that the point has no waveform packet and
        therefore has no descriptor.
        """
        if descriptor_index == 0:
            return None
        return self.data.get(
            WavePacketDescriptorRecordId.from_index(
                WavePacketDescriptorIndex(descriptor_index)
            )
        )

    def dtype(self) -> np.dtype | None:
        try:
            descriptor: WaveformPacketDescriptor = next(iter(self.data.values()))
        except StopIteration:
            return None
        return descriptor.dtype()
