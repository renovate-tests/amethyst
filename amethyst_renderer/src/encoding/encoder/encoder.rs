use crate::encoding::{
    buffer::BufferStride, EncPerInstanceProperties, EncProperties, EncodeBufferBuilder,
};
use amethyst_core::specs::SystemData;
use shred::ResourceId;

#[derive(Debug, Clone, Copy)]
pub struct OpEncode {
    pub entity_id: u32,
    pub write_index: u32,
}

/// A definition of a strategy to encode specified per-instance properties from world
/// by iteration over declared set of components. The encoder might also
/// use additional resources from the world.
///
/// Per-instance properties are limited to ones that can be uploaded as part of a buffer.
/// If you want to encode a texture, use `BatchEncoder` type.
///
/// Every encoder must push exactly one value per iterated entity to the buffer.
pub trait InstanceEncoder<'a>: Send + Sync + 'static + std::fmt::Debug {
    /// List of shader properties that this encoder encodes
    type Properties: EncPerInstanceProperties;
    /// SystemData that is used during encoding operations
    type SystemData: SystemData<'a>;

    /// Get a runtime list of shader properties encoded by this encoder
    fn get_props() -> <Self::Properties as EncProperties>::PropsIter {
        Self::Properties::get_props()
    }

    /// Resources with Read acess required for encoding
    fn reads() -> Vec<ResourceId> {
        <Self::SystemData as SystemData>::reads()
    }

    /// Resources with Write acess required for encoding
    fn writes() -> Vec<ResourceId> {
        <Self::SystemData as SystemData>::writes()
    }

    /// Do the encoding, filling the provided buffer with encoded data.
    ///
    ///
    /// # Safety
    /// * Caller must guarantee that the bitset count matches the buffer length.
    /// * Implementer must ensure that for every `bitset` entry
    ///   there is exactly one `buffer.write` call.
    unsafe fn encode(
        ops: &Vec<OpEncode>,
        data: Self::SystemData,
        buffer_builder: &EncodeBufferBuilder<'_>,
    );
}

/// A definition of a strategy to encode specified per-batch properties from world
/// by iteration over declared set of components. The encoder might also
/// use additional resources from the world.
///
/// Every encoder must push exactly one value per iterated entity to the buffer.
pub trait BatchEncoder<'a>: Send + Sync + 'static + std::fmt::Debug {
    /// List of shader properties that this encoder encodes
    type Properties: EncProperties;
    /// SystemData that is used during encoding operations
    type SystemData: SystemData<'a>;

    /// Get a runtime list of shader properties encoded by this encoder
    fn get_props() -> <Self::Properties as EncProperties>::PropsIter {
        Self::Properties::get_props()
    }

    /// Size of single batch key in bytes
    fn batch_key_size() -> usize;

    /// Resources with Read acess required for encoding
    fn reads() -> Vec<ResourceId> {
        <Self::SystemData as SystemData>::reads()
    }

    /// Resources with Write acess required for encoding
    fn writes() -> Vec<ResourceId> {
        <Self::SystemData as SystemData>::writes()
    }

    /// Do the encoding, filling the provided buffer with encoded data.
    ///
    ///
    /// # Safety
    /// * Caller must guarantee that the bitset count matches the buffer length.
    /// * Implementer must ensure that for every `bitset` entry
    ///   there is exactly one `buffer.write` call.
    unsafe fn encode(
        ops: &Vec<OpEncode>,
        data: Self::SystemData,
        buffer_builder: &EncodeBufferBuilder<'_>,
    );

    /// Run batch key resolution, filling batch key stride with batch keys
    /// for passed range of encoder write definitions.
    ///
    /// # Safety
    ///
    /// * Caller must guarantee that the batch_data count matches the batch_stride length.
    /// * Implementer must ensure that for every `batch_data` entry
    ///   there is exactly one `buffer.write` call.
    unsafe fn encode_batch_keys(
        ops: &Vec<OpEncode>,
        data: Self::SystemData,
        batch_stride: &mut BufferStride<'_, u8>,
    );
}

/// A definition of a strategy to encode specified global properties from world.
/// The encoder might also use additional resources from the world.
pub trait GlobalsEncoder<'a>: Send + Sync + 'static + std::fmt::Debug {
    /// List of shader properties that this encoder encodes
    type Properties: EncProperties;
    /// SystemData that is used during encoding operations
    type SystemData: SystemData<'a>;

    /// Get a runtime list of shader global properties encoded by this encoder
    fn get_props() -> <Self::Properties as EncProperties>::PropsIter {
        Self::Properties::get_props()
    }

    /// Resources with Read acess required for encoding
    fn reads() -> Vec<ResourceId> {
        <Self::SystemData as SystemData>::reads()
    }

    /// Resources with Write acess required for encoding
    fn writes() -> Vec<ResourceId> {
        <Self::SystemData as SystemData>::writes()
    }

    /// Do the encoding, filling the provided buffer with encoded data at index 0.
    fn encode(data: Self::SystemData, buffer_builder: &EncodeBufferBuilder<'_>);
}
