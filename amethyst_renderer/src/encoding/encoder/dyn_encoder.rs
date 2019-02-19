use crate::encoding::{
    buffer::BufferStride, encoder::OpEncode, BatchEncoder, EncodeBufferBuilder, EncodedProp,
    GlobalsEncoder, InstanceEncoder,
};
use amethyst_core::specs::SystemData;
use shred::{ResourceId, Resources};
use std::{any::Any, marker::PhantomData, sync::Arc};

/// A list of dynamic encoders separated by their type
#[derive(Default, Debug, Clone)]
pub struct BunchOfEncoders {
    /// A list of dynamic globals encoders
    pub globals: Vec<Arc<dyn DynGlobalsEncoder>>,
    /// A list of dynamic batch encoders
    pub batch: Vec<Arc<dyn DynBatchEncoder>>,
    /// A list of dynamic instance encoders
    pub instance: Vec<Arc<dyn DynInstanceEncoder>>,
}

/// A dynamic systemdata that can be lazily fetched for the encoder to use during encoding.
///
/// As SystemData does not implement Any, it is impossible to ask for it's TypeId and implement it
/// using downcast, which would probably make it more obvious what's happening.
/// The closest thing that we can check (and really the only one that matters for correctness)
/// is it's list of declared read/write resources. That way the encoding system is guaranteed
/// to never read any resources that were not declared during it's registration.
pub struct LazyFetch<'a> {
    res: &'a Resources,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
}
impl<'a> LazyFetch<'a> {
    fn fetch<D: SystemData<'a>>(&self) -> D {
        assert_eq!(
            D::reads(),
            self.reads,
            "Lazily fetched resource reads does not match requested ones"
        );
        assert_eq!(
            D::writes(),
            self.writes,
            "Lazily fetched resource writes does not match requested ones"
        );
        D::fetch(&self.res)
    }
}

/// Dynamic type that can hold any encoder
pub trait DynEncoder: Any + Send + Sync + std::fmt::Debug {
    /// Tries to match this encoder agains a set of properties that need to be encoded.
    /// If the encoder was matched, the passed list is modified by removing the passed
    /// properties.
    ///
    /// Returns if the match was successful.
    fn try_match_props(&self, props: &mut Vec<EncodedProp>) -> bool;

    /// Fetch resources required for encoding
    fn lazy_fetch<'a>(&self, res: &'a Resources) -> LazyFetch<'a> {
        LazyFetch {
            res,
            reads: self.reads(),
            writes: self.writes(),
        }
    }
    /// reads of resources required for encoding
    fn reads(&self) -> Vec<ResourceId>;
    /// writes of resources required for encoding
    fn writes(&self) -> Vec<ResourceId>;
}

pub trait DynInstanceEncoder: DynEncoder {
    /// Run encoding operation of type-erased encoder
    ///
    /// # Safety
    ///
    /// * Caller must guarantee that the ops count matches the buffer length.
    unsafe fn encode(
        &self,
        ops: &Vec<OpEncode>,
        encoder_data: &LazyFetch<'_>,
        buffer_builder: &EncodeBufferBuilder<'_>,
    );
}

pub trait DynBatchEncoder: DynEncoder {
    /// Get the size of batch key type
    fn batch_key_size(&self) -> usize;

    /// Run batch key resolution for type-erased encoder
    ///
    /// # Safety
    ///
    /// * Caller must guarantee that the ops count matches the buffer length.
    unsafe fn encode_batch_keys(
        &self,
        ops: &Vec<OpEncode>,
        encoder_data: &LazyFetch<'_>,
        buffer: &mut BufferStride<'_, u8>,
    );

    /// Run encoding operation of type-erased encoder
    ///
    /// # Safety
    ///
    /// * Caller must guarantee that the ops count matches the buffer length.
    unsafe fn encode(
        &self,
        ops: &Vec<OpEncode>,
        encoder_data: &LazyFetch<'_>,
        buffer_builder: &EncodeBufferBuilder<'_>,
    );
}

pub trait DynGlobalsEncoder: DynEncoder {
    /// Run batch key resolution for type-erased encoder
    ///
    /// # Safety
    ///
    /// * Caller must guarantee that buffer length is exactly 1.
    unsafe fn encode(&self, encoder_data: &LazyFetch<'_>, buffer_builder: &EncodeBufferBuilder<'_>);
}

macro_rules! impl_dyn_encoder {
    ($($impl_struct:ident $base_encoder:ident),*) => {$(
        #[derive(Debug)]
        struct $impl_struct<T>(PhantomData<T>);

        impl<T> DynEncoder for $impl_struct<T>
        where
            T: for<'a> $base_encoder<'a>,
        {
            fn try_match_props(&self, encoded_props: &mut Vec<EncodedProp>) -> bool {
                let is_match = T::get_props().all(|prop| {
                    encoded_props
                        .iter()
                        .find(|&&enc_prop| prop == enc_prop)
                        .is_some()
                });

                if is_match {
                    // TODO: get rid of this unfortunate allocation.
                    // Cannot swap_remove items from the vec while iterating over it.
                    let mut indices = T::get_props()
                        .map(|prop| {
                            encoded_props
                                .iter()
                                .position(|&enc_prop| prop == enc_prop)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();

                    // Indices must be removed from largest to smallest,
                    // so the swaps are not going to end up replacing
                    // element that should be removed in next iterations.
                    indices.sort();
                    for index in indices.into_iter().rev() {
                        encoded_props.swap_remove(index);
                    }
                }
                is_match
            }

            fn reads(&self) -> Vec<ResourceId> {
                T::reads()
            }

            fn writes(&self) -> Vec<ResourceId> {
                T::writes()
            }
        }
    )*};
}
impl_dyn_encoder!(
    DynInstanceEncoderImpl InstanceEncoder,
    DynBatchEncoderImpl BatchEncoder,
    DynGlobalsEncoderImpl GlobalsEncoder
);

impl<T> DynInstanceEncoder for DynInstanceEncoderImpl<T>
where
    T: for<'a> InstanceEncoder<'a>,
{
    unsafe fn encode(
        &self,
        ops: &Vec<OpEncode>,
        encoder_data: &LazyFetch<'_>,
        buffer_builder: &EncodeBufferBuilder<'_>,
    ) {
        T::encode(ops, encoder_data.fetch(), buffer_builder);
    }
}

impl<T> DynBatchEncoder for DynBatchEncoderImpl<T>
where
    T: for<'a> BatchEncoder<'a>,
{
    fn batch_key_size(&self) -> usize {
        T::batch_key_size()
    }

    unsafe fn encode_batch_keys(
        &self,
        ops: &Vec<OpEncode>,
        encoder_data: &LazyFetch<'_>,
        buffer: &mut BufferStride<'_, u8>,
    ) {
        T::encode_batch_keys(ops, encoder_data.fetch(), buffer)
    }

    unsafe fn encode(
        &self,
        ops: &Vec<OpEncode>,
        encoder_data: &LazyFetch<'_>,
        buffer_builder: &EncodeBufferBuilder<'_>,
    ) {
        T::encode(ops, encoder_data.fetch(), buffer_builder);
    }
}

impl<T> DynGlobalsEncoder for DynGlobalsEncoderImpl<T>
where
    T: for<'a> GlobalsEncoder<'a>,
{
    unsafe fn encode(
        &self,
        encoder_data: &LazyFetch<'_>,
        buffer_builder: &EncodeBufferBuilder<'_>,
    ) {
        T::encode(encoder_data.fetch(), buffer_builder);
    }
}

pub(crate) fn into_dyn_instance<T: for<'a> InstanceEncoder<'a>>() -> impl DynInstanceEncoder {
    DynInstanceEncoderImpl::<T>(PhantomData)
}

pub(crate) fn into_dyn_batch<T: for<'a> BatchEncoder<'a>>() -> impl DynBatchEncoder {
    DynBatchEncoderImpl::<T>(PhantomData)
}

pub(crate) fn into_dyn_global<T: for<'a> GlobalsEncoder<'a>>() -> impl DynGlobalsEncoder {
    DynGlobalsEncoderImpl::<T>(PhantomData)
}
