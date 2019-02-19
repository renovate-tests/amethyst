use crate::encoding::{
    buffer::{BufferStride, EncodeBufferBuilder},
    data::{EncodingData, EncodingDef},
    encoder::{encoder::GlobalsEncoder, OpEncode},
    properties::{BufferEncoding, EncodingValue},
    BatchEncoder, EncPerInstanceProperties, EncProperties, EncodeBuffer, FetchedData,
    InstanceEncoder,
};
use amethyst_core::specs::SystemData;
use std::marker::PhantomData;

/// A marker struct used for ensuring that the encoding loop was called.
pub struct LoopResult(());

pub trait LoopingInstanceEncoder<'a>: 'static + Send + Sync + std::fmt::Debug {
    type Properties: EncPerInstanceProperties;
    type Components: EncodingDef + 'a;
    type SystemData: SystemData<'a>;

    fn encode<'j>(
        encode_loop: impl EncodeLoop<'a, 'j, Self::Components, Self::Properties>,
        system_data: Self::SystemData,
    ) -> LoopResult;
}

pub trait LoopingBatchEncoder<'a>: 'static + Send + Sync + std::fmt::Debug {
    type Properties: EncProperties;
    type Components: EncodingDef + 'a;
    type SystemData: SystemData<'a>;
    type BatchKey: BufferEncoding;

    fn encode<'j>(
        encode_loop: impl EncodeBatchLoop<'a, 'j, Self::Components, Self::Properties>,
        system_data: Self::SystemData,
    ) -> LoopResult;

    fn encode_batch_keys<'j>(
        encode_loop: impl EncodeKeyLoop<'a, 'j, Self::Components, Self::BatchKey>,
        system_data: Self::SystemData,
    ) -> LoopResult;
}

pub type EncodedGlobals<T: EncProperties> =
    <<T as EncProperties>::EncodedType as EncodingValue>::OptValue;
pub trait SimpleGlobalsEncoder<'a>: 'static + Send + Sync + std::fmt::Debug {
    type Properties: EncProperties;
    type SystemData: SystemData<'a>;

    fn encode<'j>(system_data: Self::SystemData) -> EncodedGlobals<Self::Properties>;
}

// TODO(frizi): use Into<InstanceEncoderImpl> instead, so more types can be implemented
impl<'a, T: LoopingInstanceEncoder<'a>> InstanceEncoder<'a> for T {
    type Properties = T::Properties;
    type SystemData = (
        <T::Components as EncodingData<'a>>::SystemData,
        T::SystemData,
    );

    unsafe fn encode(
        ops: &Vec<OpEncode>,
        (input_data, system_data): Self::SystemData,
        buffer_builder: &EncodeBufferBuilder<'_>,
    ) {
        let buffer = buffer_builder.build::<T::Properties>();
        let encode_loop = EncodeLoopImpl::new(ops, &input_data, buffer);
        T::encode(encode_loop, system_data);
    }
}

impl<'a, T: LoopingBatchEncoder<'a>> BatchEncoder<'a> for T {
    type Properties = T::Properties;
    type SystemData = (
        <T::Components as EncodingData<'a>>::SystemData,
        T::SystemData,
    );

    fn batch_key_size() -> usize {
        std::mem::size_of::<T::BatchKey>()
    }

    unsafe fn encode_batch_keys(
        ops: &Vec<OpEncode>,
        (input_data, system_data): Self::SystemData,
        batch_stride: &mut BufferStride<'_, u8>,
    ) {
        let encode_loop = EncodeKeyLoopImpl::new(ops, &input_data, batch_stride);
        T::encode_batch_keys(encode_loop, system_data);
    }

    unsafe fn encode(
        ops: &Vec<OpEncode>,
        (input_data, system_data): Self::SystemData,
        buffer_builder: &EncodeBufferBuilder<'_>,
    ) {
        let buffer = buffer_builder.build_batch::<T::Properties>();
        let encode_loop = EncodeBatchLoopImpl::new(ops, &input_data, buffer);
        T::encode(encode_loop, system_data);
    }
}

impl<'a, T: SimpleGlobalsEncoder<'a>> GlobalsEncoder<'a> for T {
    type Properties = T::Properties;
    type SystemData = T::SystemData;

    fn encode(data: Self::SystemData, buffer_builder: &EncodeBufferBuilder<'_>) {
        let mut buffer = buffer_builder.build_batch::<T::Properties>();
        let encoded = T::Properties::resolve(T::encode(data));
        buffer.write(encoded, 0);
    }
}

pub trait EncodeLoop<'a, 'j, I, O>
where
    I: EncodingDef,
    O: EncPerInstanceProperties,
    Self: Sized,
{
    fn run<F>(self, mapper: F) -> LoopResult
    where
        F: Fn(
            <<I as EncodingData<'a>>::FetchedData as FetchedData<'j>>::Ref,
        ) -> <O::EncodedInstType as EncodingValue>::OptValue;
}

pub trait EncodeBatchLoop<'a, 'j, I, O>
where
    I: EncodingDef,
    O: EncProperties,
    Self: Sized,
{
    fn run<F>(self, mapper: F) -> LoopResult
    where
        F: Fn(
            <<I as EncodingData<'a>>::FetchedData as FetchedData<'j>>::Ref,
        ) -> <O::EncodedType as EncodingValue>::OptValue;
}

pub trait EncodeKeyLoop<'a, 'j, I, O>
where
    I: EncodingDef,
    O: BufferEncoding,
    Self: Sized,
{
    fn run<F>(self, mapper: F) -> LoopResult
    where
        F: Fn(<<I as EncodingData<'a>>::FetchedData as FetchedData<'j>>::Ref) -> O;
}

struct EncodeBatchLoopImpl<'a, 'e, 'j, I, O, B>
where
    I: EncodingDef + 'a,
    O: EncProperties,
    B: EncodeBuffer<O::EncodedType>,
{
    marker: PhantomData<(I, O)>,
    ops: &'e Vec<OpEncode>,
    input_data: &'j <I as EncodingData<'a>>::SystemData,
    buffer: B,
}

impl<'a, 'e, 'j, I, O, B> EncodeBatchLoopImpl<'a, 'e, 'j, I, O, B>
where
    I: EncodingDef,
    O: EncProperties,
    B: EncodeBuffer<O::EncodedType>,
{
    fn new(
        ops: &'e Vec<OpEncode>,
        input_data: &'j <I as EncodingData<'a>>::SystemData,
        buffer: B,
    ) -> Self {
        Self {
            marker: PhantomData,
            ops,
            input_data,
            buffer,
        }
    }
}

struct EncodeLoopImpl<'a, 'e, 'j, I, O, B>
where
    I: EncodingDef + 'a,
    O: EncPerInstanceProperties,
    B: EncodeBuffer<O::EncodedInstType>,
{
    marker: PhantomData<(I, O)>,
    ops: &'e Vec<OpEncode>,
    input_data: &'j <I as EncodingData<'a>>::SystemData,
    buffer: B,
}

impl<'a, 'e, 'j, I, O, B> EncodeLoopImpl<'a, 'e, 'j, I, O, B>
where
    I: EncodingDef,
    O: EncPerInstanceProperties,
    B: EncodeBuffer<O::EncodedInstType>,
{
    fn new(
        ops: &'e Vec<OpEncode>,
        input_data: &'j <I as EncodingData<'a>>::SystemData,
        buffer: B,
    ) -> Self {
        Self {
            marker: PhantomData,
            ops,
            input_data,
            buffer,
        }
    }
}

pub struct EncodeKeyLoopImpl<'a, 'e, 'j, 'b, 's, I, O>
where
    I: EncodingDef + 'a,
    O: BufferEncoding,
{
    marker: PhantomData<(I, O)>,
    ops: &'e Vec<OpEncode>,
    input_data: &'j <I as EncodingData<'a>>::SystemData,
    stride: &'s mut BufferStride<'b, u8>,
}

impl<'a, 'e, 'j, 'b, 's, I, O> EncodeKeyLoopImpl<'a, 'e, 'j, 'b, 's, I, O>
where
    I: EncodingDef,
    O: BufferEncoding,
{
    fn new(
        ops: &'e Vec<OpEncode>,
        input_data: &'j <I as EncodingData<'a>>::SystemData,
        stride: &'s mut BufferStride<'b, u8>,
    ) -> Self {
        Self {
            marker: PhantomData,
            ops,
            input_data,
            stride,
        }
    }
}

impl<'a: 'j, 'e, 'j, I, O, B> EncodeLoop<'a, 'j, I, O> for EncodeLoopImpl<'a, 'e, 'j, I, O, B>
where
    I: EncodingDef,
    O: EncPerInstanceProperties,
    B: EncodeBuffer<O::EncodedInstType>,
{
    fn run<F>(mut self, mapper: F) -> LoopResult
    where
        F: Fn(
            <<I as EncodingData<'a>>::FetchedData as FetchedData<'j>>::Ref,
        ) -> <O::EncodedInstType as EncodingValue>::OptValue,
    {
        for &OpEncode {
            entity_id,
            write_index,
        } in self.ops
        {
            let components = <I as EncodingDef>::get_data(self.input_data, entity_id);
            self.buffer
                .write(O::resolve_inst(mapper(components)), write_index as usize);
        }

        LoopResult(())
    }
}

impl<'a: 'j, 'e, 'j, I, O, B> EncodeBatchLoop<'a, 'j, I, O>
    for EncodeBatchLoopImpl<'a, 'e, 'j, I, O, B>
where
    I: EncodingDef,
    O: EncProperties,
    B: EncodeBuffer<O::EncodedType>,
{
    fn run<F>(mut self, mapper: F) -> LoopResult
    where
        F: Fn(
            <<I as EncodingData<'a>>::FetchedData as FetchedData<'j>>::Ref,
        ) -> <O::EncodedType as EncodingValue>::OptValue,
    {
        for &OpEncode {
            entity_id,
            write_index,
        } in self.ops
        {
            let components = <I as EncodingDef>::get_data(self.input_data, entity_id);
            self.buffer
                .write(O::resolve(mapper(components)), write_index as usize);
        }

        LoopResult(())
    }
}

impl<'a: 'j, 'e, 'j, 'b, 's, I, O> EncodeKeyLoop<'a, 'j, I, O>
    for EncodeKeyLoopImpl<'a, 'e, 'j, 'b, 's, I, O>
where
    I: EncodingDef,
    O: BufferEncoding,
{
    fn run<F>(self, mapper: F) -> LoopResult
    where
        F: Fn(<<I as EncodingData<'a>>::FetchedData as FetchedData<'j>>::Ref) -> O,
    {
        for &OpEncode {
            entity_id,
            write_index,
        } in self.ops
        {
            let components = <I as EncodingDef>::get_data(self.input_data, entity_id);
            self.stride
                .write_at(write_index as usize, mapper(components).as_bytes());
        }
        LoopResult(())
    }
}
