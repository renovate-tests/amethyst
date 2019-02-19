mod buffer;
mod data;
mod encoders_impl;
mod properties;
mod properties_impl;
// mod query;
mod encoder;
mod render_group;
mod renderable;
mod resolver;

#[cfg(test)]
mod test;

pub use self::{
    buffer::{EncodeBuffer, EncodeBufferBuilder},
    data::{Encode, EncodingData, FetchedData},
    encoder::{
        BatchEncoder, BunchOfEncoders, DynBatchEncoder, DynGlobalsEncoder, DynInstanceEncoder,
        EncoderStorage, EncoderStorageBuilder, GlobalsEncoder, InstanceEncoder, LazyFetch,
    },
    properties::{
        EncMat4x4, EncMat4x4i, EncMat4x4u, EncPerInstanceProperties, EncProperties, EncProperty,
        EncTexture, EncVec2, EncVec2i, EncVec2u, EncVec4, EncVec4i, EncVec4u, EncodedDescriptor,
        EncodedProp, EncodingValue, IterableEncoding, ShaderInput, ShaderInputType,
    },
    properties_impl::*,
    render_group::*,
    renderable::{EncoderPipeline, EncodingLayout, Shader},
    resolver::{PipelineListResolver, SimplePipelineResolver},
};
// use self::query::{EncodingQuery, EvaluatedQuery},
