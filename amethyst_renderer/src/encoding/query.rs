use super::{
    buffer::EncodeBufferBuilder,
    pipeline::{EncoderPipeline, EncodingLayout, LayoutProp},
    stream_encoder::{AnyEncoder, StreamEncoder},
    PipelinesResolver,
};
use crate::encoding::resolver::IntoPipelinesResolver;
use amethyst_core::specs::{Entities, Join, ReadStorage, SystemData};
use fnv::FnvHashMap;
use log::warn;
use shred::Resources;
use std::marker::PhantomData;

/// Stores all registered encoders
pub struct EncoderStorage {
    available_encoders: Vec<Box<dyn AnyEncoder>>,
}

/// A builder type for `EncoderStorage`. Allows registering encoders.
#[derive(Default)]
pub struct EncoderStorageBuilder {
    encoders: Vec<Box<dyn AnyEncoder>>,
}

impl EncoderStorageBuilder {
    /// Register an encoder type
    pub fn with_encoder<E: for<'a> StreamEncoder<'a> + 'static>(mut self) -> Self {
        use super::stream_encoder::into_any;
        self.encoders.push(Box::new(into_any::<E>()));
        self
    }
    /// Finalize the list of registered encoders and retreive the resulting storage.
    pub fn build(self) -> EncoderStorage {
        EncoderStorage {
            available_encoders: self.encoders,
        }
    }
}

impl EncoderStorage {
    /// Create a new builder for this type
    pub fn build() -> EncoderStorageBuilder {
        EncoderStorageBuilder { encoders: vec![] }
    }

    /// Retreive the list of encoders that together encode given set of props without any overlaps.
    pub fn encoders_for_props(
        &self,
        layout_props: &Vec<LayoutProp>,
    ) -> Option<Vec<&Box<dyn AnyEncoder>>> {
        let mut matched_encoders = vec![];
        let mut not_found_props = layout_props.iter().map(|p| p.prop).collect::<Vec<_>>();
        for encoder in &self.available_encoders {
            if encoder.try_match_props(&mut not_found_props) {
                matched_encoders.push(encoder);
            }
        }

        if not_found_props.len() > 0 {
            None
        } else {
            Some(matched_encoders)
        }
    }
}

/// Defines a query to the encoding system.
///
/// Every query has one “central” component `T` that must be present on entities of interest.
/// This allows to avoid unintentional multiple renders by many passes.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct EncodingQuery<R>
where
    R: PipelinesResolver,
{
    pipeline_resolver: R,
}

/// A query that was already `evaluate`d. Holds a precomputed lists of entities matching a set of pipelines.
/// This must be recreated every time the central encoding component was inserted, updated or removed.
#[derive(Debug)]
pub struct EvaluatedQuery {
    pipelines: Vec<EncoderPipeline>,
}

impl<R> EncodingQuery<R>
where
    R: PipelinesResolver,
{
    /// Create new query for given component type.
    /// Must provide a way to resolve layouts from that component.
    ///
    /// The required `PipelinesResolver` type is implemented for closures
    /// that extracts the shader handle from a component.
    /// ```rust,ignore
    /// let query = EncodingQuery::new(|component: &MyComponent| component.shader.clone());
    /// ```
    ///
    /// More complex `PipelinesResolver` type can be implemented as needed,
    /// but then the implementer must ensure that the returned layout
    /// is memoized where applicable, because every returned layout instance
    /// will be encoded in a separate pipeline.
    pub fn new<I: IntoPipelinesResolver<Resolver = R>>(pipeline_resolver: I) -> Self {
        EncodingQuery {
            pipeline_resolver: pipeline_resolver.into(),
        }
    }

    /// Evaluate the query on world, finding the right entities to encode later.
    /// This step can be cached, as long as central entities list wes not modified
    /// between evaluation and encoding.
    pub fn evaluate(&mut self, res: &Resources) -> EvaluatedQuery {
        EvaluatedQuery {
            pipelines: self.pipeline_resolver.resolve(res),
        }
    }
}

impl EvaluatedQuery {
    /// Calculate the size requirement for the encoded buffer.
    pub fn ubo_size(&self) -> usize {
        self.pipelines
            .iter()
            .map(|pipeline| pipeline.ubo_size())
            .sum()
    }

    /// Perform encoding into an arbitrary byte buffer.
    /// The buffer slice must have length equal to the value returned from `ubo_size` method.
    pub fn encode(&self, res: &Resources, buffer: &mut [u8]) {
        assert_eq!(
            buffer.len(),
            self.ubo_size(),
            "The UBO buffer to encode has incorrect size"
        );

        let encoder_storage = res.fetch::<EncoderStorage>();

        let mut start = 0;
        for pipeline in &self.pipelines {
            let ref mut sub_buffer = buffer[start..start + pipeline.ubo_size()];
            dbg!(pipeline.ubo_size());
            start += pipeline.ubo_size();
            let indices = pipeline.indices();
            let layout = pipeline.layout();
            let bitset = pipeline.bitset();

            if let Some(encoders) = encoder_storage.encoders_for_props(&layout.props) {
                let buffer_builder = EncodeBufferBuilder::create(layout, sub_buffer);
                for encoder in encoders {
                    unsafe {
                        encoder.encode(bitset, &indices, res, &buffer_builder);
                    }
                }
            } else {
                warn!(
                    "Cannot find suitable encoders for shader props {:?}",
                    layout.props
                )
            }
        }
    }
}
