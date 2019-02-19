use crate::encoding::{
    buffer::{BufferStride, EncodeBufferBuilder},
    pipeline::{EncoderPipeline, LayoutProp},
    resolver::IntoPipelineResolver,
    stream_encoder::{AnyEncoder, OpEncode, InstanceEncoder},
    PipelineResolver,
};
use fnv::FnvHashMap;
use hibitset::BitSetLike;
use log::warn;
use shred::Resources;
use std::sync::Arc;

/// Number of entities probed for batching at once.
/// Higher values require more memory,
/// lower values mean less virtual calls and setup code
const BATCH_ROUND_SIZE: usize = 1024;

/// Defines a query to the encoding system.
///
/// Every query has one “central” component `T` that must be present on entities of interest.
/// This allows to avoid unintentional multiple renders by many passes.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct EncodingQuery<R>
where
    R: PipelineResolver,
{
    pipeline_resolver: R,
    pipelines: Vec<EvaluatedPipeline>,
}


#[derive(Debug)]
struct EvaluatedPipeline {
    pipeline: EncoderPipeline,
    encoders: Vec<Arc<dyn AnyEncoder>>,
    batch_per_entity: Vec<u16>,
    batch_offsets: Vec<u32>,
    encoder_batch_writes: Vec<OpEncode>,
}

impl<R> EncodingQuery<R>
where
    R: PipelineResolver,
{
    /// Create new query for given component type.
    /// Must provide a way to resolve layouts from that component.
    ///
    /// The required `PipelineResolver` type is implemented for closures
    /// that extracts the shader handle from a component.
    /// ```rust,ignore
    /// let query = EncodingQuery::new(|component: &MyComponent| component.shader.clone());
    /// ```
    ///
    /// More complex `PipelineResolver` type can be implemented as needed,
    /// but then the implementer must ensure that the returned layout
    /// is memoized where applicable, because every returned layout instance
    /// will be encoded in a separate pipeline.
    pub fn new<I: IntoPipelineResolver<Resolver = R>>(pipeline_resolver: I) -> Self {
        EncodingQuery {
            pipeline_resolver: pipeline_resolver.into(),
        }
    }

    fn evaluate_batches<I>(&self, iter: I, res: &Resources) -> Vec<EvaluatedPipeline>
    where
        I: Iterator<Item = (EncoderPipeline, Vec<Arc<dyn AnyEncoder>>)>,
    {
        struct BatchInfo {
            count: u32,
            batch_id: u16,
        }

        // all temporary allocations are taken out from loop, so space can be reused
        let mut encoder_key_sizes: Vec<usize> = Vec::with_capacity(16);
        let mut encoder_key_writes = Vec::with_capacity(16);
        let mut batches_by_key: FnvHashMap<Vec<u8>, BatchInfo> = Default::default();
        let mut batch_keys_buffer: Vec<u8> = vec![];

        iter.map(|(pipeline, encoders)| {
            let entities_count = pipeline.entities_count();

            let mut encoder_batch_writes = Vec::with_capacity(16);
            let mut batch_per_entity: Vec<u16> = Vec::with_capacity(entities_count);
            let mut next_batch_id: u16 = 0;

            encoder_key_sizes.clear();
            encoder_key_sizes.extend(encoders.iter().map(|e| e.batch_key_size()));
            let key_sizes_sum = encoder_key_sizes.iter().sum();

            let mut bitset_iter = pipeline.bitset().iter();
            let total_batching_rounds = (entities_count + BATCH_ROUND_SIZE - 1) / BATCH_ROUND_SIZE;
            for round in 0..total_batching_rounds {
                let iter_count =
                    BATCH_ROUND_SIZE.min(entities_count - round * BATCH_ROUND_SIZE) as u32;

                batch_keys_buffer.resize(key_sizes_sum * (iter_count as usize), 0);
                encoder_key_writes.clear();
                encoder_key_writes.extend((0..iter_count).map(|index| OpEncode {
                    entity_id: bitset_iter.next().unwrap(),
                    write_index: index,
                }));

                for (encoder, mut stride) in encoders.iter().zip(BufferStride::from_sizes(
                    &mut batch_keys_buffer,
                    &encoder_key_sizes,
                )) {
                    unsafe {
                        // safe because we know that both `encoder_key_writes.len()`
                        // and `stride.contiguous_count` are both equal to iter_count
                        encoder.encode_batch_keys(&encoder_key_writes, res, &mut stride);
                    }
                }

                for (index, chunk) in batch_keys_buffer.chunks_exact(key_sizes_sum).enumerate() {
                    if let Some(batch) = batches_by_key.get_mut(chunk) {
                        batch_per_entity.push(batch.batch_id);
                        batch.count += 1;
                    } else {
                        batch_per_entity.push(next_batch_id);
                        batches_by_key.insert(
                            chunk.iter().cloned().collect(),
                            BatchInfo {
                                count: 1,
                                batch_id: next_batch_id,
                            },
                        );
                        encoder_batch_writes.push(OpEncode {
                            entity_id: encoder_key_writes[index].entity_id,
                            write_index: next_batch_id as u32,
                        });
                        next_batch_id += 1;
                    }
                }
            }
            assert!(
                bitset_iter.next().is_none(),
                "Entities iterator was not fully drained in batch collection phase"
            );

            // offsets are calculated in two phases, because hashmap iteration does not preserve insertion order
            let mut batch_offsets = vec![0; batches_by_key.len()];
            for batch in batches_by_key.values() {
                batch_offsets[batch.batch_id as usize] = batch.count;
            }
            batch_offsets.iter_mut().fold(0, |sum, entry| {
                let count = *entry;
                *entry = sum;
                sum + count
            });

            EvaluatedPipeline {
                pipeline,
                encoders,
                batch_per_entity,
                batch_offsets,
                encoder_batch_writes,
            }
        })
        .collect()
    }

    /// Evaluate the query on world, finding the right entities to encode.
    /// This steps determines the encoding pipelines, the encoders that will be used
    /// and computes the initial work of batching, which is necessary to retreive
    /// sizes of buffers that need to be externally allocated for encoding.
    ///
    /// This step can be cached, as long as the world was not modified
    /// between evaluation and encoding.
    pub fn encode(&mut self, res: &Resources) -> bool {
        // TODO: process only changed entities
        let encoder_storage = res.fetch::<EncoderStorage>();
        let iter = self
            .pipeline_resolver
            .resolve(res)
            .into_iter()
            .filter_map(|pipeline| {
                match encoder_storage.encoders_for_props(&pipeline.layout().props) {
                    Some(encoders) => Some((pipeline, encoders)),
                    None => {
                        warn!(
                            "Cannot find suitable encoders for layout {:?}",
                            pipeline.layout()
                        );
                        None
                    }
                }
            });

        self.pipelines = self.evaluate_batches(iter, res);

        // EvaluatedQuery {
        //     pipelines: self.evaluate_batches(iter, res),
        // }

        // changed?
        true
    }
}

impl EvaluatedQuery {
    /// Calculate the size requirement for the encoded buffer.
    pub fn ubo_size(&self) -> usize {
        self.pipelines.iter().map(|p| p.pipeline.ubo_size()).sum()
    }

    /// Perform encoding into an arbitrary byte buffer.
    /// The buffer slice must have length equal to the value returned from `ubo_size` method.
    pub fn encode(&self, res: &Resources, buffer: &mut [u8]) {
        assert_eq!(
            buffer.len(),
            self.ubo_size(),
            "The UBO buffer to encode has incorrect size"
        );

        let mut indices: Vec<u32> = vec![];
        let mut next_indices_per_batch: Vec<u32> = vec![];
        let mut ubo_offset: usize = 0;

        for evaluated in &self.pipelines {
            next_indices_per_batch.clear();
            next_indices_per_batch.extend(&evaluated.batch_offsets);

            indices.extend(evaluated.batch_per_entity.iter().map(|&batch_id| {
                let offset = next_indices_per_batch[batch_id as usize];
                next_indices_per_batch[batch_id as usize] += 1;
                offset
            }));

            let pipeline = &evaluated.pipeline;
            let layout = pipeline.layout();
            // TODO: split layout into batch_layout and nonbatch_layout;
            let batch_layout = layout;
            let nonbatch_layout = layout;

            let ubo_size = pipeline.ubo_size();

            let batch_buffer_builder = EncodeBufferBuilder::create(
                batch_layout,
                &mut buffer[ubo_offset..ubo_offset + ubo_size],
            );

            for encoder in &evaluated.encoders {
                unsafe {
                    encoder.encode_batch(
                        &evaluated.encoder_batch_writes,
                        res,
                        &batch_buffer_builder,
                    );
                }
            }

            let buffer_builder = EncodeBufferBuilder::create(
                nonbatch_layout,
                &mut buffer[ubo_offset..ubo_offset + ubo_size],
            );
            for encoder in &evaluated.encoders {
                unsafe {
                    encoder.encode(pipeline.bitset(), &indices, res, &buffer_builder);
                }
            }
            ubo_offset += ubo_size;
        }
    }
}
