use crate::{
    encoding::{
        buffer::{BufferStride, EncodeBufferBuilder},
        encoder::OpEncode,
        render_group::PsoDescBuilder,
        resolver::{PipelineListResolver, ResolverCacheLayer, SimplePipelineResolver},
        EncodedDescriptor,
    },
    mesh::Mesh,
    BunchOfEncoders, EncodedProp, LazyFetch,
};
use amethyst_assets::{Asset, AssetStorage, Handle, ProcessingState};
use amethyst_core::specs::{world::Index, Component, Entity, VecStorage};
use amethyst_error::Error;
use fnv::FnvHashMap;
use gfx_hal::{pso::GraphicsShaderSet, Backend};
use hibitset::{BitSet, BitSetLike};
use rendy::{
    command::RenderPassEncoder,
    factory::Factory,
    memory::Write,
    resource::buffer::{Buffer, UniformBuffer},
};
use shred::{Accessor, AccessorCow, DynamicSystemData, ReadExpect, ResourceId, Resources, System};
use std::marker::PhantomData;
use veclist::VecList;

/// Number of entities probed for batching at once.
/// Higher values require more memory,
/// lower values mean more virtual calls and setup overhead
const BATCH_ROUND_SIZE: usize = 1024;

/// Shader structure placeholder
/// TODO: use actual shaders
pub struct Shader {
    /// Temporary way to test against hardcoded layout
    pub mock_layout: EncodingLayout,
}

impl Asset for Shader {
    const NAME: &'static str = "Shader";
    type HandleStorage = VecStorage<Handle<Self>>;
    type Data = Self;
}
impl Into<Result<ProcessingState<Shader>, Error>> for Shader {
    fn into(self) -> Result<ProcessingState<Shader>, Error> {
        Ok(ProcessingState::Loaded(self))
    }
}

/// Placeholder descriptor set type
/// TODO: use rendy descriptor set
#[derive(Debug)]
struct DescriptorSet;

trait Renderable: Component + std::fmt::Debug {
    fn resolver() -> RenderableResolver<Self> {
        RenderableResolver::new()
    }
    fn name() -> &'static str;
    fn shader(&self) -> &Handle<Shader>;
    fn mesh(&self) -> Option<&Handle<Mesh>>;
}

#[derive(Default, Debug)]
struct InnerRenderableResolver<T: Renderable>(PhantomData<T>);
unsafe impl<T: Renderable> Send for InnerRenderableResolver<T> {}
unsafe impl<T: Renderable> Sync for InnerRenderableResolver<T> {}

impl<T: Renderable> SimplePipelineResolver for InnerRenderableResolver<T> {
    type Component = T;
    type PipelineUniqKey = (u32, Option<u32>);

    fn name() -> &'static str {
        T::name()
    }

    fn pipeline_key(
        &self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> Self::PipelineUniqKey {
        (
            component.shader().id(),
            component.mesh().map(|mesh| mesh.id()),
        )
    }

    fn resolve<B: Backend>(
        &self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> Option<EncoderPipeline<B>> {
        let mesh_storage = res.fetch::<AssetStorage<Mesh>>();
        let shader_storage = res.fetch::<AssetStorage<Shader>>();
        let mesh_res = component.mesh().and_then(|mesh| mesh_storage.get(mesh));
        let shader_res = shader_storage.get(component.shader());
        if let (Some(mesh), Some(shader)) = (mesh_res, shader_res) {
            // TODO: check if mesh conforms to shader layout
        }
        unimplemented!()
    }
}

#[derive(Debug)]
struct RenderableResolver<T: Renderable> {
    inner: ResolverCacheLayer<InnerRenderableResolver<T>>,
}

impl<T: Renderable> RenderableResolver<T> {
    pub fn new() -> Self {
        Self {
            inner: ResolverCacheLayer::new(InnerRenderableResolver(PhantomData)),
        }
    }
}

impl<T: Renderable> PipelineListResolver for RenderableResolver<T> {
    fn name() -> &'static str {
        T::name()
    }

    fn resolve<B: Backend>(&mut self, res: &Resources) -> Vec<EncoderPipeline<B>> {
        self.inner.resolve(res)
    }
}

/***************************
 * LAYOUT AND ENCODED DATA *
 ***************************/

struct EncodedRenderPass<B: Backend> {
    pipelines: VecList<EncoderPipeline<B>>,
}

// pipeline is per shader set
// pipelines are collected by iterating over renderables and grabbing shaders
#[derive(Debug)]
pub struct EncoderPipeline<B: Backend> {
    // gfx_pipeline: Pipeline,
    globals_buffer: Option<Buffer<B>>,
    batch_buffer: Option<Buffer<B>>,
    instances_buffer: Option<Buffer<B>>,
    globals_descriptors: Vec<EncodedDescriptor>,
    batch_descriptors: Vec<EncodedDescriptor>,
    layout: EncodingLayout,
    entities: BitSet,
    entities_count: u32,
    batch_per_index: Vec<u16>,
    batches: VecList<Batch<B>>,
    encoders: BunchOfEncoders,
    // TODO:
    // PSO: gfx_hal::pso::GraphicsPipelineDesc
}

impl<B: Backend> EncoderPipeline<B> {
    pub fn new(layout: EncodingLayout) -> Self {
        // TODO: add PSO definition to the structure
        Self {
            globals_buffer: None,
            batch_buffer: None,
            instances_buffer: None,
            globals_descriptors: Vec::new(),
            batch_descriptors: Vec::new(),
            layout,
            entities: BitSet::new(),
            entities_count: 0,
            batch_per_index: Vec::new(),
            batches: VecList::new(),
            encoders: Default::default(),
        }
    }

    fn shader_set(&self) -> GraphicsShaderSet<'_, B> {
        unimplemented!()
    }

    fn pipeline_layout(&self) -> &B::PipelineLayout {
        unimplemented!()
    }

    pub fn draw_inline(
        &mut self,
        encoder: &mut RenderPassEncoder<'_, B>,
        pso_desc_builder: &PsoDescBuilder<'_, B>,
    ) {
        let gfx_pipeline = pso_desc_builder.build(self.shader_set(), self.pipeline_layout());
    }

    fn entities_iter<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        (&self.entities).iter()
    }

    /// Add entity id to the pipeline.
    #[inline]
    pub fn add_id(&mut self, id: Index) {
        if !self.entities.add(id) {
            self.entities_count += 1;
        }
    }

    /// Remove entity id from the pipeline.
    #[inline]
    pub fn remove_id(&mut self, id: Index) {
        if self.entities.remove(id) {
            self.entities_count -= 1;
        }
    }

    /// Remove all associated entities from the pipeline.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.entities_count = 0;
    }
}

// struct EncodingBuffer {
//     buffer: Buffer,
//     descriptors: Vec<EncodedDescriptor>,
//     layout: EncodingLayout,
// }

// struct InstanceEncodingBuffer {
//     buffer: Buffer,
//     layout: EncodingLayout,
// }

// BATCH IS per DRAW CALL!
// batches are collected by encoding a pipeline into batch keys and deduplication
#[derive(Debug)]
struct Batch<B: Backend> {
    sets: Vec<DescriptorSet>,
    vertices: Vec<Buffer<B>>,
    vertex_count: u32, // vertices PER INSTANCE
    instanced_vertices: Vec<Buffer<B>>,
    instance_count: u32,
}

/// A set of all descriptors and buffer writes in an encoding.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct EncodingLayout {
    pub globals_buffer: BufferLayout,
    pub globals_descriptors: DescriptorsLayout,
    pub batch_buffer: BufferLayout,
    pub batch_descriptors: DescriptorsLayout,
    pub instances_buffer: BufferLayout,
}

impl EncodingLayout {
    /// Extract encoding layout from shader
    pub fn from_shader(shader: &Shader) -> Self {
        // TODO: cheating here, needs a real shader with proper
        // spirv-reflect data to implement that properly
        shader.mock_layout.clone()
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct DescriptorsLayout {
    pub props: Vec<EncodedProp>,
}

/// A set of shader properties at specific offsets.
/// The type should guarantee that all properties are non-overlapping.
/// TODO: do the actual validation at creation time.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct BufferLayout {
    /// A list of all properties at specific offset
    pub props: Vec<BufferLayoutProp>,
    /// Total number of bytes required for the block, including padding
    pub padded_size: u32,
}

// A single shader property at specific buffer offset
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct BufferLayoutProp {
    /// Name and type of the property. Determines the encoders to run.
    pub prop: EncodedProp,
    /// Offset in bytes from the start of the layout block.
    /// Instructs the `BufferWriter` where to put the encoded data.
    pub absolute_offset: u32,
}

struct EncodingTarget {
    renderable_id: [u8; 16],
    // TODO:
    // viewport: Option<EncodingViewport>,
}

// struct EncodingViewport {}

// single prop: descriptor_sets[].bindings[].block.members[]
//  name: .name
//  absolute_offset: .absolute_offset
//  type:
//    .type_description
//      .type_name -> non-empty = struct
//      .traits.numeric.matrix
//      .traits.numeric.matrix
//      .traits.array
//        .dims
//        .stride

/********************
 * ENCODING SYSTEMS *
 ********************/
// /// a system that fetches resources necessary for a set of dynamic encoders
// pub struct EncodingSystemWrap<'a, T: EncodingSystem<'a>>(T, PhantomData<&'a ()>);
// pub trait EncodingSystem<'a> {
//     type SystemData: DynamicSystemData<'a>;
//     fn encoders(&self) -> &BunchOfEncoders;
//     fn run(&mut self, encoders_data: BunchOfEncodersData<'a>, data: Self::SystemData);

//     fn system(self) -> EncodingSystemWrap<'a, Self>
//     where
//         Self: Sized,
//     {
//         EncodingSystemWrap(self, PhantomData)
//     }
// }

pub struct EncodersDataAccessor<T: Accessor> {
    encoders: BunchOfEncoders,
    inner: T,
}

macro_rules! chain_map_encoders {
    ($bunch:expr, $mapper:expr) => {{
        let bunch = $bunch;
        std::iter::empty()
            .chain(bunch.globals.iter().flat_map($mapper))
            .chain(bunch.batch.iter().flat_map($mapper))
            .chain(bunch.instance.iter().flat_map($mapper))
    }};
}

macro_rules! encoders_into_data {
    ($bunch:expr, $mapper:expr) => {{
        let bunch = $bunch;
        BunchOfEncodersData {
            globals: bunch.globals.iter().map($mapper).collect(),
            batch: bunch.batch.iter().map($mapper).collect(),
            instance: bunch.instance.iter().map($mapper).collect(),
        }
    }};
}

impl<T: Accessor> EncodersDataAccessor<T> {
    pub fn new(inner: T, encoders: BunchOfEncoders) -> Self {
        Self { inner, encoders }
    }
}

impl<T: Accessor> Accessor for EncodersDataAccessor<T> {
    fn try_new() -> Option<Self> {
        None
    }
    fn reads(&self) -> Vec<ResourceId> {
        chain_map_encoders!(&self.encoders, |e| e.reads())
            .chain(self.inner.reads())
            .collect()
    }
    fn writes(&self) -> Vec<ResourceId> {
        chain_map_encoders!(&self.encoders, |e| e.reads())
            .chain(self.inner.reads())
            .collect()
    }
}

struct BunchOfEncodersData<'a> {
    globals: Vec<LazyFetch<'a>>,
    batch: Vec<LazyFetch<'a>>,
    instance: Vec<LazyFetch<'a>>,
}

pub struct EncodingSystemData<'a, T: DynamicSystemData<'a>> {
    system_data: T,
    encoders_data: BunchOfEncodersData<'a>,
}

impl<'a, T: DynamicSystemData<'a>> DynamicSystemData<'a> for EncodingSystemData<'a, T> {
    type Accessor = EncodersDataAccessor<T::Accessor>;
    fn setup(access: &Self::Accessor, res: &mut Resources) {
        T::setup(&access.inner, res)
    }
    fn fetch(access: &Self::Accessor, res: &'a Resources) -> Self {
        EncodingSystemData {
            system_data: T::fetch(&access.inner, res),
            encoders_data: encoders_into_data!(&access.encoders, |e| e.lazy_fetch(res)),
        }
    }
}

// impl<'a, T: EncodingSystem<'a>> System<'a> for EncodingSystemWrap<'a, T> {
//     type SystemData = EncodingSystemData<'a, <T as EncodingSystem<'a>>::SystemData>;

//     fn run(&mut self, data: Self::SystemData) {
//         self.0.run(data.encoders_data, data.system_data);
//     }

//     fn accessor<'b>(&'b self) -> AccessorCow<'a, 'b, Self> {
//         // TODO: allow overriding inner accessor constructor
//         AccessorCow::Owned(EncodersDataAccessor::new(
//             Accessor::try_new().unwrap(),
//             self.0.encoders().clone(),
//         ))
//     }
// }

// find all pipelines
// pipeline has a few collections:
// encoders for globals
// encoders for batching
// encoders for instances

// construct systems for them

#[derive(Debug)]
struct BatchInfo {
    count: u32,
    batch_id: u16,
}

#[derive(Debug)]
pub struct PipelineEncodingSystem<B: Backend> {
    pipeline: EncoderPipeline<B>,
    encoder_key_sizes: Vec<usize>,
    encoder_key_writes: Vec<OpEncode>,
    encoder_batch_writes: Vec<OpEncode>,
    encoder_instance_writes: Option<Vec<OpEncode>>,
    batch_keys_buffer: Vec<u8>,
    batches_by_key: FnvHashMap<Vec<u8>, BatchInfo>,
    batch_per_entity: Vec<u16>,
    batch_offsets: Vec<u32>,
}

impl<B: Backend> PipelineEncodingSystem<B> {
    pub fn new(pipeline: EncoderPipeline<B>) -> Self {
        Self {
            pipeline,
            encoder_key_sizes: Vec::new(),
            encoder_key_writes: Vec::new(),
            encoder_batch_writes: Vec::new(),
            encoder_instance_writes: None,
            batch_keys_buffer: Vec::new(),
            batches_by_key: FnvHashMap::default(),
            batch_per_entity: Vec::new(),
            batch_offsets: Vec::new(),
        }
    }

    fn instance_writes<'a>(&'a mut self) -> impl Iterator<Item = OpEncode> + 'a {
        {
            self.batch_offsets.clear();
            self.batch_offsets
                .reserve(self.pipeline.batches.upper_bound());
            let mut total = 0;
            for batch_id in 0..self.pipeline.batches.upper_bound() {
                self.batch_offsets.push(total);
                total += self
                    .pipeline
                    .batches
                    .get(batch_id)
                    .map_or(0, |b| b.instance_count);
            }
        }

        let batch_offsets = &mut self.batch_offsets;
        self.batch_per_entity
            .iter()
            .zip(&self.pipeline.entities)
            .map(move |(&batch, entity_id)| {
                let idx = batch_offsets[batch as usize];
                batch_offsets[batch as usize] = idx + 1;
                OpEncode {
                    entity_id,
                    write_index: idx,
                }
            })
    }

    fn encoders(&self) -> &BunchOfEncoders {
        &self.pipeline.encoders
    }

    pub fn pipeline(&self) -> &EncoderPipeline<B> {
        &self.pipeline
    }
}

impl<'t, 'a, B: Backend> System<'a> for &'t mut PipelineEncodingSystem<B> {
    type SystemData = EncodingSystemData<'a, (ReadExpect<'a, Factory<B>>)>;

    fn accessor<'b>(&'b self) -> AccessorCow<'a, 'b, Self> {
        // TODO: allow overriding inner accessor constructor
        AccessorCow::Owned(EncodersDataAccessor::new(
            Accessor::try_new().unwrap(),
            self.pipeline.encoders.clone(),
        ))
    }

    fn run(
        &mut self,
        EncodingSystemData {
            encoders_data,
            system_data,
        }: Self::SystemData,
    ) {
        let factory = system_data;
        // 1. collect pipelinespipelinespipelines
        // done outside

        // 2. collect/update batching
        let entities_count = self.pipeline.entities_count as usize;
        let total_batching_rounds = (entities_count + BATCH_ROUND_SIZE - 1) / BATCH_ROUND_SIZE;

        self.encoder_batch_writes.clear();
        self.batches_by_key.clear();
        self.batch_per_entity.clear();
        self.batch_per_entity.reserve(entities_count);
        self.encoder_key_sizes.clear();
        self.encoder_key_sizes.extend(
            self.pipeline
                .encoders
                .batch
                .iter()
                .map(|e| e.batch_key_size()),
        );

        {
            let key_sizes_sum = self.encoder_key_sizes.iter().sum();
            let mut next_batch_id: u16 = 0;
            let mut entities_iter = self.pipeline.entities_iter();
            for round in 0..total_batching_rounds {
                let iter_count =
                    BATCH_ROUND_SIZE.min(entities_count - round * BATCH_ROUND_SIZE) as u32;

                self.batch_keys_buffer
                    .resize(key_sizes_sum * (iter_count as usize), 0);
                self.encoder_key_writes.clear();
                self.encoder_key_writes
                    .extend((0..iter_count).map(|index| OpEncode {
                        entity_id: entities_iter.next().unwrap(),
                        write_index: index,
                    }));

                let enc = &self.pipeline.encoders.batch;
                let batch_zip = enc.iter().zip(&encoders_data.batch);
                for ((encoder, data), mut stride) in batch_zip.zip(BufferStride::from_sizes(
                    &mut self.batch_keys_buffer,
                    &self.encoder_key_sizes,
                )) {
                    unsafe {
                        // safe because we know that both `encoder_key_writes.len()`
                        // and `stride.contiguous_count` are both equal to iter_count
                        debug_assert_eq!(self.encoder_key_writes.len(), iter_count as usize);
                        debug_assert_eq!(stride.contiguous_count(), iter_count as usize);
                        encoder.encode_batch_keys(&self.encoder_key_writes, &data, &mut stride);
                    }
                }

                for (index, chunk) in self
                    .batch_keys_buffer
                    .chunks_exact(key_sizes_sum)
                    .enumerate()
                {
                    if let Some(batch) = self.batches_by_key.get_mut(chunk) {
                        self.batch_per_entity.push(batch.batch_id);
                        batch.count += 1;
                    } else {
                        self.batch_per_entity.push(next_batch_id);
                        self.batches_by_key.insert(
                            chunk.iter().cloned().collect(),
                            BatchInfo {
                                count: 1,
                                batch_id: next_batch_id,
                            },
                        );
                        self.encoder_batch_writes.push(OpEncode {
                            entity_id: self.encoder_key_writes[index].entity_id,
                            write_index: next_batch_id as u32,
                        });
                        next_batch_id += 1;
                    }
                }
            }
            assert!(
                entities_iter.next().is_none(),
                "Entities iterator was not fully drained in batch collection phase"
            );
        }

        {
            let mut op_writes = self.encoder_instance_writes.take().unwrap();
            op_writes.reserve(entities_count);
            op_writes.extend(self.instance_writes());
            self.encoder_instance_writes.replace(op_writes);
        }

        // 3. prepare buffers (and possibly reallocate)
        let globals_buffer_size = self.pipeline.layout.globals_buffer.padded_size as u64;
        let batch_buffer_size = self.pipeline.layout.batch_buffer.padded_size as u64
            * self.encoder_batch_writes.len() as u64;
        let instances_buffer_size = self.pipeline.layout.instances_buffer.padded_size as u64
            * self.encoder_instance_writes.as_ref().unwrap().len() as u64;

        ensure_buffer(
            &factory,
            &mut self.pipeline.globals_buffer,
            globals_buffer_size,
            0,
        );

        ensure_buffer(
            &factory,
            &mut self.pipeline.batch_buffer,
            batch_buffer_size,
            batch_buffer_size / 2, // allocate extra 50% on top
        );

        ensure_buffer(
            &factory,
            &mut self.pipeline.instances_buffer,
            instances_buffer_size,
            instances_buffer_size / 2, // allocate extra 50% on top
        );

        if self
            .pipeline
            .globals_buffer
            .filter(|b| b.size() < globals_buffer_size)
            .is_none()
        {
            self.pipeline.globals_buffer.replace(
                factory
                    .create_buffer(1, globals_buffer_size, UniformBuffer)
                    .unwrap(),
            );
        }

        // 4. reencode dirty globals
        if let Some(buffer) = &mut self.pipeline.globals_buffer {
            let buffer_layout = &self.pipeline.layout.globals_buffer;
            let descs_layout = &self.pipeline.layout.globals_descriptors;
            let descriptors = &mut self.pipeline.globals_descriptors;
            let enc = &self.pipeline.encoders.globals;
            with_buffer_write(
                buffer,
                factory.device(),
                0..globals_buffer_size,
                |raw_buffer| {
                    let globals_buf = EncodeBufferBuilder::create(
                        buffer_layout,
                        descs_layout,
                        raw_buffer,
                        descriptors,
                    );
                    for (encoder, data) in enc.iter().zip(encoders_data.globals) {
                        unsafe {
                            encoder.encode(&data, &globals_buf);
                        }
                    }
                },
            )
            .unwrap();
        }

        // 5. reencode dirty batches
        if let Some(buffer) = &mut self.pipeline.batch_buffer {
            let buffer_layout = &self.pipeline.layout.batch_buffer;
            let descs_layout = &self.pipeline.layout.batch_descriptors;
            let descriptors = &mut self.pipeline.batch_descriptors;
            let enc = &self.pipeline.encoders.batch;
            let writes = &self.encoder_batch_writes;
            with_buffer_write(
                buffer,
                factory.device(),
                0..batch_buffer_size,
                |raw_buffer| {
                    let batch_buf = EncodeBufferBuilder::create(
                        buffer_layout,
                        descs_layout,
                        raw_buffer,
                        descriptors,
                    );
                    for (encoder, data) in enc.iter().zip(encoders_data.batch) {
                        unsafe {
                            encoder.encode(writes, &data, &batch_buf);
                        }
                    }
                },
            )
            .unwrap();
        }

        // 5. reencode dirty instances
        if let Some(buffer) = &mut self.pipeline.instances_buffer {
            let buffer_layout = &self.pipeline.layout.instances_buffer;
            let enc = &self.pipeline.encoders.instance;
            let writes = self.encoder_instance_writes.as_ref().unwrap();
            with_buffer_write(
                buffer,
                factory.device(),
                0..instances_buffer_size,
                |raw_buffer| {
                    let instances_buf = EncodeBufferBuilder::create(
                        buffer_layout,
                        &DescriptorsLayout { props: Vec::new() },
                        raw_buffer,
                        &mut [],
                    );
                    for (encoder, data) in enc.iter().zip(encoders_data.instance) {
                        unsafe {
                            encoder.encode(writes, &data, &instances_buf);
                        }
                    }
                },
            )
            .unwrap();
        }

        // buffers are written
    }
}

fn with_buffer_write<B: Backend, T>(
    buffer: &mut Buffer<B>,
    device: &impl gfx_hal::Device<B>,
    range: std::ops::Range<u64>,
    f: impl FnOnce(&mut [u8]) -> T,
) -> Result<T, failure::Error> {
    unsafe {
        let mut mapped = buffer.map(device, range.clone())?;
        let mut raw_buffer = mapped.write(device, range)?;
        Ok(f(raw_buffer.slice()))
    }
}

fn ensure_buffer<B: Backend>(
    factory: &Factory<B>,
    buffer: &mut Option<Buffer<B>>,
    min_size: u64,
    padding: u64,
) -> bool {
    if buffer.as_ref().filter(|b| b.size() < min_size).is_none() {
        buffer.replace(
            factory
                .create_buffer(1, min_size + padding, UniformBuffer)
                .unwrap(),
        );
        true
    } else {
        false
    }
}
