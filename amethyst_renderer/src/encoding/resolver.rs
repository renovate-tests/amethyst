use crate::encoding::EncoderPipeline;
use amethyst_core::specs::{Component, Entities, Entity, Join, ReadStorage, SystemData};
use fnv::FnvHashMap;
use gfx_hal::Backend;
use shred::Resources;
use std::{collections::hash_map::Entry, hash::Hash};

/// The most general pipeline resolver trait. Used during first stage of encoding to
/// retreive list of pipelines that will be rendered in the render pass.
pub trait PipelineListResolver: std::fmt::Debug + Send + Sync {
    /// resolver name
    fn name() -> &'static str;
    /// Resolve a list of pipelines from world
    fn resolve<B: Backend>(&mut self, res: &Resources) -> Vec<EncoderPipeline<B>>;
}

pub enum PipelineResolution<B: Backend> {
    Skip,
    NewPipeline { pipeline: EncoderPipeline<B> },
    KnownPipeline { index: usize },
}

/// Ability to resolve pipeline based on a component or optionally on other parts of the entity. Used in the first phase of world encoding.
pub trait SimplePipelineResolver: std::fmt::Debug + Send + Sync {
    /// Component that is iterated for pipeline resolution.
    type Component: Component;
    /// Type used for deduplication of pipelines
    type PipelineUniqKey: Hash + Eq + std::fmt::Debug + Send + Sync;

    /// resolver name
    fn name() -> &'static str;

    /// Resolve a pipeline from the world based on a component information
    fn resolve<B: Backend>(
        &self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> Option<EncoderPipeline<B>>;
    /// Get the unique key for resolved pipeline. Only one pipeline will be resolved for every unique key.
    fn pipeline_key(
        &self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> Self::PipelineUniqKey;
}

pub trait CachedPipelineResolver: std::fmt::Debug + Send + Sync {
    type Component: Component;
    /// resolver name
    fn name() -> &'static str;
    fn clear(&mut self);
    fn resolve<B: Backend>(
        &mut self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> PipelineResolution<B>;
}

impl<R: CachedPipelineResolver> PipelineListResolver for R {
    fn name() -> &'static str {
        R::name()
    }

    fn resolve<B: Backend>(&mut self, res: &Resources) -> Vec<EncoderPipeline<B>> {
        let mut pipelines: Vec<EncoderPipeline<B>> = vec![];

        let component_storage = <ReadStorage<'_, R::Component>>::fetch(res);
        let entities = <Entities<'_>>::fetch(res);

        for (component, entity) in (&component_storage, &entities).join() {
            match R::resolve(self, component, &entity, res) {
                PipelineResolution::Skip => {}
                PipelineResolution::NewPipeline { mut pipeline } => {
                    pipeline.add_id(entity.id());
                    pipelines.push(pipeline);
                }
                PipelineResolution::KnownPipeline { index } => {
                    let pipeline = pipelines
                        .get_mut(index)
                        .expect("KnownPipeline index is incorrect");
                    pipeline.add_id(entity.id());
                }
            };
        }
        R::clear(self);
        pipelines
    }
}

/// A pipeline resolution layer that provides caching required for simple resolvers to function property.
/// This layer adds a guarantee that for given cache key only single pipeline will ever be resolved.
#[derive(Debug)]
pub struct ResolverCacheLayer<R: SimplePipelineResolver> {
    inner: R,
    pipeline_index_cache: FnvHashMap<R::PipelineUniqKey, Option<usize>>,
    next_pipeline: usize,
}

impl<R: SimplePipelineResolver> ResolverCacheLayer<R> {
    pub fn new(inner: R) -> Self {
        ResolverCacheLayer {
            inner,
            pipeline_index_cache: Default::default(),
            next_pipeline: 0,
        }
    }
}

impl<R: SimplePipelineResolver> CachedPipelineResolver for ResolverCacheLayer<R> {
    type Component = R::Component;

    fn name() -> &'static str {
        R::name()
    }

    fn clear(&mut self) {
        self.next_pipeline = 0;
        self.pipeline_index_cache.clear();
    }
    fn resolve<B: Backend>(
        &mut self,
        component: &R::Component,
        entity: &Entity,
        res: &Resources,
    ) -> PipelineResolution<B> {
        match self
            .pipeline_index_cache
            .entry(self.inner.pipeline_key(component, entity, res))
        {
            Entry::Occupied(entry) => {
                if let Some(index) = *entry.get() {
                    PipelineResolution::KnownPipeline { index }
                } else {
                    PipelineResolution::Skip
                }
            }
            Entry::Vacant(entry) => {
                if let Some(pipeline) = self.inner.resolve(component, entity, res) {
                    entry.insert(Some(self.next_pipeline));
                    self.next_pipeline += 1;
                    PipelineResolution::NewPipeline { pipeline }
                } else {
                    entry.insert(None);
                    PipelineResolution::Skip
                }
            }
        }
    }
}
