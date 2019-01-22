use super::EncodingLayout;
use crate::encoding::pipeline::EncoderPipeline;
use amethyst_core::specs::{Component, Entity, SystemData};
use amethyst_core::specs::{Entities, Join, ReadStorage};
use fnv::FnvHashMap;
use shred::Resources;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait PipelinesResolver {
    fn resolve(&mut self, res: &Resources) -> Vec<EncoderPipeline>;
}

pub enum LayoutResolution {
    Skip,
    NewLayout {
        layout: EncodingLayout,
        batch: usize,
    },
    ReuseLayout {
        index: usize,
        batch: usize,
    },
}

/// Ability to resolve pipeline layout based on a component or optionally on other parts of the entity. Used in the first phase of world encoding.
pub trait LayoutResolver {
    type Component: Component;
    type LayoutCacheKey: Hash + Eq;
    type Batch: Hash + Eq;
    /// Resolve a pipeline from the world based on a component information
    fn resolve(
        &self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> Option<EncodingLayout>;
    fn batch(&self, component: &Self::Component, entity: &Entity, res: &Resources) -> Self::Batch;
    fn layout_key(
        &self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> Self::LayoutCacheKey;
}

pub struct FnLayoutResolver<C, B, K, ResFn, BatchFn, KeyFn>
where
    C: Component,
    B: Hash + Eq,
    K: Hash + Eq,
    ResFn: Fn(&C, &Entity, &Resources) -> Option<EncodingLayout>,
    BatchFn: Fn(&C, &Entity, &Resources) -> B,
    KeyFn: Fn(&C, &Entity, &Resources) -> K,
{
    res_fn: ResFn,
    batch_fn: BatchFn,
    key_fn: KeyFn,
    marker: PhantomData<(C, B, K)>,
}

impl<C, B, K, ResFn, BatchFn, KeyFn> FnLayoutResolver<C, B, K, ResFn, BatchFn, KeyFn>
where
    C: Component,
    B: Hash + Eq,
    K: Hash + Eq,
    ResFn: Fn(&C, &Entity, &Resources) -> Option<EncodingLayout>,
    BatchFn: Fn(&C, &Entity, &Resources) -> B,
    KeyFn: Fn(&C, &Entity, &Resources) -> K,
{
    /// Create new FnLayoutResolver using three provided methods
    pub fn new(res: ResFn, batch: BatchFn, key: KeyFn) -> Self {
        Self {
            res_fn: res,
            batch_fn: batch,
            key_fn: key,
            marker: PhantomData,
        }
    }
}
impl<C, B, K, ResFn, BatchFn, KeyFn> LayoutResolver
    for FnLayoutResolver<C, B, K, ResFn, BatchFn, KeyFn>
where
    C: Component,
    B: Hash + Eq,
    K: Hash + Eq,
    ResFn: Fn(&C, &Entity, &Resources) -> Option<EncodingLayout>,
    BatchFn: Fn(&C, &Entity, &Resources) -> B,
    KeyFn: Fn(&C, &Entity, &Resources) -> K,
{
    type Component = C;
    type LayoutCacheKey = K;
    type Batch = B;
    /// Resolve a pipeline from the world based on a component information
    fn resolve(&self, c: &Self::Component, e: &Entity, r: &Resources) -> Option<EncodingLayout> {
        (self.res_fn)(c, e, r)
    }
    fn batch(&self, c: &Self::Component, e: &Entity, r: &Resources) -> Self::Batch {
        (self.batch_fn)(c, e, r)
    }
    fn layout_key(&self, c: &Self::Component, e: &Entity, r: &Resources) -> Self::LayoutCacheKey {
        (self.key_fn)(c, e, r)
    }
}

pub trait CachedLayoutResolver {
    type Component: Component;
    fn clear(&mut self);
    fn resolve(
        &mut self,
        component: &Self::Component,
        entity: &Entity,
        res: &Resources,
    ) -> LayoutResolution;
}

impl<R: CachedLayoutResolver> PipelinesResolver for R {
    fn resolve(&mut self, res: &Resources) -> Vec<EncoderPipeline> {
        let mut pipelines: Vec<EncoderPipeline> = vec![];

        let component_storage = <ReadStorage<'_, R::Component>>::fetch(res);
        let entities = <Entities<'_>>::fetch(res);

        for (component, entity) in (&component_storage, &entities).join() {
            match R::resolve(self, component, &entity, res) {
                LayoutResolution::Skip => {}
                LayoutResolution::NewLayout { layout, batch } => {
                    let mut pipeline = EncoderPipeline::with_layout(layout);
                    pipeline.add_id(entity.id(), batch);
                    pipelines.push(pipeline);
                }
                LayoutResolution::ReuseLayout { index, batch } => {
                    let pipeline = pipelines
                        .get_mut(index)
                        .expect("ReuseLayout index is incorrect");
                    pipeline.add_id(entity.id(), batch);
                }
            };
        }
        R::clear(self);
        pipelines
    }
}

pub struct ConstLayoutResolver<T: Component> {
    marker: PhantomData<T>,
    layout: EncodingLayout,
    resolved_once: bool,
}

impl<T: Component> CachedLayoutResolver for ConstLayoutResolver<T> {
    type Component = T;
    fn clear(&mut self) {
        self.resolved_once = false;
    }
    fn resolve(&mut self, _: &T, _: &Entity, _: &Resources) -> LayoutResolution {
        if self.resolved_once {
            LayoutResolution::ReuseLayout { index: 0, batch: 0 }
        } else {
            self.resolved_once = true;
            LayoutResolution::NewLayout {
                layout: self.layout.clone(),
                batch: 0,
            }
        }
    }
}

pub struct ResolverCacheLayer<R: LayoutResolver> {
    inner: R,
    layout_cache: FnvHashMap<R::LayoutCacheKey, Option<usize>>,
    batch_index: FnvHashMap<R::Batch, usize>,
    next_layout: usize,
    next_batch: usize,
}

impl<R: LayoutResolver> ResolverCacheLayer<R> {
    pub fn new(inner: R) -> Self {
        ResolverCacheLayer {
            inner,
            layout_cache: Default::default(),
            batch_index: Default::default(),
            next_layout: 0,
            next_batch: 0,
        }
    }

    fn resolve_batch(
        &mut self,
        component: &R::Component,
        entity: &Entity,
        res: &Resources,
    ) -> usize {
        match self
            .batch_index
            .entry(self.inner.batch(component, entity, res))
        {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let idx = self.next_batch;
                self.next_batch += 1;
                entry.insert(idx);
                idx
            }
        }
    }
}

impl<R: LayoutResolver> CachedLayoutResolver for ResolverCacheLayer<R> {
    type Component = R::Component;
    fn clear(&mut self) {
        self.next_layout = 0;
        self.next_batch = 0;
        self.layout_cache.clear();
        self.batch_index.clear();
    }
    fn resolve(
        &mut self,
        component: &R::Component,
        entity: &Entity,
        res: &Resources,
    ) -> LayoutResolution {
        match self
            .layout_cache
            .entry(self.inner.layout_key(component, entity, res))
        {
            Entry::Occupied(entry) => {
                if let Some(index) = *entry.get() {
                    LayoutResolution::ReuseLayout {
                        index,
                        batch: self.resolve_batch(component, entity, res),
                    }
                } else {
                    LayoutResolution::Skip
                }
            }
            Entry::Vacant(entry) => {
                if let Some(layout) = self.inner.resolve(component, entity, res) {
                    entry.insert(Some(self.next_layout));
                    self.next_layout += 1;
                    LayoutResolution::NewLayout {
                        layout,
                        batch: self.resolve_batch(component, entity, res),
                    }
                } else {
                    entry.insert(None);
                    LayoutResolution::Skip
                }
            }
        }
    }
}

impl<T: Component> From<EncodingLayout> for ConstLayoutResolver<T> {
    fn from(layout: EncodingLayout) -> Self {
        ConstLayoutResolver {
            marker: PhantomData,
            layout,
            resolved_once: false,
        }
    }
}

impl<R: LayoutResolver> From<R> for ResolverCacheLayer<R> {
    fn from(inner: R) -> Self {
        ResolverCacheLayer::new(inner)
    }
}

pub trait IntoPipelinesResolver {
    type Resolver: PipelinesResolver;
    fn into(self) -> Self::Resolver;
}

impl<R> IntoPipelinesResolver for R
where
    R: LayoutResolver,
    R: Into<ResolverCacheLayer<R>>,
{
    type Resolver = ResolverCacheLayer<R>;
    fn into(self) -> Self::Resolver {
        self.into()
    }
}

impl<T: Component> IntoPipelinesResolver for ConstLayoutResolver<T> {
    type Resolver = ConstLayoutResolver<T>;
    fn into(self) -> Self::Resolver {
        self
    }
}
