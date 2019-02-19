use crate::encoding::{
    encoder::{
        dyn_encoder::{into_dyn_batch, into_dyn_global, into_dyn_instance, DynEncoder},
        BunchOfEncoders,
    },
    renderable::BufferLayoutProp,
    BatchEncoder, GlobalsEncoder, InstanceEncoder,
};
use std::sync::Arc;

/// Stores all registered encoders
pub struct EncoderStorage {
    encoders: BunchOfEncoders,
}

/// A builder type for `EncoderStorage`. Allows registering encoders.
#[derive(Default)]
pub struct EncoderStorageBuilder {
    encoders: BunchOfEncoders,
}

impl EncoderStorageBuilder {
    /// Register an encoder type
    pub fn with_globals_encoder<E: for<'a> GlobalsEncoder<'a> + 'static + std::fmt::Debug>(
        mut self,
    ) -> Self {
        self.encoders.globals.push(Arc::new(into_dyn_global::<E>()));
        self
    }
    pub fn with_batch_encoder<E: for<'a> BatchEncoder<'a> + 'static + std::fmt::Debug>(
        mut self,
    ) -> Self {
        self.encoders.batch.push(Arc::new(into_dyn_batch::<E>()));
        self
    }
    pub fn with_instance_encoder<E: for<'a> InstanceEncoder<'a> + 'static + std::fmt::Debug>(
        mut self,
    ) -> Self {
        self.encoders
            .instance
            .push(Arc::new(into_dyn_instance::<E>()));
        self
    }

    /// Finalize the list of registered encoders and retreive the resulting storage.
    pub fn build(self) -> EncoderStorage {
        EncoderStorage {
            encoders: self.encoders,
        }
    }
}

impl EncoderStorage {
    /// Create a new builder for this type
    pub fn build() -> EncoderStorageBuilder {
        EncoderStorageBuilder {
            encoders: Default::default(),
        }
    }

    fn match_group<T: DynEncoder + ?Sized>(
        layout_props: &Vec<BufferLayoutProp>,
        encoders: &Vec<Arc<T>>,
    ) -> Option<Vec<Arc<T>>> {
        let mut matched_encoders = vec![];
        let mut not_found_props = layout_props.iter().map(|p| p.prop).collect::<Vec<_>>();
        for encoder in encoders {
            if encoder.try_match_props(&mut not_found_props) {
                matched_encoders.push(encoder.clone());
            }
        }
        if not_found_props.len() > 0 {
            None
        } else {
            Some(matched_encoders)
        }
    }

    /// Retreive the list of encoders that together encode given set of props without any overlaps.
    pub fn encoders_for_props(
        &self,
        layout_props: &Vec<BufferLayoutProp>,
    ) -> Option<BunchOfEncoders> {
        let globals = Self::match_group(layout_props, &self.encoders.globals);
        let batch = Self::match_group(layout_props, &self.encoders.batch);
        let instance = Self::match_group(layout_props, &self.encoders.instance);

        if let (Some(globals), Some(batch), Some(instance)) = (globals, batch, instance) {
            Some(BunchOfEncoders {
                globals,
                batch,
                instance,
            })
        } else {
            None
        }
    }
}
