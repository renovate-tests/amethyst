use super::{EncodingValue, IterableEncoding};
use crate::encoding::{
    properties::{EncPerInstanceProperties, EncProperties, EncodedDescriptor, PerInstanceValue},
    renderable::{BufferLayout, DescriptorsLayout},
};
use std::{
    cell::{RefCell, RefMut},
    marker::PhantomData,
};

/// Trait that defines the encoding buffer writing stragety for a specified
/// shader layout.
/// Every encoder must push exactly one value per iterated entity to the buffer.
pub trait EncodeBuffer<T>
where
    T: EncodingValue,
{
    /// Push encoded values to the buffer. Must be called exactly once for every entry
    /// in the provided encoding iterator.
    fn write(&mut self, data: T::Value, index: usize);
}

pub struct BufferStride<'a, T: 'static> {
    begin: *mut T,
    stride: isize,
    elem_count: isize,
    contiguous_count: usize,
    life: PhantomData<&'a mut T>,
}

impl<'a, T: 'static> BufferStride<'a, T> {
    /// Create a list of strides for given buffer defined by sizes of consecutive subtypes
    pub fn from_sizes<'s>(
        slice: &'a mut [T],
        sizes: &'s [usize],
    ) -> impl Iterator<Item = BufferStride<'a, T>> + 's {
        let stride: usize = sizes.iter().sum();
        assert!(
            stride > 0 && slice.len() % stride == 0,
            "Buffer size {} must be a multiple of layout stride {}",
            slice.len(),
            stride
        );
        let elem_count = slice.len() / stride;
        let mut_ptr = slice.as_mut_ptr();
        let mut begin_offset: usize = 0;
        sizes.iter().map(move |size| {
            let begin = unsafe { mut_ptr.offset(begin_offset as isize) };
            begin_offset += size;

            BufferStride {
                begin,
                stride: stride as isize,
                elem_count: elem_count as isize,
                contiguous_count: *size,
                life: PhantomData,
            }
        })
    }

    /// Create a list of 1-element wide strides for given buffer
    pub fn from_ones<'s>(
        slice: &'a mut [T],
        stride: usize,
    ) -> impl Iterator<Item = BufferStride<'a, T>> + 's {
        assert!(
            stride > 0 && slice.len() % stride == 0,
            "Buffer size {} must be a multiple of layout stride {}",
            slice.len(),
            stride
        );

        let elem_count = slice.len() / stride;
        let mut_ptr = slice.as_mut_ptr();
        (0..stride).map(move |offset| {
            let begin = unsafe { mut_ptr.offset(offset as isize) };
            BufferStride {
                begin,
                stride: stride as isize,
                elem_count: elem_count as isize,
                contiguous_count: 1,
                life: PhantomData,
            }
        })
    }

    /// Create a list of strides for given buffer matching all separate entries in the layout
    pub fn from_layout<'l>(
        slice: &'a mut [T],
        layout: &'l BufferLayout,
    ) -> impl Iterator<Item = BufferStride<'a, T>> + 'l {
        let stride: usize = layout.padded_size as usize;

        assert!(
            stride > 0 && slice.len() % stride == 0,
            "Buffer size {} must be a multiple of layout stride {}",
            slice.len(),
            stride
        );

        let elem_count = slice.len() / stride;
        let mut_ptr = slice.as_mut_ptr();

        // Let's assume that layout is well-formed and has no overlaps
        // TODO: this should be guaranteed by layout type itself
        layout.props.iter().map(move |layout_prop| {
            let begin = unsafe { mut_ptr.offset(layout_prop.absolute_offset as isize) };
            let size = layout_prop.prop.0.ubo_size();

            BufferStride {
                begin,
                stride: stride as isize,
                elem_count: elem_count as isize,
                contiguous_count: size,
                life: PhantomData,
            }
        })
    }

    pub fn contiguous_count(&self) -> usize {
        self.contiguous_count
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut [T] {
        debug_assert!(
            (idx as isize) < self.elem_count,
            "strided buffer out of bounds: idx: {}, count: {}",
            idx,
            self.elem_count
        );
        unsafe {
            let write_ptr = self.begin.offset(self.stride * idx as isize);
            std::slice::from_raw_parts_mut(write_ptr, self.contiguous_count)
        }
    }

    pub fn write_at(&mut self, idx: usize, data: &[T])
    where
        T: Copy,
    {
        self.get_mut(idx).copy_from_slice(data);
    }

    pub fn move_at(&mut self, idx: usize, data: impl Iterator<Item = T>) {
        let dst = self.get_mut(idx);
        for (i, src) in data.enumerate() {
            dst[i] = src;
        }
    }
}

/// A structure that allows writing encoded typed data into binary buffer
/// given the strides for every subtype.
pub struct BufferWriter<'a, 'b, T: EncodingValue + PerInstanceValue> {
    strides: Vec<RefMut<'b, BufferStride<'a, u8>>>,
    marker: PhantomData<T>,
}

impl<'a, 'b, T: EncodingValue + PerInstanceValue> BufferWriter<'a, 'b, T> {
    /// Create a typed buffer writer from set of buffer strides.
    fn new(strides: Vec<RefMut<'b, BufferStride<'a, u8>>>) -> Self {
        debug_assert_eq!(<T::Value as IterableEncoding>::num_descriptors(), 0);
        Self {
            strides,
            marker: PhantomData,
        }
    }
}

impl<'a, 'b, T: EncodingValue + PerInstanceValue> EncodeBuffer<T> for BufferWriter<'a, 'b, T> {
    fn write(&mut self, data: T::Value, index: usize) {
        data.for_each_buffer(|stride_idx: usize, bytes: &[u8]| {
            self.strides[stride_idx].write_at(index, bytes);
        });
    }
}

/// A structure that allows writing encoded typed data and descriptors into respective buffers.
pub struct BatchBufferWriter<'a, 'b, T: EncodingValue> {
    bin_strides: Vec<RefMut<'b, BufferStride<'a, u8>>>,
    desc_strides: Vec<RefMut<'b, BufferStride<'a, EncodedDescriptor>>>,
    marker: PhantomData<T>,
}

impl<'a, 'b, T: EncodingValue> BatchBufferWriter<'a, 'b, T> {
    /// Create a typed batch buffer writer from set of buffer strides.
    fn new(
        bin_strides: Vec<RefMut<'b, BufferStride<'a, u8>>>,
        desc_strides: Vec<RefMut<'b, BufferStride<'a, EncodedDescriptor>>>,
    ) -> Self {
        for desc_stride in &desc_strides {
            debug_assert_eq!(
                desc_stride.contiguous_count, 1,
                "Descriptor strides must always have only one element"
            );
        }
        debug_assert_eq!(
            desc_strides.len(),
            <T::Value as IterableEncoding>::num_descriptors(),
        );

        Self {
            bin_strides,
            desc_strides,
            marker: PhantomData,
        }
    }
}

impl<'a, 'b, T: EncodingValue> EncodeBuffer<T> for BatchBufferWriter<'a, 'b, T> {
    fn write(&mut self, data: T::Value, index: usize) {
        data.for_each_buffer(|stride_idx: usize, bytes: &[u8]| {
            self.bin_strides[stride_idx].write_at(index, bytes);
        });
        data.for_each_descriptor(|stride_idx: usize, descriptor: EncodedDescriptor| {
            self.desc_strides[stride_idx].move_at(index, std::iter::once(descriptor));
        });
    }
}

/// A builder for `BufferWriter`. Does the job of figuring out which strides should be written into in what order.
pub struct EncodeBufferBuilder<'a> {
    buffer_layout: BufferLayout,
    descs_layout: DescriptorsLayout,
    bin_strides: Vec<RefCell<BufferStride<'a, u8>>>,
    desc_strides: Vec<RefCell<BufferStride<'a, EncodedDescriptor>>>,
}

impl<'a> EncodeBufferBuilder<'a> {
    /// Create a `BufferWriteBuilder` with specific shader layout
    /// and buffer that's going to be written during encoding.
    pub fn create(
        buffer_layout: &BufferLayout,
        descs_layout: &DescriptorsLayout,
        raw_buffer: &'a mut [u8],
        desc_buffer: &'a mut [EncodedDescriptor],
    ) -> Self {
        let num_descriptors = descs_layout.props.len();
        Self {
            bin_strides: BufferStride::from_layout(raw_buffer, &buffer_layout)
                .map(RefCell::new)
                .collect(),
            desc_strides: BufferStride::from_ones(desc_buffer, num_descriptors)
                .map(RefCell::new)
                .collect(),
            buffer_layout: buffer_layout.clone(),
            descs_layout: descs_layout.clone(),
        }
    }

    /// Build a `BufferWriter` tailored for encoding of specific type.
    ///
    /// Works under an assumption that there is only one property of given name in a shader layout.
    pub fn build<'b, T: EncPerInstanceProperties>(
        &'b self,
    ) -> BufferWriter<'a, 'b, T::EncodedInstType> {
        let props_in_encoding_order = T::get_props();
        let stride_indices = props_in_encoding_order.map(|prop| {
            self.buffer_layout
                .props
                .iter()
                .position(|layout_prop| layout_prop.prop == prop)
                .expect("Trying to encode a prop that is not a part of provided buffer layout")
        });

        let bin_strides = stride_indices
            .map(|i| {
                self.bin_strides[i]
                    .try_borrow_mut()
                    .ok()
                    .expect("Trying to encode the same property multiple times")
            })
            .collect::<Vec<_>>();

        BufferWriter::new(bin_strides)
    }

    /// Build a `BufferWriter` tailored for encoding of specific type.
    ///
    /// Works under an assumption that there is only one property of given name in a shader layout.
    pub fn build_batch<'b, T: EncProperties>(
        &'b self,
    ) -> BatchBufferWriter<'a, 'b, T::EncodedType> {
        let bin_stride_indices = T::get_props().filter_map(|prop| {
            self.buffer_layout
                .props
                .iter()
                .position(|layout_prop| layout_prop.prop == prop)
        });

        let desc_stride_indices = T::get_props().filter_map(|prop| {
            self.descs_layout
                .props
                .iter()
                .position(|&layout_prop| layout_prop == prop)
        });

        let bin_strides = bin_stride_indices
            .map(|i| {
                self.bin_strides[i]
                    .try_borrow_mut()
                    .ok()
                    .expect("Trying to encode the same property multiple times")
            })
            .collect::<Vec<_>>();

        let desc_strides = desc_stride_indices
            .map(|i| {
                self.desc_strides[i]
                    .try_borrow_mut()
                    .ok()
                    .expect("Trying to encode the same property multiple times")
            })
            .collect::<Vec<_>>();

        debug_assert_eq!(
            bin_strides.len() + desc_strides.len(),
            T::get_props().count(),
            "Trying to encode a prop that is not a part of provided buffer or descriptors layout"
        );

        BatchBufferWriter::new(bin_strides, desc_strides)
    }
}
