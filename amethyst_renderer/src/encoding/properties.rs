use crate::Texture;
use amethyst_assets::Handle;
use std::iter::{empty, once, Chain, Empty, Once};

/// A wrapper type for returned descriptor writes
///
/// This type is currently a mock to allow encoding
/// handles for gpu resources, as there are no real descriptors yet
/// TODO: use real descriptors once rendy lands
#[derive(Debug)]
pub enum EncodedDescriptor {
    /// Descriptor with texture binding
    Texture(Handle<Texture>),
}

/// Marker trait for values that can be encoded in per-instance encoders.
/// Required to prevent scenarios where descriptors are encoded and later ignored.
pub trait PerInstanceValue {}
impl<T> PerInstanceValue for T
where
    T: ShaderInputType,
    T::Repr: BufferEncoding,
{
}

/// Marker trait for encoder properties that can be declared in per-instance encoders.
pub trait EncPerInstanceProperties: EncProperties {
    /// A value that is the result of an per-instance encoding.
    type EncodedInstType: EncodingValue + PerInstanceValue;
    /// A version of EncodingValue::resolve that works on local `PerInstanceValue` version
    fn resolve_inst(
        optional: <Self::EncodedInstType as EncodingValue>::OptValue,
    ) -> <Self::EncodedInstType as EncodingValue>::Value;
}
impl<T> EncPerInstanceProperties for T
where
    T: EncProperties,
    T::EncodedType: PerInstanceValue,
{
    type EncodedInstType = T::EncodedType;

    fn resolve_inst(
        optional: <Self::EncodedInstType as EncodingValue>::OptValue,
    ) -> <Self::EncodedInstType as EncodingValue>::Value {
        T::resolve(optional)
    }
}

/// Trait that provides a conversion of encoding result into a byte slice.
///
/// Implementer must guarantee that the type's memory layout is strictly defined.
/// Usually that means that the implementing struct needs a `#[repr(C)]`
/// and all types of it's fields are also defined in that way.
pub unsafe trait BufferEncoding: Sized {
    /// Convert a structure with known layout into a slice of bytes.
    fn as_bytes(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;

        unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<Self>()) }
    }
}

unsafe impl BufferEncoding for () {}
unsafe impl BufferEncoding for u8 {}
unsafe impl BufferEncoding for u16 {}
unsafe impl BufferEncoding for u32 {}
unsafe impl BufferEncoding for u64 {}
unsafe impl BufferEncoding for i8 {}
unsafe impl BufferEncoding for i16 {}
unsafe impl BufferEncoding for i32 {}
unsafe impl BufferEncoding for i64 {}
unsafe impl BufferEncoding for f32 {}
unsafe impl BufferEncoding for f64 {}
unsafe impl BufferEncoding for bool {}
unsafe impl BufferEncoding for char {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 2] {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 3] {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 4] {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 5] {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 6] {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 7] {}
unsafe impl<T: BufferEncoding> BufferEncoding for [T; 8] {}

/// Represents a single shader uniform or attribute input.
pub trait ShaderInputType {
    /// A type of encoded data
    const TY: ShaderInput;
    /// Binding destination
    /// Type level data representation that's produced in the encoding phase by `InstanceEncoder`.
    /// Note that this type must have a strictly defined layout that matches what GPU will expect.
    type Repr: IterableEncoding;
    // /// Retreive the size of data in binary buffer.
    // fn ubo_size() -> usize;
}

/// Allows visiting the u8 representation of all separate parts of encoded value.
/// The visiting is always performed in the same order as defined in the `Properties` of an encoder.
pub trait IterableEncoding: Sized {
    /// Retreive the size of type in uniform buffer.
    fn ubo_size() -> usize {
        0
    }
    /// Iterate over all encoded buffers in given returned value
    #[inline(always)]
    fn for_each_buffer(&self, f: impl FnMut(usize, &[u8])) {
        self.for_each_buffer_internal(0, f);
    }

    #[doc(hidden)]
    #[inline(always)]
    fn for_each_buffer_internal<F: FnMut(usize, &[u8])>(&self, idx: usize, f: F) -> (usize, F) {
        (idx, f)
    }

    /// Retreive the number of returned descriptors.
    #[inline(always)]
    fn num_descriptors() -> usize {
        0
    }

    /// Iterate over all encoded buffers in given returned value
    #[inline(always)]
    fn for_each_descriptor(self, f: impl FnMut(usize, EncodedDescriptor)) {
        self.for_each_descriptor_internal(0, f);
    }

    #[doc(hidden)]
    #[inline(always)]
    fn for_each_descriptor_internal<F>(self, idx: usize, f: F) -> (usize, F)
    where
        F: FnMut(usize, EncodedDescriptor),
    {
        (idx, f)
    }
}

impl<T: BufferEncoding> IterableEncoding for T {
    #[inline(always)]
    fn ubo_size() -> usize {
        std::mem::size_of::<T>()
    }

    #[inline(always)]
    fn for_each_buffer_internal<F: FnMut(usize, &[u8])>(&self, idx: usize, mut f: F) -> (usize, F) {
        f(idx, self.as_bytes());
        (idx + 1, f)
    }
}

impl IterableEncoding for Handle<Texture> {
    #[inline(always)]
    fn num_descriptors() -> usize {
        1
    }
    #[inline(always)]
    fn for_each_descriptor_internal<F>(self, idx: usize, mut f: F) -> (usize, F)
    where
        F: FnMut(usize, EncodedDescriptor),
    {
        f(idx, EncodedDescriptor::Texture(self));
        (idx + 1, f)
    }
}

macro_rules! define_shader_inputs {
    ($($(#[$meta:meta])* $typename:ident => $repr:ty),*,) => {

        /// An enum of all supported uniform or attribute types
        #[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
        pub enum ShaderInput {
            $(
                $(#[$meta])*
                $typename,
            )*
        }

        impl ShaderInput {
            /// Retreive the size of type in uniform buffer.
            /// Returns 0 for data outside of the binary buffer.
            pub fn ubo_size(&self) -> usize {
                unimplemented!()
                // match self {
                //     $(ShaderInput::$typename => $typename::ubo_size(),)*
                // }
            }
        }

        $(
            $(#[$meta])*
            pub struct $typename;
            impl ShaderInputType for $typename {
                const TY: ShaderInput = ShaderInput::$typename;
                type Repr = $repr;
            }
        )*

    };
}

// TODO: support more types, like boolean vectors or any scalars
define_shader_inputs! {
    /// A vector of 4 single precision floats
    EncVec4 => [f32; 4],
    /// A vector of 2 single precision floats
    EncVec2 => [f32; 2],
    /// A matrix of 4x4 single precision floats
    EncMat4x4 => [[f32; 4]; 4],
    /// A vector of 4 signed integers
    EncVec4i => [i32; 4],
    /// A vector of 2 signed integers
    EncVec2i => [i32; 2],
    /// A vector of 4x4 signed integers
    EncMat4x4i => [[i32; 4]; 4],
    /// A vector of 4 unsigned integers
    EncVec4u => [u32; 4],
    /// A vector of 2 unsigned integers
    EncVec2u => [u32; 2],
    /// A matrix of 4x4 unsigned integers
    EncMat4x4u => [[u32; 4]; 4],
    /// A 2d texture
    EncTexture => Handle<Texture>,
}

/// Combined type that maps a shader attribute layout (a tuple of `ShaderInputType`s)
/// into the corresponding output of an encoder.
pub trait EncodingValue {
    /// A value that is the result of an encoding. Any encoder output must eventually
    /// be resolved to that type at some stage.
    type Value: IterableEncoding;
    /// Optional version of the encoding value. This is what encoders actually pass to the `BufferWriter`.
    type OptValue;
    /// Resolve the optional value into a valid encoding output, using fallback values where needed.
    fn resolve(optional: Self::OptValue, fallback: Self::Value) -> Self::Value;
}

impl EncodingValue for () {
    type Value = ();
    type OptValue = ();
    fn resolve(_: Self::OptValue, _: Self::Value) -> Self::Value {
        ()
    }
}

impl<A> EncodingValue for A
where
    A: ShaderInputType,
{
    type Value = A::Repr;
    type OptValue = Option<A::Repr>;
    fn resolve(optional: Self::OptValue, fallback: Self::Value) -> Self::Value {
        optional.unwrap_or(fallback)
    }
}

/// A compile-time definition of a shader property to encode.
///
/// It is defined by a combination of `ShaderInputType` and a property name.
/// Allows to target properties like `mat4 model` or `vec2 pos`;
pub trait EncProperty {
    /// Name of property used in the shader source code
    const PROPERTY: &'static str;
    /// Type of property used in the shader source code
    type EncodedType: ShaderInputType + EncodingValue;

    /// Get all runtime shader properties for this type of encodable attribute
    fn prop() -> EncodedProp {
        (Self::EncodedType::TY, Self::PROPERTY)
    }

    /// Retreive the size in bytes of underlying property representation.
    fn size() -> usize {
        std::mem::size_of::<<Self::EncodedType as EncodingValue>::Value>()
    }

    /// Retreive fallback value for missing encoded output
    fn fallback() -> <Self::EncodedType as EncodingValue>::Value;
}

/// A runtime representation for unique encodable shader input property
pub type EncodedProp = (ShaderInput, &'static str);

/// A compile-time list of `EncProperty`s.
pub trait EncProperties {
    /// A combined compile-time value representing all encoded properties
    type EncodedType: EncodingValue;
    /// An props iterator type returned from `get_props`
    type PropsIter: Iterator<Item = EncodedProp>;

    /// Retreive a vec of associated (type name, property, byte offset, byte size) tuples at runtime
    fn get_props() -> Self::PropsIter;

    /// Retreive fallback value for missing encoded output
    fn fallback() -> <Self::EncodedType as EncodingValue>::Value;

    /// Resolve encoded optional output, potentially filling a missing value with fallback
    fn resolve(
        optional: <Self::EncodedType as EncodingValue>::OptValue,
    ) -> <Self::EncodedType as EncodingValue>::Value {
        <Self::EncodedType as EncodingValue>::resolve(optional, Self::fallback())
    }
}

impl EncProperties for () {
    type EncodedType = ();
    type PropsIter = Empty<EncodedProp>;
    fn get_props() -> Self::PropsIter {
        empty()
    }
    fn fallback() -> () {
        ()
    }
}

impl<A: EncProperty> EncProperties for A {
    type EncodedType = A::EncodedType;
    type PropsIter = Once<EncodedProp>;

    #[inline(always)]
    fn get_props() -> Self::PropsIter {
        once(A::prop())
    }

    #[inline(always)]
    fn fallback() -> <Self::EncodedType as EncodingValue>::Value {
        A::fallback()
    }
}

macro_rules! recursive_iter {
    (@value $first:expr, $($rest:expr),*) => {
        $first.chain(recursive_iter!(@value $($rest),*))
    };
    (@value $last:expr) => {
        $last
    };
    (@type $first:ty, $($rest:ty),*) => {
        Chain<$first, recursive_iter!(@type $($rest),*)>
    };
    (@type $last:ty) => {
        $last
    };
}

macro_rules! impl_tuple_properties {
    ($($from:ident $idx:tt),*) => {

        impl<$($from,)*> EncProperties for ($($from),*,)
        where $($from: EncProperties),*,
        {
            type EncodedType = ($($from::EncodedType),*,);
            type PropsIter = recursive_iter!(@type $($from::PropsIter),*);

            #[inline(always)]
            fn get_props() -> Self::PropsIter {
                recursive_iter!(@value $($from::get_props()),*)
            }
            #[inline(always)]
            fn fallback() -> <Self::EncodedType as EncodingValue>::Value {
                ($($from::fallback()),*,)
            }
        }

        impl<$($from,)*> IterableEncoding for ($($from),*,)
            where $($from: IterableEncoding),*,
        {
            #[inline(always)]
            fn ubo_size() -> usize {
                let size = 0;
                $(let size = size + $from::ubo_size();)*
                size
            }
            #[inline(always)]
            fn for_each_buffer_internal<FN: FnMut(usize, &[u8])>(&self, idx: usize, f: FN) -> (usize, FN) {
                $(let (idx, f) = self.$idx.for_each_buffer_internal(idx, f);)*
                (idx, f)
            }
            #[inline(always)]
            fn num_descriptors() -> usize {
                let sum = 0;
                $(let sum = sum + $from::num_descriptors();)*
                sum
            }
            #[inline(always)]
            fn for_each_descriptor_internal<FN>(self, idx: usize, f: FN) -> (usize, FN)
            where
                FN: FnMut(usize, EncodedDescriptor),
            {
                $(let (idx, f) = self.$idx.for_each_descriptor_internal(idx, f);)*
                (idx, f)
            }
        }

        impl<$($from,)*> PerInstanceValue for ($($from),*,)
            where $($from: EncodingValue + PerInstanceValue),*,
        {}

        impl<$($from,)*> EncodingValue for ($($from),*,)
            where $($from: EncodingValue),*,
        {
            type Value = ($($from::Value),*,);
            type OptValue = ($($from::OptValue),*,);

            #[allow(non_snake_case)]
            fn resolve(optional: Self::OptValue, fallback: Self::Value) -> Self::Value {
                ($(<$from as EncodingValue>::resolve(optional.$idx, fallback.$idx)),*,)
            }
        }
    }
}

impl_tuple_properties! {A 0}
impl_tuple_properties! {A 0, B 1}
impl_tuple_properties! {A 0, B 1, C 2}
impl_tuple_properties! {A 0, B 1, C 2, D 3}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6 }
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13}
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14 }
impl_tuple_properties! {A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14 , P 15}
