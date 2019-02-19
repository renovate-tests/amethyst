use super::{EncProperty, EncodingLayout, EncodingQuery, FnPipelineResolver, Shader};
use crate::{Sprite, SpriteRender, SpriteSheet};
use amethyst_assets::{AssetStorage, Handle, Loader, Processor};
use amethyst_core::{
    nalgebra::Matrix4,
    specs::{Component, RunNow, VecStorage, World},
    GlobalTransform, Time,
};
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

// floats with useful binary representation
const AAAAAAAA: f32 = -3.0316488252093987e-13;
const BBBBBBBB: f32 = -0.005729166325181723;
const CCCCCCCC: f32 = -107374176.0;
const DDDDDDDD: f32 = -1998397155538108400.0;

struct TestCentralComponent(Handle<Shader>);
impl Component for TestCentralComponent {
    type Storage = VecStorage<Self>;
}

pub struct HandleFake {
    _id: std::sync::Arc<u32>,
    marker: std::marker::PhantomData<()>,
}
impl HandleFake {
    /// Create fake handle for test mocking purposes
    fn new<H>(fake_id: u32) -> amethyst_assets::Handle<H> {
        let fake = Self {
            _id: std::sync::Arc::new(fake_id),
            marker: std::marker::PhantomData,
        };
        unsafe { std::mem::transmute(fake) }
    }
}

fn mock_world() -> World {
    use super::{
        pipeline::{EncodingLayout, LayoutProp},
        properties_impl::*,
    };
    use crate::Rgba;
    use amethyst_core::specs::world::Builder;

    let mut world = World::new();
    let pool = Arc::new(ThreadPoolBuilder::default().build().unwrap());
    world.add_resource(pool.clone());
    world.add_resource(Loader::new(".", pool));
    world.add_resource(Time::default());
    world.add_resource(AssetStorage::<SpriteSheet>::default());
    world.add_resource(AssetStorage::<Shader>::default());
    // world.add_resource(AssetStorage::<EncodingLayout>::default());
    // world.add_resource(LayoutResolveCache::default());
    world.register::<GlobalTransform>();
    world.register::<SpriteRender>();
    world.register::<Rgba>();
    world.register::<TestCentralComponent>();

    let (sprite_sheet, shader_xy, shader_tint, shader_xy_tint, shader_xy_tint_reorder) = {
        let loader = world.read_resource::<Loader>();
        let sprite_sheet = loader.load_from_data(
            SpriteSheet {
                texture: HandleFake::new(0),
                sprites: vec![
                    Sprite::from_pixel_values(128, 128, 64, 64, 0, 0, [0.0, 0.0]),
                    Sprite::from_pixel_values(128, 128, 64, 64, 64, 0, [0.0, 0.0]),
                    Sprite::from_pixel_values(128, 128, 64, 64, 0, 64, [0.0, 0.0]),
                    Sprite::from_pixel_values(128, 128, 64, 64, 64, 64, [0.0, 0.0]),
                ],
            },
            (),
            &world.res.fetch::<AssetStorage<SpriteSheet>>(),
        );

        let shader_xy = loader.load_from_data(
            Shader {
                mock_layout: EncodingLayout {
                    padded_size: (Pos2DProperty::size()
                        + DirXProperty::size()
                        + DirYProperty::size()) as _,
                    props: vec![
                        LayoutProp {
                            prop: Pos2DProperty::prop(),
                            absolute_offset: 0,
                        },
                        LayoutProp {
                            prop: DirXProperty::prop(),
                            absolute_offset: Pos2DProperty::size() as _,
                        },
                        LayoutProp {
                            prop: DirYProperty::prop(),
                            absolute_offset: (Pos2DProperty::size() + DirXProperty::size()) as _,
                        },
                    ],
                },
            },
            (),
            &world.res.fetch::<AssetStorage<Shader>>(),
        );

        let shader_tint = loader.load_from_data(
            Shader {
                mock_layout: EncodingLayout {
                    padded_size: TintProperty::size() as _,
                    props: vec![LayoutProp {
                        prop: TintProperty::prop(),
                        absolute_offset: 0,
                    }],
                },
            },
            (),
            &world.res.fetch::<AssetStorage<Shader>>(),
        );

        let shader_xy_tint = loader.load_from_data(
            Shader {
                mock_layout: EncodingLayout {
                    padded_size: (Pos2DProperty::size()
                        + DirXProperty::size()
                        + DirYProperty::size()
                        + TintProperty::size()) as _,
                    props: vec![
                        LayoutProp {
                            prop: Pos2DProperty::prop(),
                            absolute_offset: 0,
                        },
                        LayoutProp {
                            prop: DirXProperty::prop(),
                            absolute_offset: Pos2DProperty::size() as _,
                        },
                        LayoutProp {
                            prop: DirYProperty::prop(),
                            absolute_offset: (Pos2DProperty::size() + DirXProperty::size()) as _,
                        },
                        LayoutProp {
                            prop: TintProperty::prop(),
                            absolute_offset: (Pos2DProperty::size()
                                + DirXProperty::size()
                                + DirYProperty::size())
                                as _,
                        },
                    ],
                },
            },
            (),
            &world.res.fetch::<AssetStorage<Shader>>(),
        );

        let shader_xy_tint_reorder = loader.load_from_data(
            Shader {
                mock_layout: EncodingLayout {
                    padded_size: (Pos2DProperty::size()
                        + DirXProperty::size()
                        + DirYProperty::size()
                        + TintProperty::size()) as _,
                    props: vec![
                        LayoutProp {
                            prop: TintProperty::prop(),
                            absolute_offset: 0,
                        },
                        LayoutProp {
                            prop: DirYProperty::prop(),
                            absolute_offset: TintProperty::size() as _,
                        },
                        LayoutProp {
                            prop: DirXProperty::prop(),
                            absolute_offset: (TintProperty::size() + DirYProperty::size()) as _,
                        },
                        LayoutProp {
                            prop: Pos2DProperty::prop(),
                            absolute_offset: (TintProperty::size()
                                + DirYProperty::size()
                                + DirXProperty::size())
                                as _,
                        },
                    ],
                },
            },
            (),
            &world.res.fetch::<AssetStorage<Shader>>(),
        );

        (
            sprite_sheet,
            shader_xy,
            shader_tint,
            shader_xy_tint,
            shader_xy_tint_reorder,
        )
    };

    // a few examples for testing:
    // one empty entity
    // one that has all valid components
    // one missing GlobalTransform
    // one missing Rgba - optionals feature

    // world.create_entity().build();
    // world
    //     .create_entity()
    //     .with(TestCentralComponent(shader_xy_tint.clone()))
    //     .with(GlobalTransform::new())
    //     .with(SpriteRender {
    //         sprite_sheet: sprite_sheet.clone(),
    //         sprite_number: 0,
    //     })
    //     .with(Rgba::BLUE)
    //     .build();

    // world
    //     .create_entity()
    //     .with(TestCentralComponent(shader_tint.clone()))
    //     .with(SpriteRender {
    //         sprite_sheet: sprite_sheet.clone(),
    //         sprite_number: 1,
    //     })
    //     .with(Rgba::RED)
    //     .build();
    // world
    //     .create_entity()
    //     .with(TestCentralComponent(shader_xy.clone()))
    //     .with(GlobalTransform::new())
    //     .with(SpriteRender {
    //         sprite_sheet: sprite_sheet.clone(),
    //         sprite_number: 2,
    //     })
    //     .build();

    world
        .create_entity()
        .with(TestCentralComponent(shader_xy_tint.clone()))
        .with(GlobalTransform(Matrix4::new(
            1.0, 0.0, 0.0, AAAAAAAA, //
            0.0, 1.0, 0.0, BBBBBBBB, //
            0.0, 0.0, 1.0, CCCCCCCC, //
            0.0, 0.0, 0.0, 1.0, //
        )))
        .with(Rgba(DDDDDDDD, DDDDDDDD, DDDDDDDD, DDDDDDDD))
        .with(SpriteRender {
            sprite_sheet: sprite_sheet.clone(),
            sprite_number: 2,
        })
        .build();

    // world
    //     .create_entity()
    //     .with(TestCentralComponent(shader_xy_tint_reorder.clone()))
    //     .with(GlobalTransform::new())
    //     .with(Rgba::BLUE)
    //     .with(SpriteRender {
    //         sprite_sheet: sprite_sheet.clone(),
    //         sprite_number: 3,
    //     })
    //     .build();

    <Processor<SpriteSheet>>::new().run_now(&world.res);
    <Processor<Shader>>::new().run_now(&world.res);
    world
}

#[test]
pub fn test_querying() {
    let ref mut res = mock_world().res;

    use super::{
        encoders_impl::{RgbaTintEncoder, SpriteTransformEncoder},
        EncoderStorage,
    };

    res.insert(
        EncoderStorage::build()
            .with_encoder::<RgbaTintEncoder>()
            .with_encoder::<SpriteTransformEncoder>()
            .build(),
    );

    let mut query = EncodingQuery::new(FnPipelineResolver::new(
        |c: &TestCentralComponent, _, res: &shred::Resources| {
            let storage = res.fetch::<AssetStorage<_>>();
            storage
                .get(&c.0)
                .map(|shader| EncodingLayout::from_shader(shader))
        },
        |_, _, _| 0,
        |c: &TestCentralComponent, _, _| c.0.id(),
    ));

    let evaluated = query.evaluate(res);
    println!("evaluated: {:?}", evaluated);

    let size = evaluated.ubo_size();
    let mut buffer = vec![0u8; size];
    let result = evaluated.encode(&res, &mut buffer);

    println!("result: {:?}", result);
    println!("buffer: {:x?}", buffer);
}
