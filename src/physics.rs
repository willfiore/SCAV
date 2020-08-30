use std::ops::Index;
use nphysics3d::object::DefaultBodyHandle;

pub use nphysics3d::object::{DefaultBodySet, DefaultColliderSet, RigidBodyDesc, RigidBody, BodyStatus, Ground, ColliderDesc, Collider, BodyPartHandle};
pub use nphysics3d::force_generator::DefaultForceGeneratorSet;
pub use nphysics3d::joint::DefaultJointConstraintSet;
pub use nphysics3d::{ncollide3d::shape::{ShapeHandle, Plane, Cuboid}, world::{DefaultMechanicalWorld, DefaultGeometricalWorld}};
pub use na::{Unit, Vector3};

pub struct PhysicsWorld {
    pub mechanical_world: DefaultMechanicalWorld<f32>,
    pub geometrical_world: DefaultGeometricalWorld<f32>,

    pub bodies: DefaultBodySet<f32>,
    pub colliders: DefaultColliderSet<f32>,
    pub joint_constraints: DefaultJointConstraintSet<f32>,
    pub force_generators: DefaultForceGeneratorSet<f32>,

    pub rigid_body_handles: Vec<DefaultBodyHandle>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut physics_world = PhysicsWorld {
            mechanical_world: DefaultMechanicalWorld::new(Vector3::new(0.0, -9.81, 0.0)),
            geometrical_world: DefaultGeometricalWorld::new(),
            bodies: DefaultBodySet::new(),
            colliders: DefaultColliderSet::new(),
            joint_constraints: DefaultJointConstraintSet::new(),
            force_generators: DefaultForceGeneratorSet::new(),
            rigid_body_handles: Vec::new(),
        };

        // Ground
        {
            let body = physics_world.bodies.insert(Ground::new());
            let shape = ShapeHandle::new(Plane::new(Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0))));

            physics_world.rigid_body_handles.push(body);

            let collider = ColliderDesc::new(shape)
                .build(BodyPartHandle(body, 0));

            physics_world.colliders.insert(collider);
        }

        // Cubes
        {
            let num = 6;
            let rad = 0.5 - 0.005;

            let cuboid = ShapeHandle::new(Cuboid::new(Vector3::repeat(rad)));

            let shift = rad * 2.5;
            let centerx = shift * (num as f32) / 2.0;
            let centery = shift / 2.0;
            let centerz = shift * (num as f32) / 2.0;
            let height = 100.0;

            for i in 0usize..num {
                for j in 0usize..num {
                    for k in 0usize..num {
                        let x = i as f32 * shift - centerx;
                        let y = j as f32 * shift + centery + height;
                        let z = k as f32 * shift - centerz;

                        // Build the rigid body.
                        let rb = RigidBodyDesc::new()
                            .translation(Vector3::new(x, y, z))
                            .build();

                        // Insert the rigid body to the body set.
                        let rb_handle = physics_world.bodies.insert(rb);

                        physics_world.rigid_body_handles.push(rb_handle);

                        // Build the collider.
                        let co = ColliderDesc::new(cuboid.clone())
                            .density(1.0)
                            .build(BodyPartHandle(rb_handle, 0));

                        // Insert the collider to the body set.
                        physics_world.colliders.insert(co);
                    }
                }
            }
        }

        physics_world
    }

    pub fn set_timestep(&mut self, dt: f32) {
        self.mechanical_world.set_timestep(dt);
    }
    
    pub fn step(&mut self) {
        self.mechanical_world.step(
            &mut self.geometrical_world,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joint_constraints,
            &mut self.force_generators
        );
    }
}
