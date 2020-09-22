use rapier3d::pipeline::PhysicsPipeline;
use rapier3d::dynamics::{JointSet, RigidBodySet, IntegrationParameters, RigidBodyBuilder};
use rapier3d::geometry::{BroadPhase, NarrowPhase, ColliderSet, ColliderBuilder};
use rapier3d::na::Vector3;

pub struct PhysicsWorld {
    pub pipeline: PhysicsPipeline,
    pub gravity: Vector3<f32>,
    pub integration_parameters: IntegrationParameters,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub joints: JointSet,

    pub event_handler: ()
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut w = PhysicsWorld {
            pipeline:     PhysicsPipeline::new(),
            gravity: Vector3::new(0.0, -9.81, 0.0),
            integration_parameters: IntegrationParameters::default(),
            broad_phase:  BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies:       RigidBodySet::new(),
            colliders:    ColliderSet::new(),
            joints:       JointSet::new(),
            event_handler: ()
        };

        w.integration_parameters.ccd_on_penetration_enabled = true;

        // Ground
        {
            let ground_size = 100.0;
            let ground_height = 0.1;

            let rb = RigidBodyBuilder::new_static()
                .translation(0.0, 0.0, 0.0)
                .build();

            let ground_handle = w.bodies.insert(rb);

            let collider = ColliderBuilder::cuboid(
                ground_size, ground_height, ground_size
            ).build();

            w.colliders.insert(collider, ground_handle, &mut w.bodies);
        }

        // Cubes
        {
            let num = 6;
            let rad = 0.5;

            let shift = rad * 2.5;
            let centerx = shift * (num as f32) / 2.0;
            let centery = shift / 2.0;
            let centerz = shift * (num as f32) / 2.0;
            let height = 10.0;

            for i in 0usize..num {
                for j in 0usize..num {
                    for k in 0usize..num {
                        let x = i as f32 * shift - centerx;
                        let y = j as f32 * shift + centery + height;
                        let z = k as f32 * shift - centerz;

                        // Build the rigid body.
                        let rb = RigidBodyBuilder::new_dynamic()
                            .translation(x, y, z)
                            .build();

                        let rb_handle = w.bodies.insert(rb);

                        // Build the collider.
                        let co = ColliderBuilder::cuboid(rad, rad, rad)
                            .density(1.0)
                            .build();

                        // Insert the collider to the body set.
                        w.colliders.insert(co, rb_handle, &mut w.bodies);
                    }
                }
            }
        }

        w
    }

    pub fn set_timestep(&mut self, dt: f32) {
        self.integration_parameters.set_dt(dt);
    }
    
    pub fn step(&mut self) {
        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joints,
            &self.event_handler
        )
    }
}
