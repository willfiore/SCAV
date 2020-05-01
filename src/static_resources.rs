use crate::renderer::{Model, Vertex};

pub fn model_cube() -> Model {

    let vertices = vec![
        // Back face
        Vertex { position: na::Vector3::new(-0.5,  0.5,  0.5), color: na::Vector3::new(1.0, 0.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5,  0.5,  0.5), color: na::Vector3::new(1.0, 0.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5, -0.5,  0.5), color: na::Vector3::new(1.0, 0.0, 0.0) },
        Vertex { position: na::Vector3::new(-0.5, -0.5,  0.5), color: na::Vector3::new(1.0, 0.0, 0.0) },

        // Right face
        Vertex { position: na::Vector3::new( 0.5,  0.5,  0.5), color: na::Vector3::new(0.0, 1.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5,  0.5, -0.5), color: na::Vector3::new(0.0, 1.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5, -0.5, -0.5), color: na::Vector3::new(0.0, 1.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5, -0.5,  0.5), color: na::Vector3::new(0.0, 1.0, 0.0) },

        // Left face
        Vertex { position: na::Vector3::new(-0.5, -0.5,  0.5), color: na::Vector3::new(0.0, 0.0, 1.0) },
        Vertex { position: na::Vector3::new(-0.5, -0.5, -0.5), color: na::Vector3::new(0.0, 0.0, 1.0) },
        Vertex { position: na::Vector3::new(-0.5,  0.5, -0.5), color: na::Vector3::new(0.0, 0.0, 1.0) },
        Vertex { position: na::Vector3::new(-0.5,  0.5,  0.5), color: na::Vector3::new(0.0, 0.0, 1.0) },

        // Front face
        Vertex { position: na::Vector3::new(-0.5, -0.5,  -0.5), color: na::Vector3::new(1.0, 1.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5, -0.5,  -0.5), color: na::Vector3::new(1.0, 1.0, 0.0) },
        Vertex { position: na::Vector3::new( 0.5,  0.5,  -0.5), color: na::Vector3::new(1.0, 1.0, 0.0) },
        Vertex { position: na::Vector3::new(-0.5,  0.5,  -0.5), color: na::Vector3::new(1.0, 1.0, 0.0) },

        // Bottom face
        Vertex { position: na::Vector3::new(-0.5,  0.5, -0.5), color: na::Vector3::new(1.0, 0.0, 1.0) },
        Vertex { position: na::Vector3::new( 0.5,  0.5, -0.5), color: na::Vector3::new(1.0, 0.0, 1.0) },
        Vertex { position: na::Vector3::new( 0.5,  0.5,  0.5), color: na::Vector3::new(1.0, 0.0, 1.0) },
        Vertex { position: na::Vector3::new(-0.5,  0.5,  0.5), color: na::Vector3::new(1.0, 0.0, 1.0) },

        // Top face
        Vertex { position: na::Vector3::new(-0.5, -0.5,  0.5), color: na::Vector3::new(0.0, 1.0, 1.0) },
        Vertex { position: na::Vector3::new( 0.5, -0.5,  0.5), color: na::Vector3::new(0.0, 1.0, 1.0) },
        Vertex { position: na::Vector3::new( 0.5, -0.5, -0.5), color: na::Vector3::new(0.0, 1.0, 1.0) },
        Vertex { position: na::Vector3::new(-0.5, -0.5, -0.5), color: na::Vector3::new(0.0, 1.0, 1.0) },
    ];

    let indices = vec![
        0u32, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20,
    ];

    Model { vertices, indices }
}