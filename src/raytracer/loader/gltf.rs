use crate::raytracer::accel::Blas;
use crate::raytracer::camera::Camera;
use crate::raytracer::material::{AlphaMode, Material};
use crate::raytracer::shape::TriangleMesh;
use crate::raytracer::texture::Texture;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::path::Path;

pub struct GltfScene {
    pub blases: Vec<Blas<TriangleMesh>>,
    pub instances: Vec<(u32, Mat4)>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub camera: Option<Camera>,
}

pub fn load_gltf<P: AsRef<Path>>(path: P) -> Result<GltfScene, GltfError> {
    let path = path.as_ref();
    let (document, buffers, images) = gltf::import(path)?;

    let textures = load_textures(&document, &images);
    let materials = load_materials(&document, &textures);

    let axis_correction = detect_axis_correction(&document);

    let (blases, instances) = load_meshes(&document, &buffers, &materials, axis_correction);
    let camera = load_camera(&document, axis_correction);

    Ok(GltfScene {
        blases,
        instances,
        materials,
        textures,
        camera,
    })
}

fn detect_axis_correction(_document: &gltf::Document) -> Option<Mat4> {
    None
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn load_texture_data(img: &gltf::image::Data, convert_srgb: bool) -> Texture {
    let width = img.width;
    let height = img.height;

    let data: Vec<Vec3> = match img.format {
        gltf::image::Format::R8G8B8 => img
            .pixels
            .chunks(3)
            .map(|p| {
                let r = p[0] as f32 / 255.0;
                let g = p[1] as f32 / 255.0;
                let b = p[2] as f32 / 255.0;
                if convert_srgb {
                    Vec3::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
                } else {
                    Vec3::new(r, g, b)
                }
            })
            .collect(),
        gltf::image::Format::R8G8B8A8 => img
            .pixels
            .chunks(4)
            .map(|p| {
                let r = p[0] as f32 / 255.0;
                let g = p[1] as f32 / 255.0;
                let b = p[2] as f32 / 255.0;
                if convert_srgb {
                    Vec3::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
                } else {
                    Vec3::new(r, g, b)
                }
            })
            .collect(),
        gltf::image::Format::R16G16B16 => img
            .pixels
            .chunks(6)
            .map(|p| {
                let r = u16::from_le_bytes([p[0], p[1]]) as f32 / 65535.0;
                let g = u16::from_le_bytes([p[2], p[3]]) as f32 / 65535.0;
                let b = u16::from_le_bytes([p[4], p[5]]) as f32 / 65535.0;
                Vec3::new(r, g, b)
            })
            .collect(),
        gltf::image::Format::R16G16B16A16 => img
            .pixels
            .chunks(8)
            .map(|p| {
                let r = u16::from_le_bytes([p[0], p[1]]) as f32 / 65535.0;
                let g = u16::from_le_bytes([p[2], p[3]]) as f32 / 65535.0;
                let b = u16::from_le_bytes([p[4], p[5]]) as f32 / 65535.0;
                Vec3::new(r, g, b)
            })
            .collect(),
        _ => vec![Vec3::splat(0.5); (width * height) as usize],
    };
    Texture::Image { data, width, height }
}

fn build_srgb_texture_set(document: &gltf::Document) -> std::collections::HashSet<usize> {
    let mut srgb_images = std::collections::HashSet::new();

    let tex_to_img: Vec<usize> = document
        .textures()
        .map(|t| t.source().index())
        .collect();

    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();

        if let Some(info) = pbr.base_color_texture() {
            if let Some(&img_idx) = tex_to_img.get(info.texture().index()) {
                srgb_images.insert(img_idx);
            }
        }

        if let Some(info) = mat.emissive_texture() {
            if let Some(&img_idx) = tex_to_img.get(info.texture().index()) {
                srgb_images.insert(img_idx);
            }
        }
    }

    srgb_images
}

fn load_textures(document: &gltf::Document, images: &[gltf::image::Data]) -> Vec<Texture> {
    let srgb_set = build_srgb_texture_set(document);

    images
        .iter()
        .enumerate()
        .map(|(idx, img)| {
            let convert_srgb = srgb_set.contains(&idx);
            load_texture_data(img, convert_srgb)
        })
        .collect()
}

fn load_materials(document: &gltf::Document, _textures: &[Texture]) -> Vec<Material> {
    let tex_to_img: Vec<u32> = document
        .textures()
        .map(|t| t.source().index() as u32)
        .collect();

    let get_image_index = |tex_index: usize| -> Option<u32> {
        tex_to_img.get(tex_index).copied()
    };

    document
        .materials()
        .map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let base_color_texture = pbr
                .base_color_texture()
                .and_then(|t| get_image_index(t.texture().index()));
            let metallic_roughness_texture = pbr
                .metallic_roughness_texture()
                .and_then(|t| get_image_index(t.texture().index()));
            let normal_texture = mat.normal_texture().and_then(|t| get_image_index(t.texture().index()));
            let normal_scale = mat.normal_texture().map_or(1.0, |t| t.scale());
            let occlusion_texture = mat.occlusion_texture().and_then(|t| get_image_index(t.texture().index()));
            let occlusion_strength = mat.occlusion_texture().map_or(1.0, |t| t.strength());
            let emissive_texture = mat.emissive_texture().and_then(|t| get_image_index(t.texture().index()));
            let emissive = mat.emissive_factor();

            let alpha_mode = match mat.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };

            let transmission = mat
                .transmission()
                .map_or(0.0, |t| t.transmission_factor());
            let ior = mat.ior().unwrap_or(1.5);

            Material {
                base_color_factor: Vec4::from_array(base_color),
                base_color_texture,
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                metallic_roughness_texture,
                normal_texture,
                normal_scale,
                occlusion_texture,
                occlusion_strength,
                emissive_texture,
                emissive_factor: Vec3::from_array(emissive),
                alpha_mode,
                alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
                double_sided: mat.double_sided(),
                ior,
                transmission,
                ..Default::default()
            }
        })
        .collect()
}

fn compute_global_transforms(document: &gltf::Document) -> Vec<Mat4> {
    let node_count = document.nodes().count();
    let mut global_transforms = vec![Mat4::IDENTITY; node_count];

    fn traverse(
        node: gltf::Node,
        parent_transform: Mat4,
        global_transforms: &mut [Mat4],
    ) {
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        let global = parent_transform * local;
        global_transforms[node.index()] = global;

        for child in node.children() {
            traverse(child, global, global_transforms);
        }
    }

    for scene in document.scenes() {
        for node in scene.nodes() {
            traverse(node, Mat4::IDENTITY, &mut global_transforms);
        }
    }

    global_transforms
}

fn load_meshes(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    _materials: &[Material],
    axis_correction: Option<Mat4>,
) -> (Vec<Blas<TriangleMesh>>, Vec<(u32, Mat4)>) {
    let mut blases = Vec::new();
    let mut instances = Vec::new();

    let global_transforms = compute_global_transforms(document);

    for node in document.nodes() {
        if let Some(mesh) = node.mesh() {
            let mut transform = global_transforms[node.index()];

            if let Some(correction) = axis_correction {
                transform = correction * transform;
            }

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions: Vec<Vec3> = reader
                    .read_positions()
                    .map(|iter| iter.map(Vec3::from_array).collect())
                    .unwrap_or_default();

                if positions.is_empty() {
                    continue;
                }

                let normals: Vec<Vec3> = reader
                    .read_normals()
                    .map(|iter| iter.map(Vec3::from_array).collect())
                    .unwrap_or_default();

                let uvs: Vec<Vec2> = reader
                    .read_tex_coords(0)
                    .map(|iter| iter.into_f32().map(Vec2::from_array).collect())
                    .unwrap_or_default();

                let indices: Vec<[u32; 3]> = reader
                    .read_indices()
                    .map(|iter| {
                        let flat: Vec<u32> = iter.into_u32().collect();
                        flat.chunks(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        (0..positions.len() as u32)
                            .collect::<Vec<_>>()
                            .chunks(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect()
                    });

                let material_id = primitive.material().index().unwrap_or(0) as u32;

                let mesh = TriangleMesh::new(positions, normals, uvs, indices, material_id);
                let blas_index = blases.len() as u32;
                blases.push(Blas::build(vec![mesh]));
                instances.push((blas_index, transform));
            }
        }
    }

    (blases, instances)
}

fn load_camera(document: &gltf::Document, axis_correction: Option<Mat4>) -> Option<Camera> {
    let global_transforms = compute_global_transforms(document);

    for node in document.nodes() {
        if let Some(cam) = node.camera() {
            let mut transform = global_transforms[node.index()];

            if let Some(correction) = axis_correction {
                transform = correction * transform;
            }

            let position = transform.col(3).truncate();

            let forward = -transform.col(2).truncate().normalize();
            let look_at = position + forward;

            match cam.projection() {
                gltf::camera::Projection::Perspective(persp) => {
                    let fov = persp.yfov().to_degrees();
                    let camera = Camera::new(position, look_at, fov, 1.0);
                    return Some(camera);
                }
                _ => continue,
            }
        }
    }
    None
}

#[derive(Debug)]
pub enum GltfError {
    Gltf(gltf::Error),
    Io(std::io::Error),
}

impl From<gltf::Error> for GltfError {
    fn from(e: gltf::Error) -> Self {
        GltfError::Gltf(e)
    }
}

impl From<std::io::Error> for GltfError {
    fn from(e: std::io::Error) -> Self {
        GltfError::Io(e)
    }
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfError::Gltf(e) => write!(f, "glTF error: {}", e),
            GltfError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for GltfError {}
