use std::time::Instant;

mod raytracer;

use raytracer::App;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--benchmark" || a == "-b") {
        run_benchmark(&args);
        return;
    }

    let gltf_path = args.get(1).map(|s| s.as_str()).unwrap_or("assets/models/bistro.glb");

    let mut app = App::new(gltf_path).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    });

    while app.window.is_open() {
        let now = Instant::now();
        let delta_time = (now - app.state.last_frame).as_secs_f32();
        app.state.last_frame = now;

        let (input, a_pressed) = app.window.get_camera_input_with_keys();
        let was_moving = app.state.is_moving;
        app.state.is_moving = input.has_movement();

        if a_pressed {
            app.state.adaptive_mode = !app.state.adaptive_mode;
            println!("Adaptive sampling: {}", if app.state.adaptive_mode { "ON" } else { "OFF" });
        }

        if input.scroll_delta != 0.0 {
            let speed_multiplier = 1.05_f32.powf(input.scroll_delta);
            app.config.move_speed = (app.config.move_speed * speed_multiplier).clamp(1.0, 1000.0);
        }

        if app.state.is_moving {
            app.update_camera(&input, delta_time);
        }

        if was_moving && !app.state.is_moving {
            app.state.on_camera_stopped();
        }

        if app.state.is_moving {
            app.render_lowres_frame();
        } else {
            app.render_fullres_frame();
        }
    }

    app.export_results();
}

fn run_benchmark(args: &[String]) {
    use raytracer::camera::Camera;
    use raytracer::framebuffer::Framebuffer;
    use raytracer::loader::gltf::load_gltf;
    use raytracer::material::Material;
    use raytracer::renderloop::{ParallelRenderLoop, RenderLoop};
    use raytracer::renderer::PathTracer;
    use raytracer::scene::Scene;
    use raytracer::sky::TextureSky;
    use raytracer::texture::Texture;
    use glam::Vec3;

    let gltf_path = args.iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .unwrap_or("assets/models/bistro.glb");

    println!("Benchmark: {}", gltf_path);

    println!("Loading model...");
    let load_start = Instant::now();
    let gltf_scene = load_gltf(gltf_path).expect("Failed to load glTF");
    let load_time = load_start.elapsed();
    println!(
        "  Loaded in {:.3}s: {} instances, {} materials, {} textures",
        load_time.as_secs_f32(),
        gltf_scene.instances.len(),
        gltf_scene.materials.len(),
        gltf_scene.textures.len()
    );

    println!("Building acceleration structure...");
    let build_start = Instant::now();

    let width = 640;
    let height = 360;
    let aspect = width as f32 / height as f32;

    let mut camera = gltf_scene.camera.unwrap_or_else(|| {
        Camera::new(Vec3::new(0.0, 1.0, 3.0), Vec3::new(0.0, 0.5, 0.0), 60.0, aspect)
    });
    camera.aspect_ratio = aspect;
    camera.update_matrix();

    let mut materials = gltf_scene.materials;
    if materials.is_empty() {
        materials.push(Material::default());
    }

    let sky_texture = Texture::solid(Vec3::new(0.5, 0.6, 0.8));
    let sky = TextureSky::new(sky_texture);

    let scene = Scene::new(gltf_scene.blases, gltf_scene.instances, materials, gltf_scene.textures, camera, sky, None);
    let build_time = build_start.elapsed();
    println!("  Built in {:.3}s", build_time.as_secs_f32());

    let renderer = PathTracer::<8>::new();
    let render_loop = ParallelRenderLoop;
    let mut framebuffer = Framebuffer::new(width, height);

    let num_samples: usize = 16;
    println!("Rendering {} samples at {}x{}...", num_samples, width, height);

    let render_start = Instant::now();
    for sample in 0..num_samples {
        let pixels = render_loop.render_pass(
            &scene,
            &scene.camera,
            &renderer,
            width,
            height,
            sample as u32,
        );
        framebuffer.accumulate(&pixels);

        if (sample + 1) % 4 == 0 {
            let elapsed = render_start.elapsed().as_secs_f32();
            let samples_per_sec = (sample + 1) as f32 / elapsed;
            println!("  Sample {}/{}: {:.2} spp/sec", sample + 1, num_samples, samples_per_sec);
        }
    }
    let render_time = render_start.elapsed();

    let total_primary_rays = (width * height * num_samples) as f64;
    let mrays_per_sec = total_primary_rays / render_time.as_secs_f64() / 1_000_000.0;

    println!();
    println!("Results:");
    println!("  Build time:   {:.3}s", build_time.as_secs_f32());
    println!("  Render time:  {:.3}s", render_time.as_secs_f32());
    println!("  Primary rays: {:.2}M", total_primary_rays / 1_000_000.0);
    println!("  Performance:  {:.2} Mprimary/s", mrays_per_sec);
    println!("  Samples/sec:  {:.2}", num_samples as f32 / render_time.as_secs_f32());
}
