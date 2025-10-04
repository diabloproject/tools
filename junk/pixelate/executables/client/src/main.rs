use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::input::mouse::MouseWheel;
use engine_proto::engine::data_service_client::DataServiceClient;
use engine_proto::engine::*;
use tonic::transport::Channel;
use std::sync::Arc;
use tokio::sync::Mutex;
use crossbeam_channel::{unbounded, Receiver, TryRecvError};
use tokio::runtime::Runtime;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Pixelate Client".to_string(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(TokioRuntime(Arc::new(
            tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
        )))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            check_connection, 
            check_fetch, 
            check_push, 
            update_canvas, 
            handle_input, 
            camera_controls,
            periodic_update
        ))
        .run();
}

#[derive(Resource, Clone)]
struct TokioRuntime(Arc<Runtime>);

#[derive(Resource)]
struct GrpcClient(Arc<Mutex<DataServiceClient<Channel>>>);

#[derive(Resource)]
struct CanvasData {
    width: u64,
    height: u64,
    center_x: i64,
    center_y: i64,
    pixels: Vec<u32>,
}

#[derive(Resource)]
struct UpdateCanvasFlag(bool);

#[derive(Resource)]
struct CanvasImageHandle(Handle<Image>);

#[derive(Resource)]
struct LastUpdateTimer(Timer);

async fn connect_client() -> Result<DataServiceClient<Channel>, String> {
    DataServiceClient::connect("http://127.0.1:50051")
        .await
        .map_err(|e| e.to_string())
}

// Convert color ID (0-255) to RGB
fn color_id_to_rgb(id: u32) -> [u8; 3] {
    match id {
        0 => [0, 0, 0],           // Black
        1 => [255, 255, 255],     // White
        2 => [255, 0, 0],         // Red
        3 => [0, 255, 0],         // Green
        4 => [0, 0, 255],         // Blue
        5 => [255, 255, 0],       // Yellow
        6 => [255, 0, 255],       // Magenta
        7 => [0, 255, 255],       // Cyan
        8 => [128, 128, 128],     // Gray
        255 => [240, 240, 240],   // Light gray (default/empty)
        _ => {
            // Generate a color based on the ID for any other value
            let r = ((id * 137) % 256) as u8;
            let g = ((id * 211) % 256) as u8;
            let b = ((id * 97) % 256) as u8;
            [r, g, b]
        }
    }
}

// Convert column-major pixel data (server format) to row-major (image format)
// Server sends: for x in 0..width { for y in 0..height { pixels[x*height + y] } }
// Image needs: for y in 0..height { for x in 0..width { pixels[y*width + x] } }
fn column_major_to_row_major(pixels: &[u32], width: u64, height: u64) -> Vec<u32> {
    let mut result = vec![0u32; (width * height) as usize];
    for x in 0..width {
        for y in 0..height {
            let col_major_index = (x * height + y) as usize;
            let row_major_index = (y * width + x) as usize;
            result[row_major_index] = pixels[col_major_index];
        }
    }
    result
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, runtime: Res<TokioRuntime>) {
    commands.spawn(Camera2d);

    // Connect to server asynchronously
    let (tx, rx) = unbounded();
    let rt = runtime.0.clone();
    std::thread::spawn(move || {
        let result = rt.block_on(connect_client());
        let _ = tx.send(result);
    });

    commands.insert_resource(ConnectTask(rx));

    // Placeholder canvas data
    let canvas_data = CanvasData {
        width: 100,
        height: 100,
        center_x: 0,
        center_y: 0,
        pixels: vec![0; 10000],
    };

    // Create initial image
    // Convert from column-major (server) to row-major (image)
    let row_major_pixels = column_major_to_row_major(&canvas_data.pixels, canvas_data.width, canvas_data.height);
    let mut data = Vec::with_capacity((canvas_data.width * canvas_data.height * 4) as usize);
    for &color_id in &row_major_pixels {
        let rgb = color_id_to_rgb(color_id);
        data.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    let mut image = Image::new(
        bevy::render::render_resource::Extent3d {
            width: canvas_data.width as u32,
            height: canvas_data.height as u32,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        data,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    // Set to nearest neighbor filtering for crisp pixels
    image.sampler = ImageSampler::nearest();
    let image_handle = images.add(image);

    // Spawn sprite with scaling
    commands.spawn((
        Sprite::from_image(image_handle.clone()),
        Transform::from_translation(Vec3::new(
            canvas_data.center_x as f32,
            canvas_data.center_y as f32,
            0.0,
        ))
        .with_scale(Vec3::splat(4.0)), // Scale up the sprite
        CanvasSprite,
    ));

    commands.insert_resource(canvas_data);
    commands.insert_resource(UpdateCanvasFlag(true));
    commands.insert_resource(CanvasImageHandle(image_handle));
    commands.insert_resource(LastUpdateTimer(Timer::from_seconds(1.0, TimerMode::Repeating)));
}

#[derive(Resource)]
struct ConnectTask(Receiver<Result<DataServiceClient<Channel>, String>>);

fn check_connection(
    mut commands: Commands,
    connect_task: ResMut<ConnectTask>,
    client: Option<Res<GrpcClient>>,
    runtime: Res<TokioRuntime>,
) {
    if client.is_some() {
        return;
    }
    match connect_task.0.try_recv() {
        Ok(result) => {
            match result {
                Ok(client) => {
                    println!("✓ Connected to gRPC server!");
                    let grpc_client = Arc::new(Mutex::new(client));
                    commands.insert_resource(GrpcClient(grpc_client.clone()));
                    // Fetch initial snapshot
                    let (tx, rx) = unbounded();
                    let rt = runtime.0.clone();
                    std::thread::spawn(move || {
                        println!("Fetching initial snapshot...");
                        let result = rt.block_on(fetch_snapshot(grpc_client));
                        println!("Initial snapshot received: {}x{}, {} pixels", 
                                 result.width, result.height, result.pixels.len());
                        let _ = tx.send(result);
                    });
                    commands.insert_resource(FetchTask(rx));
                }
                Err(e) => {
                    eprintln!("✗ Failed to connect: {:?}", e);
                }
            }
        }
        Err(TryRecvError::Empty) => {
            // Task still running
        }
        Err(TryRecvError::Disconnected) => {
            eprintln!("✗ Connection task disconnected unexpectedly");
        }
    }
}

async fn fetch_snapshot(client: Arc<Mutex<DataServiceClient<Channel>>>) -> CanvasData {
    let request = tonic::Request::new(SnapshotRequest { timestamp: 0 });
    let mut client = client.lock().await;
    if let Ok(response) = client.snapshot(request).await {
        let resp = response.into_inner();
        CanvasData {
            width: resp.width,
            height: resp.height,
            center_x: resp.center_x,
            center_y: resp.center_y,
            pixels: resp.pixel_values,
        }
    } else {
        CanvasData {
            width: 100,
            height: 100,
            center_x: 0,
            center_y: 0,
            pixels: vec![0; 10000],
        }
    }
}

#[derive(Resource)]
struct FetchTask(Receiver<CanvasData>);

fn check_fetch(
    mut commands: Commands,
    mut fetch_task: Option<ResMut<FetchTask>>,
    mut canvas: ResMut<CanvasData>,
    mut update_flag: ResMut<UpdateCanvasFlag>,
) {
    if let Some(task) = fetch_task.as_mut() {
        match task.0.try_recv() {
            Ok(new_canvas) => {
                println!("✓ Snapshot received in render thread: {}x{}, {} pixels", 
                         new_canvas.width, new_canvas.height, new_canvas.pixels.len());
                *canvas = new_canvas;
                update_flag.0 = true;
                commands.remove_resource::<FetchTask>();
            }
            Err(TryRecvError::Empty) => {
                // Task still running
            }
            Err(TryRecvError::Disconnected) => {
                // Task completed or failed
                println!("✗ Fetch task disconnected");
                commands.remove_resource::<FetchTask>();
            }
        }
    }
}

fn update_canvas(
    canvas: Res<CanvasData>,
    mut images: ResMut<Assets<Image>>,
    image_handle: Res<CanvasImageHandle>,
    mut update_flag: ResMut<UpdateCanvasFlag>,
) {
    if !update_flag.0 {
        return;
    }
    update_flag.0 = false;

    println!("Updating canvas texture with new data...");
    
    // Update the texture with new pixels using the stored handle
    if let Some(image) = images.get_mut(&image_handle.0) {
        // Convert from column-major (server) to row-major (image)
        let row_major_pixels = column_major_to_row_major(&canvas.pixels, canvas.width, canvas.height);
        let mut data = Vec::with_capacity((canvas.width * canvas.height * 4) as usize);
        for &color_id in &row_major_pixels {
            let rgb = color_id_to_rgb(color_id);
            data.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        image.data = Some(data);
        image.resize(bevy::render::render_resource::Extent3d {
            width: canvas.width as u32,
            height: canvas.height as u32,
            depth_or_array_layers: 1,
        });
        println!("✓ Canvas texture updated!");
    } else {
        println!("✗ Could not get mutable image from handle");
    }
}

#[derive(Component)]
struct CanvasSprite;

#[derive(Resource, Default)]
struct DragState {
    dragging: bool,
    last_pos: Option<Vec2>,
}

fn camera_controls(
    mut camera_query: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    mut scroll_events: EventReader<MouseWheel>,
    mut drag_state: Local<Option<DragState>>,
) {
    if drag_state.is_none() {
        *drag_state = Some(DragState::default());
    }
    let drag_state = drag_state.as_mut().unwrap();
    
    let Ok((mut camera_transform, mut projection)) = camera_query.single_mut() else {
        return;
    };
    
    // Get the orthographic projection scale
    let scale = if let Projection::Orthographic(ref mut ortho) = projection.as_mut() {
        // Zoom with mouse wheel
        for event in scroll_events.read() {
            let zoom_delta = -event.y * 0.03;
            ortho.scale = (ortho.scale + zoom_delta).max(0.1).min(10.0);
        }
        ortho.scale
    } else {
        1.0
    };
    
    // Pan with Shift + Left Mouse Button drag
    let shift_pressed = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    
    if let Ok(window) = windows.single() {
        if let Some(cursor_pos) = window.cursor_position() {
            if buttons.pressed(MouseButton::Left) && shift_pressed {
                if let Some(last_pos) = drag_state.last_pos {
                    let delta = cursor_pos - last_pos;
                    // Move camera in opposite direction to simulate dragging the canvas
                    camera_transform.translation.x -= delta.x * scale;
                    camera_transform.translation.y += delta.y * scale;
                }
                drag_state.last_pos = Some(cursor_pos);
                drag_state.dragging = true;
            } else {
                drag_state.last_pos = Some(cursor_pos);
                if !buttons.pressed(MouseButton::Left) {
                    drag_state.dragging = false;
                }
            }
        }
    }
}

fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    sprite_query: Query<&Transform, With<CanvasSprite>>,
    canvas: Res<CanvasData>,
    client: Option<Res<GrpcClient>>,
    mut commands: Commands,
    runtime: Res<TokioRuntime>,
) {
    // Only draw if shift is NOT pressed (to avoid drawing while panning)
    let shift_pressed = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    
    if buttons.just_pressed(MouseButton::Left) && !shift_pressed {
        if let Some(client) = client {
            if let Ok(window) = windows.single() {
                if let Some(cursor_pos) = window.cursor_position() {
                    if let Ok((camera, camera_transform)) = camera_query.single() {
                        if let Ok(sprite_transform) = sprite_query.single() {
                            // Convert screen coordinates to world coordinates
                            if let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_pos) {
                                // The sprite is positioned at (center_x, center_y) in world space
                                // and is scaled by 4.0
                                let sprite_pos = sprite_transform.translation.truncate();
                                let sprite_scale = sprite_transform.scale.x; // Uniform scale
                                
                                // Calculate position relative to sprite center in world space
                                let relative_world = world_pos - sprite_pos;
                                
                                // Divide by scale to get position in unscaled sprite space
                                // In sprite space, the sprite is canvas.width x canvas.height pixels
                                // with origin at center, so ranges from -width/2 to +width/2
                                let sprite_local_x = relative_world.x / sprite_scale;
                                let sprite_local_y = relative_world.y / sprite_scale;
                                
                                // Convert from sprite-local coordinates (centered at 0,0) 
                                // to pixel coordinates (0-based from top-left)
                                // Note: Y is flipped (positive Y is up in world, down in pixel coords)
                                let pixel_x = (sprite_local_x + (canvas.width as f32 / 2.0)).floor() as i64;
                                let pixel_y = ((canvas.height as f32 / 2.0) - sprite_local_y).floor() as i64;
                                
                                // Convert from canvas-local pixel coords to global pixel coords
                                let global_pixel_x = pixel_x + canvas.center_x;
                                let global_pixel_y = pixel_y + canvas.center_y;
                                
                                println!("Click at screen: {:?}, world: {:?}, sprite_local: ({:.2}, {:.2}), canvas_pixel: ({}, {}), global_pixel: ({}, {})", 
                                         cursor_pos, world_pos, sprite_local_x, sprite_local_y,
                                         pixel_x, pixel_y, global_pixel_x, global_pixel_y);
                                
                                // Check bounds
                                if pixel_x >= 0 && pixel_x < canvas.width as i64 && 
                                   pixel_y >= 0 && pixel_y < canvas.height as i64 {
                                    // Send push_pixel with red color (ID 2)
                                    let client_clone = client.0.clone();
                                    let (tx, rx) = unbounded();
                                    let rt = runtime.0.clone();
                                    std::thread::spawn(move || {
                                        rt.block_on(push_pixel(global_pixel_x, global_pixel_y, 2, 1, client_clone));
                                        let _ = tx.send(());
                                    });
                                    commands.insert_resource(PushTask(rx));
                                } else {
                                    println!("Click outside canvas bounds");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

async fn push_pixel(x: i64, y: i64, color: u32, user_id: u64, client: Arc<Mutex<DataServiceClient<Channel>>>) {
    let request = tonic::Request::new(PushPixelRequest {
        x,
        y,
        color,
        user_id,
    });
    let mut client = client.lock().await;
    let _ = client.push_pixel(request).await;
}

#[derive(Resource)]
struct PushTask(Receiver<()>);

fn check_push(mut commands: Commands, mut push_task: Option<ResMut<PushTask>>, client: Option<Res<GrpcClient>>, runtime: Res<TokioRuntime>) {
    if let Some(task) = push_task.as_mut() {
        match task.0.try_recv() {
            Ok(_) => {
                commands.remove_resource::<PushTask>();
                // Trigger fetch after push
                if let Some(client) = client {
                    let client_clone = client.0.clone();
                    let (tx, rx) = unbounded();
                    let rt = runtime.0.clone();
                    std::thread::spawn(move || {
                        let result = rt.block_on(fetch_snapshot(client_clone));
                        let _ = tx.send(result);
                    });
                    commands.insert_resource(FetchTask(rx));
                }
            }
            Err(TryRecvError::Empty) => {
                // Task still running
            }
            Err(TryRecvError::Disconnected) => {
                // Task completed or failed
                commands.remove_resource::<PushTask>();
            }
        }
    }
}

fn periodic_update(
    time: Res<Time>,
    mut timer: ResMut<LastUpdateTimer>,
    client: Option<Res<GrpcClient>>,
    mut commands: Commands,
    runtime: Res<TokioRuntime>,
) {
    if timer.0.tick(time.delta()).just_finished() {
        if let Some(client) = client {
            let client_clone = client.0.clone();
            let (tx, rx) = unbounded();
            let rt = runtime.0.clone();
            std::thread::spawn(move || {
                let result = rt.block_on(fetch_snapshot(client_clone));
                let _ = tx.send(result);
            });
            commands.insert_resource(FetchTask(rx));
        }
    }
}