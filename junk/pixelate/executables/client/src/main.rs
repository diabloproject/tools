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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource)]
enum RenderMode {
    Texture,
    Entities,
}

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
        .insert_resource(RenderMode::Texture)
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (
            check_connection, 
            check_fetch, 
            check_push, 
            update_canvas_texture.run_if(resource_equals(RenderMode::Texture)),
            update_canvas_entities.run_if(resource_equals(RenderMode::Entities)),
            handle_input, 
            camera_controls,
            periodic_update,
            animate_pixels,
            handle_mode_button,
            update_button_text,
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

#[derive(Resource)]
struct PixelSize(f32);

#[derive(Component)]
struct ModeButton;

#[derive(Component)]
struct ModeButtonText;

async fn connect_client() -> Result<DataServiceClient<Channel>, String> {
    DataServiceClient::connect("http://127.0.0.1:50051")
        .await
        .map_err(|e| e.to_string())
}

// Convert color ID (0-255) to RGB
fn color_id_to_rgb(id: u32) -> Color {
    let rgb = match id {
        0 => [0, 0, 0],
        1 => [255, 255, 255],
        2 => [255, 0, 0],
        3 => [0, 255, 0],
        4 => [0, 0, 255],
        5 => [255, 255, 0],
        6 => [255, 0, 255],
        7 => [0, 255, 255],
        8 => [128, 128, 128],
        255 => [240, 240, 240],
        _ => {
            let r = ((id * 137) % 256) as u8;
            let g = ((id * 211) % 256) as u8;
            let b = ((id * 97) % 256) as u8;
            [r, g, b]
        }
    };
    Color::srgb_u8(rgb[0], rgb[1], rgb[2])
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, runtime: Res<TokioRuntime>) {
    commands.spawn((
        Camera2d,
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

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
        pixels: vec![255; 10000],
    };

    // Create placeholder texture for texture mode
    let row_major_pixels = column_major_to_row_major(&canvas_data.pixels, canvas_data.width, canvas_data.height);
    let mut data = Vec::with_capacity((canvas_data.width * canvas_data.height * 4) as usize);
    for &color_id in &row_major_pixels {
        let color = color_id_to_rgb(color_id);
        let [r, g, b, _] = color.to_srgba().to_u8_array();
        data.extend_from_slice(&[r, g, b, 255]);
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
    image.sampler = ImageSampler::nearest();
    let image_handle = images.add(image);

    // Spawn sprite for texture mode (initially visible)
    // Position at world origin (0, 0), texture will be centered
    // Scale so each texture pixel = pixel_size world units
    let pixel_size_value = 4.0;
    commands.spawn((
        Sprite::from_image(image_handle.clone()),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0))
            .with_scale(Vec3::splat(pixel_size_value)),
        TextureCanvasSprite,
    ));

    commands.insert_resource(canvas_data);
    commands.insert_resource(UpdateCanvasFlag(true));
    commands.insert_resource(CanvasImageHandle(image_handle));
    commands.insert_resource(PixelSize(pixel_size_value));
    commands.insert_resource(LastUpdateTimer(Timer::from_seconds(0.1, TimerMode::Repeating)));
}

fn setup_ui(mut commands: Commands) {
    // UI root
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Start,
            align_items: AlignItems::Start,
            padding: UiRect::all(Val::Px(10.0)),
            ..default()
        },
    )).with_children(|parent| {
        // Mode toggle button
        parent.spawn((
            Button,
            Node {
                width: Val::Px(200.0),
                height: Val::Px(50.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
            ModeButton,
        )).with_children(|parent| {
            parent.spawn((
                Text::new("Mode: Texture"),
                TextFont {
                    font_size: 20.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                ModeButtonText,
            ));
        });
    });
}

// Convert column-major pixel data (server format) to row-major (image format)
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
            pixels: vec![255; 10000],
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
                println!("✗ Fetch task disconnected");
                commands.remove_resource::<FetchTask>();
            }
        }
    }
}

#[derive(Component)]
struct PixelEntity {
    x: i64,
    y: i64,
    color_id: u32,
}

#[derive(Component)]
struct TextureCanvasSprite;

#[derive(Component)]
struct AnimatePixel {
    timer: Timer,
    scale_factor: f32,
}

fn update_canvas_texture(
    canvas: Res<CanvasData>,
    mut images: ResMut<Assets<Image>>,
    image_handle: Res<CanvasImageHandle>,
    mut update_flag: ResMut<UpdateCanvasFlag>,
    texture_sprite: Query<Entity, With<TextureCanvasSprite>>,
) {
    if !update_flag.0 {
        return;
    }
    update_flag.0 = false;

    println!("Updating canvas texture...");
    
    if let Some(image) = images.get_mut(&image_handle.0) {
        let row_major_pixels = column_major_to_row_major(&canvas.pixels, canvas.width, canvas.height);
        let mut data = Vec::with_capacity((canvas.width * canvas.height * 4) as usize);
        for &color_id in &row_major_pixels {
            let color = color_id_to_rgb(color_id);
            let [r, g, b, _] = color.to_srgba().to_u8_array();
            data.extend_from_slice(&[r, g, b, 255]);
        }
        image.data = Some(data);
        image.resize(bevy::render::render_resource::Extent3d {
            width: canvas.width as u32,
            height: canvas.height as u32,
            depth_or_array_layers: 1,
        });
        println!("✓ Canvas texture updated!");
    }
}

fn update_canvas_entities(
    mut commands: Commands,
    canvas: Res<CanvasData>,
    pixel_size: Res<PixelSize>,
    mut update_flag: ResMut<UpdateCanvasFlag>,
    existing_pixels: Query<Entity, With<PixelEntity>>,
) {
    if !update_flag.0 {
        return;
    }
    update_flag.0 = false;

    println!("Updating canvas with entity-based pixels...");
    
    // Despawn existing pixels
    for entity in existing_pixels.iter() {
        commands.entity(entity).despawn();
    }
    
    // Server sends in column-major order: for x { for y { ... } }
    // So pixels[x * height + y] is the pixel at (x, y)
    for x in 0..canvas.width as i64 {
        for y in 0..canvas.height as i64 {
            let index = (x * canvas.height as i64 + y) as usize;
            let color_id = canvas.pixels.get(index).copied().unwrap_or(255);
            let color = color_id_to_rgb(color_id);
            
            // Calculate world position to match texture sprite positioning
            // Texture sprite is centered at (0, 0) with the canvas image
            // Image pixels go from (-width/2, -height/2) to (width/2, height/2) in texture space
            // Multiply by pixel_size to get world space
            
            // x and y are in canvas-local coords (0 to width/height)
            // Convert to centered coords (-width/2 to width/2)
            let centered_x = x as f32 - (canvas.width as f32 / 2.0) + 0.5;
            let centered_y = y as f32 - (canvas.height as f32 / 2.0) + 0.5;
            
            // Flip Y to match texture orientation (texture Y points down, world Y points up)
            let world_x = centered_x * pixel_size.0;
            let world_y = -centered_y * pixel_size.0;
            
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(pixel_size.0)),
                    ..default()
                },
                Transform::from_xyz(world_x, world_y, 0.0),
                PixelEntity {
                    x: x + canvas.center_x,
                    y: y + canvas.center_y,
                    color_id,
                },
            ));
        }
    }
    
    println!("✓ Created {} pixel entities!", canvas.width * canvas.height);
}

fn animate_pixels(
    time: Res<Time>,
    mut pixels: Query<(&mut Transform, &mut AnimatePixel)>,
) {
    for (mut transform, mut anim) in pixels.iter_mut() {
        anim.timer.tick(time.delta());
        if anim.timer.just_finished() {
            anim.timer.reset();
        }
        
        let progress = anim.timer.fraction();
        let scale = 1.0 + (progress * std::f32::consts::PI * 2.0).sin() * 0.2 * anim.scale_factor;
        transform.scale = Vec3::splat(scale);
    }
}

fn handle_mode_button(
    mut interaction_query: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<ModeButton>)>,
    mut mode: ResMut<RenderMode>,
    mut commands: Commands,
    texture_sprite: Query<Entity, With<TextureCanvasSprite>>,
    entity_pixels: Query<Entity, With<PixelEntity>>,
    mut update_flag: ResMut<UpdateCanvasFlag>,
) {
    for (interaction, mut color) in interaction_query.iter_mut() {
        match *interaction {
            Interaction::Pressed => {
                println!("Switching render mode...");
                *mode = match *mode {
                    RenderMode::Texture => {
                        // Switch to entities: hide texture sprite
                        for entity in texture_sprite.iter() {
                            commands.entity(entity).insert(Visibility::Hidden);
                        }
                        RenderMode::Entities
                    }
                    RenderMode::Entities => {
                        // Switch to texture: show texture sprite, hide entities
                        for entity in texture_sprite.iter() {
                            commands.entity(entity).insert(Visibility::Visible);
                        }
                        for entity in entity_pixels.iter() {
                            commands.entity(entity).despawn();
                        }
                        RenderMode::Texture
                    }
                };
                update_flag.0 = true; // Trigger update in new mode
                *color = BackgroundColor(Color::srgb(0.25, 0.25, 0.25));
            }
            Interaction::Hovered => {
                *color = BackgroundColor(Color::srgb(0.2, 0.2, 0.2));
            }
            Interaction::None => {
                *color = BackgroundColor(Color::srgb(0.15, 0.15, 0.15));
            }
        }
    }
}

fn update_button_text(
    mode: Res<RenderMode>,
    mut text_query: Query<&mut Text, With<ModeButtonText>>,
) {
    if !mode.is_changed() {
        return;
    }
    
    for mut text in text_query.iter_mut() {
        text.0 = match *mode {
            RenderMode::Texture => "Mode: Texture".to_string(),
            RenderMode::Entities => "Mode: Entities".to_string(),
        };
    }
}

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
    
    let scale = if let Projection::Orthographic(ref mut ortho) = projection.as_mut() {
        for event in scroll_events.read() {
            let zoom_delta = -event.y * 0.1;
            ortho.scale = (ortho.scale + zoom_delta).max(0.1).min(10.0);
        }
        ortho.scale
    } else {
        1.0
    };
    
    let shift_pressed = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    
    if let Ok(window) = windows.single() {
        if let Some(cursor_pos) = window.cursor_position() {
            if buttons.pressed(MouseButton::Left) && shift_pressed {
                if let Some(last_pos) = drag_state.last_pos {
                    let delta = cursor_pos - last_pos;
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
    canvas: Res<CanvasData>,
    pixel_size: Res<PixelSize>,
    mode: Res<RenderMode>,
    texture_sprite_query: Query<&Transform, With<TextureCanvasSprite>>,
    client: Option<Res<GrpcClient>>,
    mut commands: Commands,
    runtime: Res<TokioRuntime>,
) {
    let shift_pressed = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    
    if buttons.just_pressed(MouseButton::Left) && !shift_pressed {
        if let Some(client) = client {
            if let Ok(window) = windows.single() {
                if let Some(cursor_pos) = window.cursor_position() {
                    if let Ok((camera, camera_transform)) = camera_query.single() {
                        if let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_pos) {
                            let pixel_coords = match *mode {
                                RenderMode::Texture => {
                                    // Texture mode: account for sprite position and scale
                                    if let Ok(sprite_transform) = texture_sprite_query.single() {
                                        let sprite_pos = sprite_transform.translation.truncate();
                                        let sprite_scale = sprite_transform.scale.x;
                                        
                                        let sprite_local_x = (world_pos.x - sprite_pos.x) / sprite_scale;
                                        let sprite_local_y = (world_pos.y - sprite_pos.y) / sprite_scale;
                                        
                                        let pixel_x = (sprite_local_x + (canvas.width as f32 / 2.0)).floor() as i64;
                                        let pixel_y = ((canvas.height as f32 / 2.0) - sprite_local_y).floor() as i64;
                                        
                                        let global_pixel_x = pixel_x + canvas.center_x;
                                        let global_pixel_y = pixel_y + canvas.center_y;
                                        
                                        Some((global_pixel_x, global_pixel_y))
                                    } else {
                                        None
                                    }
                                }
                                RenderMode::Entities => {
                                    // Entity mode: world position to pixel coordinate
                                    // Entities are positioned centered at (0,0) just like texture
                                    // Convert world position to centered pixel coords
                                    let centered_x = world_pos.x / pixel_size.0;
                                    let centered_y = -world_pos.y / pixel_size.0; // Flip Y back
                                    
                                    // Convert from centered coords to canvas-local pixel coords
                                    let pixel_x = (centered_x + (canvas.width as f32 / 2.0)).floor() as i64;
                                    let pixel_y = (centered_y + (canvas.height as f32 / 2.0)).floor() as i64;
                                    
                                    // Convert to global pixel coords
                                    let global_pixel_x = pixel_x + canvas.center_x;
                                    let global_pixel_y = pixel_y + canvas.center_y;
                                    
                                    Some((global_pixel_x, global_pixel_y))
                                }
                            };
                            
                            if let Some((pixel_x, pixel_y)) = pixel_coords {
                                println!("Click at screen: {:?}, world: {:?}, pixel: ({}, {})", 
                                         cursor_pos, world_pos, pixel_x, pixel_y);
                                
                                // Send push_pixel with red color (ID 2)
                                let client_clone = client.0.clone();
                                let (tx, rx) = unbounded();
                                let rt = runtime.0.clone();
                                std::thread::spawn(move || {
                                    rt.block_on(push_pixel(pixel_x, pixel_y, 2, 1, client_clone));
                                    let _ = tx.send(());
                                });
                                commands.insert_resource(PushTask(rx));
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
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
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
