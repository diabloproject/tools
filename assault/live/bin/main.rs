use assault::live::ui::{Button, Context, Texture, UIHost, UIWidget};
use pixels::{Pixels, SurfaceTexture};
use std::ffi::c_void;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D11::{D3D11_CREATE_DEVICE_FLAG, D3D11_SDK_VERSION};
use windows::{
    Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE,
    Win32::Graphics::Direct3D11::{
        D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
        D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
    },
    Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_OUTDUPL_FRAME_INFO, IDXGIAdapter, IDXGIFactory1, IDXGIOutput,
        IDXGIOutput1, IDXGIOutputDuplication, IDXGIResource,
    },
    core::Interface,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Clone)]
#[allow(dead_code)]
struct CapturedFrame {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

struct ScreenCapture {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    width: u32,
    height: u32,
}

impl ScreenCapture {
    fn new() -> windows::core::Result<Self> {
        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL_11_0;

        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE(0 as *mut c_void),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                Some(&mut feature_level),
                Some(&mut context),
            )?;
        }

        let device = device.unwrap();
        let context = context.unwrap();

        // Get primary output (monitor) via DXGI
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        let adapter: IDXGIAdapter = unsafe { factory.EnumAdapters(0)? };
        let output: IDXGIOutput = unsafe { adapter.EnumOutputs(0)? };

        // Duplicate the output
        let output1: IDXGIOutput1 = output.cast()?;
        let duplication: IDXGIOutputDuplication = unsafe { output1.DuplicateOutput(&device)? };

        // Capture one frame to get dimensions
        let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource_opt: Option<IDXGIResource> = None;
        unsafe { duplication.AcquireNextFrame(1000, &mut frame_info, &mut resource_opt)? };

        let resource: IDXGIResource = resource_opt.unwrap();
        let tex: ID3D11Texture2D = resource.cast()?;

        let desc = unsafe {
            let mut d = D3D11_TEXTURE2D_DESC::default();
            tex.GetDesc(&mut d);
            d
        };

        let width = desc.Width;
        let height = desc.Height;

        unsafe {
            duplication.ReleaseFrame()?;
        }

        println!("ScreenCapture initialized: {}x{}", width, height);

        Ok(Self {
            device,
            context,
            duplication,
            width,
            height,
        })
    }

    fn capture_frame(&self) -> windows::core::Result<CapturedFrame> {
        // Acquire next frame
        let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource_opt: Option<IDXGIResource> = None;

        // Wait up to 16ms for a frame (roughly 60fps)
        let hr = unsafe {
            self.duplication
                .AcquireNextFrame(16, &mut frame_info, &mut resource_opt)
        };
        if let Err(_) = hr {
            // Timeout or error - return previous frame or error
            return Err(hr.unwrap_err());
        }

        let resource: IDXGIResource = resource_opt.unwrap();

        // The resource is a texture; get ID3D11Texture2D
        let tex: ID3D11Texture2D = resource.cast()?;

        // Describe texture (to create a staging copy)
        let desc = unsafe {
            let mut d = D3D11_TEXTURE2D_DESC::default();
            tex.GetDesc(&mut d);
            d
        };

        // Create a staging texture for CPU readback
        let mut staging_desc = desc;
        staging_desc.Usage = D3D11_USAGE_STAGING;
        staging_desc.BindFlags = 0;
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        staging_desc.MiscFlags = 0;

        let mut staging_tex: Option<ID3D11Texture2D> = None;
        unsafe {
            self.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging_tex as *mut _))?
        };
        let staging_tex = staging_tex.unwrap();

        // Copy from GPU texture to staging texture
        unsafe {
            self.context.CopyResource(&staging_tex, &tex);
        }

        // Map the staging texture to read pixels
        let mut mapped = windows::Win32::Graphics::Direct3D11::D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(&staging_tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
        }

        // Copy into Vec<u8>, accounting for RowPitch
        // Format is BGRA8 (4 bytes per pixel)
        let bpp = 4;
        let row_pitch = mapped.RowPitch as usize;
        let mut pixels = Vec::<u8>::with_capacity((self.width * self.height) as usize * bpp);

        unsafe {
            let src_ptr = mapped.pData as *const u8;
            for y in 0..self.height as usize {
                let row_start = src_ptr.add(y * row_pitch);
                let row_slice = std::slice::from_raw_parts(row_start, self.width as usize * bpp);
                pixels.extend_from_slice(row_slice);
            }

            // Unmap and release frame
            self.context.Unmap(&staging_tex, 0);
            self.duplication.ReleaseFrame()?;
        }

        // Convert BGRA to RGBA in-place
        bgra_to_rgba_in_place(&mut pixels);

        Ok(CapturedFrame {
            pixels,
            width: self.width,
            height: self.height,
        })
    }
}

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    frame_receiver: mpsc::Receiver<CapturedFrame>,
    current_frame: Option<CapturedFrame>,
    initial_width: u32,
    initial_height: u32,
    width: u32,
    height: u32,
    modified: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("Screen Capture Display")
                .with_inner_size(winit::dpi::PhysicalSize::new(
                    self.initial_width,
                    self.initial_height,
                ));

            self.window = Some(Arc::new(event_loop.create_window(window_attrs).unwrap()));
        }

        if self.pixels.is_none() {
            if let Some(window) = &self.window {
                let window_size = window.inner_size();
                let surface_texture =
                    SurfaceTexture::new(window_size.width, window_size.height, window.clone());

                let pixels =
                    Pixels::new(self.initial_width, self.initial_height, surface_texture).unwrap();

                self.pixels = Some(pixels);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Close requested, exiting...");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Try to receive latest frame(s)
                while let Ok(frame) = self.frame_receiver.try_recv() {
                    self.current_frame = Some(frame);
                }

                if let (Some(pixels), Some(frame)) = (&mut self.pixels, &self.current_frame) {
                    if self.modified {
                        pixels
                            .resize_buffer(self.width, self.height)
                            .expect("TODO: panic message");
                        self.modified = false;
                    }
                    // Copy frame data to pixels buffer
                    let pixel_frame = pixels.frame_mut();

                    // Scale frame to current window size
                    scale_frame(
                        &frame.pixels,
                        pixel_frame,
                        frame.width,
                        frame.height,
                        self.width,
                        self.height,
                    );

                    let mut texture = Texture {
                        width: self.width,
                        height: self.height,
                        data: pixel_frame,
                    };

                    let mut ctx = Context {
                        paint_texture: texture,
                        host: &UIHost {},
                        vdt: 0.0,
                        dt: 0.0,
                    };

                    Button::new((0, 0), (100, 100), "Test")
                        .with_border([0, 0, 0, 255], 4)
                        .render(&mut ctx);

                    if let Err(err) = pixels.render() {
                        eprintln!("pixels.render() failed: {err}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(pixels) = &mut self.pixels {
                    if let Err(err) = pixels.resize_surface(size.width, size.height) {
                        eprintln!("pixels.resize_surface() failed: {err}");
                        event_loop.exit();
                    } else {
                        self.width = size.width;
                        self.height = size.height;
                        self.modified = true;
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing screen capture...");

    // Create the screen capture instance
    let capture = ScreenCapture::new()?;
    let width = capture.width;
    let height = capture.height;

    // Create channel for sending frames
    let (frame_sender, frame_receiver) = mpsc::channel();

    // Spawn capture thread
    println!("Spawning capture thread...");
    thread::spawn(move || {
        println!("Capture thread started");
        let mut frame_count = 0u64;

        loop {
            match capture.capture_frame() {
                Ok(frame) => {
                    frame_count += 1;
                    if frame_count % 60 == 0 {
                        println!("Captured {} frames", frame_count);
                    }

                    if frame_sender.send(frame).is_err() {
                        println!("Frame receiver disconnected, stopping capture thread");
                        break;
                    }
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    });

    println!("Creating window and event loop...");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        window: None,
        pixels: None,
        frame_receiver,
        current_frame: None,
        initial_width: width,
        initial_height: height,
        width,
        height,
        modified: false,
    };

    event_loop.run_app(&mut app)?;

    Ok(())
}

// Simple channel swap if you want RGBA instead of BGRA
fn bgra_to_rgba_in_place(pixels: &mut [u8]) {
    for px in pixels.chunks_exact_mut(4) {
        px.swap(0, 2); // B <-> R
    }
}


pub fn scale_frame(
    src: &[u8],
    dst: &mut [u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) {
    let x_ratio = ((src_width  as u64) << 32) / dst_width  as u64;
    let y_ratio = ((src_height as u64) << 32) / dst_height as u64;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let src_stride = (src_width * 4) as usize;
    let dst_stride = (dst_width * 4) as usize;

    unsafe {
        let mut y_acc = 0u64;

        for dy in 0..dst_height {
            let sy = (y_acc >> 32).min((src_height - 1) as u64) as u32;
            y_acc += y_ratio;

            let src_row = src_ptr.add((sy * src_stride as u32) as usize);
            let dst_row = dst_ptr.add((dy * dst_stride as u32) as usize);

            let mut x_acc = 0u64;

            for dx in 0..dst_width {
                let sx = (x_acc >> 32).min((src_width - 1) as u64) as u32;
                x_acc += x_ratio;

                let s = src_row.add((sx * 4) as usize);
                let d = dst_row.add((dx * 4) as usize);

                std::ptr::copy_nonoverlapping(s, d, 4);
            }
        }
    }
}
