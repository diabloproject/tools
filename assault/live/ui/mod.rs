pub struct UIHost {

}

pub struct Texture<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a mut [u8],
}

pub struct Context<'txt> {
    /// Texture to draw to.
    pub paint_texture: Texture<'txt>,
    /// UI host
    pub host: &'txt UIHost,
    /// Virtual delta time.
    pub vdt: f32,
    /// Physical delta time, in seconds.
    pub dt: f32,
}

pub trait UIWidget {
    fn render(&self, ctx: &mut Context);
}

pub struct Button {
    pos: (u32, u32),
    size: (u32, u32),
    text: String,
    bg_color: [u8; 4], // RGBA
    text_color: [u8; 4], // RGBA
    border_color: [u8; 4], // RGBA
    border_width: u32,
    pressed: bool,
}

impl Button {
    pub fn new(pos: (u32, u32), size: (u32, u32), text: impl Into<String>) -> Self {
        Self {
            pos,
            size,
            text: text.into(),
            bg_color: [60, 60, 60, 255],        // Dark gray background
            text_color: [255, 255, 255, 255],   // White text
            border_color: [120, 120, 120, 255], // Light gray border
            border_width: 2,
            pressed: false,
        }
    }

    pub fn with_colors(mut self, bg_color: [u8; 4], text_color: [u8; 4]) -> Self {
        self.bg_color = bg_color;
        self.text_color = text_color;
        self
    }

    pub fn with_border(mut self, border_color: [u8; 4], border_width: u32) -> Self {
        self.border_color = border_color;
        self.border_width = border_width;
        self
    }

    pub fn set_pressed(&mut self, pressed: bool) {
        self.pressed = pressed;
    }

    pub fn is_pressed(&self) -> bool {
        self.pressed
    }
}

impl UIWidget for Button {
    fn render(&self, ctx: &mut Context) {
        let texture = &mut ctx.paint_texture;
        let (x, y) = self.pos;
        let (w, h) = self.size;

        // Adjust background color based on pressed state
        let bg_color = if self.pressed {
            [
                (self.bg_color[0] as f32 * 0.7) as u8,
                (self.bg_color[1] as f32 * 0.7) as u8,
                (self.bg_color[2] as f32 * 0.7) as u8,
                self.bg_color[3],
            ]
        } else {
            self.bg_color
        };

        // Draw filled rectangle for button background
        for py in y..y.saturating_add(h).min(texture.height) {
            for px in x..x.saturating_add(w).min(texture.width) {
                let idx = ((py * texture.width + px) * 4) as usize;
                if idx + 3 < texture.data.len() {
                    // Simple alpha blending
                    let alpha = bg_color[3] as f32 / 255.0;
                    let inv_alpha = 1.0 - alpha;

                    texture.data[idx] = ((bg_color[0] as f32 * alpha) + (texture.data[idx] as f32 * inv_alpha)) as u8;
                    texture.data[idx + 1] = ((bg_color[1] as f32 * alpha) + (texture.data[idx + 1] as f32 * inv_alpha)) as u8;
                    texture.data[idx + 2] = ((bg_color[2] as f32 * alpha) + (texture.data[idx + 2] as f32 * inv_alpha)) as u8;
                    texture.data[idx + 3] = 255;
                }
            }
        }

        // Draw border
        if self.border_width > 0 {
            let bw = self.border_width;

            // Top border
            for py in y..y.saturating_add(bw).min(texture.height) {
                for px in x..x.saturating_add(w).min(texture.width) {
                    draw_pixel(texture, px, py, &self.border_color);
                }
            }

            // Bottom border
            let bottom_start = y.saturating_add(h).saturating_sub(bw);
            for py in bottom_start..y.saturating_add(h).min(texture.height) {
                for px in x..x.saturating_add(w).min(texture.width) {
                    draw_pixel(texture, px, py, &self.border_color);
                }
            }

            // Left border
            for py in y..y.saturating_add(h).min(texture.height) {
                for px in x..x.saturating_add(bw).min(texture.width) {
                    draw_pixel(texture, px, py, &self.border_color);
                }
            }

            // Right border
            let right_start = x.saturating_add(w).saturating_sub(bw);
            for py in y..y.saturating_add(h).min(texture.height) {
                for px in right_start..x.saturating_add(w).min(texture.width) {
                    draw_pixel(texture, px, py, &self.border_color);
                }
            }
        }

        // TODO: Add text rendering when font system is available
        // For now, the button is just a colored rectangle
    }
}

#[inline]
fn draw_pixel(texture: &mut Texture, x: u32, y: u32, color: &[u8; 4]) {
    let idx = ((y * texture.width + x) * 4) as usize;
    if idx + 3 < texture.data.len() {
        let alpha = color[3] as f32 / 255.0;
        let inv_alpha = 1.0 - alpha;

        texture.data[idx] = ((color[0] as f32 * alpha) + (texture.data[idx] as f32 * inv_alpha)) as u8;
        texture.data[idx + 1] = ((color[1] as f32 * alpha) + (texture.data[idx + 1] as f32 * inv_alpha)) as u8;
        texture.data[idx + 2] = ((color[2] as f32 * alpha) + (texture.data[idx + 2] as f32 * inv_alpha)) as u8;
        texture.data[idx + 3] = 255;
    }
}