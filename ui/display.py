"""
Pygame-based ADAS Head-Up Display (HUD).
Renders camera feed, overlays, warnings, and telemetry in a single window.
Inspired by Pi-ADAS status display approach.
"""
import time
import os
import cv2
import numpy as np
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class AdasDisplay:
    """Real-time ADAS dashboard using Pygame."""

    # Colour palette
    # Premium BMW-style Palette
    COL_BG = (10, 11, 14)       # Deep charcoal/black
    COL_ACCENT = (0, 102, 255)  # BMW Electric Blue
    COL_ACCENT_DIM = (0, 40, 100)
    COL_TEXT = (230, 230, 240)
    COL_TEXT_DIM = (140, 145, 160)
    COL_GREEN = (0, 255, 120)
    COL_YELLOW = (255, 215, 0)
    COL_RED = (255, 40, 60)
    COL_ORANGE = (255, 120, 0)
    COL_PANEL = (25, 27, 35, 180) # Semi-transparent
    COL_BORDER = (60, 65, 80)

    def __init__(self, config):
        ui_cfg = config.get('ui', {})
        self.enabled = ui_cfg.get('enabled', True)
        self.width = ui_cfg.get('window_width', 1280)
        self.height = ui_cfg.get('window_height', 720)
        self.font_size = ui_cfg.get('font_size', 24)
        self.warn_ms = ui_cfg.get('warning_display_ms', 2000)

        self.screen = None
        self.font = None
        self.font_sm = None
        self.font_lg = None
        self.clock = pygame.time.Clock()
        self._warnings = []  # list of (text, colour, expire_time)
        self.view_mode = "focus"  # "focus" or "grid"
        self.focus_layer = "main" # "main", "depth", "seg", "lane"
        self.is_fullscreen = False
        self.selected_source = None
        self.available_cameras = []

        if self.enabled and PYGAME_AVAILABLE:
            pygame.init()
            # Try to enable hardware acceleration
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
            pygame.display.set_caption("BMW iDrive ADAS | Digital Cockpit")
            
            # Use Segoe UI on Windows for a cleaner look, fallback to Arial
            self.font_main = "segoeui" if "segoe" in pygame.font.get_fonts() else "arial"
            self.font = pygame.font.SysFont(self.font_main, self.font_size)
            self.font_sm = pygame.font.SysFont(self.font_main, self.font_size - 6)
            self.font_lg = pygame.font.SysFont(self.font_main, int(self.font_size * 1.5), bold=True)
            self.font_num = pygame.font.SysFont(self.font_main, int(self.font_size * 2.5), bold=True)
            
            self.clock = pygame.time.Clock()
            print(f"[UI] BMW-Style Dashboard initialized with font: {self.font_main}")
        else:
            if not PYGAME_AVAILABLE:
                print("[UI] Pygame not installed — display disabled.")

    def push_warning(self, text, colour=None):
        """Add a timed warning to the overlay."""
        if colour is None:
            colour = self.COL_RED
        expire = time.time() + self.warn_ms / 1000.0
        self._warnings.append((text, colour, expire))

    def render_menu(self):
        """Render a source selection menu. Returns selected source or None."""
        if not self.enabled or self.screen is None:
            return 0 # Default to 0

        # Scan for cameras if not already done
        if not self.available_cameras:
            self.available_cameras = self._scan_cameras()

        menu_running = True
        selected = None
        
        while menu_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0: selected = 0
                    if event.key == pygame.K_1: selected = 1
                    if event.key == pygame.K_2: selected = 2
                    if event.key == pygame.K_v: 
                        # Look for test videos
                        for f in ["test.mp4", "test.avi", "sample.mp4"]:
                            if os.path.exists(f):
                                selected = f
                                break
                        if not selected:
                            print("[UI] No test video found (test.mp4, etc.)")
                    if selected is not None:
                        menu_running = False

            self.screen.fill(self.COL_BG)
            self._draw_header({})
            
            y = 150
            self._draw_text("SELECT INPUT SOURCE", self.width//2 - 150, y, self.font_lg, self.COL_ACCENT)
            y += 60
            
            # Draw options
            for i, cam in enumerate(self.available_cameras):
                col = self.COL_GREEN if selected == i else self.COL_TEXT
                self._draw_text(f"[{i}] CAMERA INDEX {cam}", self.width//2 - 100, y, self.font, col)
                y += 40
            
            y += 20
            v_col = self.COL_GREEN if selected == "video" else self.COL_TEXT
            self._draw_text("[V] SELECT VIDEO FILE (Coming Soon)", self.width//2 - 150, y, self.font, v_col)
            
            self._draw_text("Press key to select...", self.width//2 - 80, self.height - 100, self.font_sm, self.COL_TEXT_DIM)
            
            pygame.display.flip()
            self.clock.tick(30)
            
        return selected

    def _scan_cameras(self, max_to_test=3):
        """Test camera indices and return available ones."""
        available = []
        for i in range(max_to_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available if available else [0]

    def render(self, frame, adas_state):
        """Render a full dashboard frame."""
        if not self.enabled or self.screen is None:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self._toggle_fullscreen()
                if event.key == pygame.K_g:
                    self.view_mode = "grid" if self.view_mode == "focus" else "focus"
                if event.key == pygame.K_v:
                    # Cycle focus layer
                    layers = ["main", "depth", "seg", "lane"]
                    curr_idx = layers.index(self.focus_layer)
                    self.focus_layer = layers[(curr_idx + 1) % len(layers)]

        self.screen.fill(self.COL_BG)

        if self.view_mode == "grid":
            self._render_grid_mode(frame, adas_state)
        else:
            self._render_focus_mode(frame, adas_state)

        # --- Top Header (Common) ---
        self._draw_header(adas_state)
        # --- Warning Overlays (Common) ---
        self._render_warnings()

        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)
        return True

    def _toggle_fullscreen(self):
        """Toggle between windowed and fullscreen mode."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height), 
                                                 pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height), 
                                                 pygame.DOUBLEBUF | pygame.HWSURFACE)

    def _render_focus_mode(self, frame, adas_state):
        """Standard high-end dashboard view or individual layer focus."""
        main_w = int(self.width * 0.65)
        main_h = int(self.height * 0.78)
        main_x = (self.width - main_w) // 2
        main_y = 70

        self._draw_background_elements()

        # Determine what to show in the main window
        display_frame = frame
        label = "VISION FEED"
        
        if self.focus_layer == "depth":
            depth = adas_state.get('vision_raw', {}).get('depth_map')
            if depth is not None:
                display_frame = (depth * 255).astype(np.uint8)
                display_frame = cv2.applyColorMap(display_frame, cv2.COLORMAP_MAGMA)
                label = "DEPTH MAP (FOCUS)"
        elif self.focus_layer == "seg":
            display_frame = adas_state.get('vision_raw', {}).get('seg_overlay')
            label = "DRIVABLE AREA (FOCUS)"
        elif self.focus_layer == "lane":
            display_frame = adas_state.get('vision_raw', {}).get('lane_unet_overlay')
            label = "LANE SEGMENTATION (FOCUS)"

        if display_frame is not None:
            surf = self._cv2_to_surface(display_frame)
            surf = pygame.transform.scale(surf, (main_w, main_h))
            border_rect = pygame.Rect(main_x - 2, main_y - 2, main_w + 4, main_h + 4)
            pygame.draw.rect(self.screen, self.COL_BORDER, border_rect, 1, border_radius=10)
            self.screen.blit(surf, (main_x, main_y))
            
            # Label
            lbl = self.font_sm.render(label, True, self.COL_ACCENT)
            self.screen.blit(lbl, (main_x + 10, main_y + 10))

        # Gauges
        speed_mps = adas_state.get('ego', {}).get('ego_speed', 0)
        self._draw_speed_gauge(150, self.height // 2 + 50, 120, speed_mps * 3.6)
        
        ctrl = adas_state.get('control', {})
        steer = ctrl.get('steering', 90) - 90
        self._draw_steering_gauge(self.width - 150, self.height // 2 + 50, 120, steer)

        # Sensor Cluster (New)
        self._draw_sensor_cluster(adas_state)
        self._draw_bottom_bar(adas_state)

    def _render_grid_mode(self, frame, adas_state):
        """CP PLUS style grid view of all layers."""
        cols, rows = 2, 2
        pad = 10
        header_h = 60
        w = (self.width - (cols + 1) * pad) // cols
        h = (self.height - header_h - (rows + 1) * pad) // rows
        
        # 1. Main Feed
        self._draw_grid_item(frame, "VISION FEED", pad, header_h + pad, w, h)
        
        # 2. Depth Map
        depth = adas_state.get('vision_raw', {}).get('depth_map')
        if depth is not None:
            depth_vis = (depth * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            self._draw_grid_item(depth_vis, "DEPTH (MIDAS)", pad*2 + w, header_h + pad, w, h)
        else:
            self._draw_grid_placeholder("DEPTH (MIDAS)", pad*2 + w, header_h + pad, w, h)

        # 3. Segmentation
        seg = adas_state.get('vision_raw', {}).get('seg_overlay')
        self._draw_grid_item(seg, "DRIVABLE AREA", pad, header_h + pad*2 + h, w, h)
        
        # 4. Lane UNet
        lane = adas_state.get('vision_raw', {}).get('lane_unet_overlay')
        self._draw_grid_item(lane, "LANE (UNET)", pad*2 + w, header_h + pad*2 + h, w, h)

    def _draw_grid_item(self, img, label, x, y, w, h):
        pygame.draw.rect(self.screen, self.COL_BORDER, (x-1, y-1, w+2, h+2), 1, border_radius=4)
        if img is not None:
            surf = self._cv2_to_surface(img)
            surf = pygame.transform.scale(surf, (w, h))
            self.screen.blit(surf, (x, y))
        else:
            pygame.draw.rect(self.screen, (20, 20, 30), (x, y, w, h))
        
        # Label overlay
        lbl_surf = self.font_sm.render(label, True, self.COL_ACCENT)
        self.screen.blit(lbl_surf, (x + 10, y + 10))

    def _draw_grid_placeholder(self, label, x, y, w, h):
        pygame.draw.rect(self.screen, (20, 20, 30), (x, y, w, h))
        pygame.draw.rect(self.screen, self.COL_BORDER, (x, y, w, h), 1)
        txt = self.font_sm.render(f"{label} [OFFLINE]", True, self.COL_TEXT_DIM)
        self.screen.blit(txt, (x + w//2 - txt.get_width()//2, y + h//2))

    def _draw_sensor_cluster(self, adas_state):
        """Fix for missing sensor data: Display ultrasonic/proximity info."""
        # Assume adas_state['sensors'] contains {'left': dist, 'right': dist, ...}
        sensors = adas_state.get('sensors', {})
        panel_w, panel_h = 240, 120
        x, y = self.width // 2 - panel_w // 2, self.height - 180
        
        # Glass panel
        s_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (30, 35, 45, 160), (0, 0, panel_w, panel_h), border_radius=10)
        self.screen.blit(s_surf, (x, y))
        
        # Sensor values
        self._draw_text("PROXIMITY SENSORS", x + 10, y + 10, self.font_sm, self.COL_ACCENT)
        
        l_dist = sensors.get('left', '--')
        r_dist = sensors.get('right', '--')
        
        col_l = self.COL_RED if isinstance(l_dist, (int, float)) and l_dist < 50 else self.COL_TEXT
        col_r = self.COL_RED if isinstance(r_dist, (int, float)) and r_dist < 50 else self.COL_TEXT
        
        self._draw_text(f"LEFT:  {l_dist} cm", x + 20, y + 45, self.font, col_l)
        self._draw_text(f"RIGHT: {r_dist} cm", x + 20, y + 75, self.font, col_r)

    def _draw_background_elements(self):
        """Draw subtle grid or background patterns."""
        for i in range(0, self.width, 100):
            pygame.draw.line(self.screen, (20, 22, 30), (i, 0), (i, self.height))
        for i in range(0, self.height, 100):
            pygame.draw.line(self.screen, (20, 22, 30), (0, i), (self.width, i))

    def _draw_speed_gauge(self, x, y, radius, val):
        """Draw a semi-circular speedometer."""
        # Gauge background arc
        rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        pygame.draw.arc(self.screen, self.COL_BORDER, rect, math.radians(150), math.radians(390), 12)
        
        # Value arc (Electric Blue)
        angle = 150 + (val / 160.0) * 240 # Range 0-160 kmh
        angle = min(390, angle)
        pygame.draw.arc(self.screen, self.COL_ACCENT, rect, math.radians(150), math.radians(angle), 14)

        # Text
        txt = self.font_num.render(f"{int(val)}", True, self.COL_TEXT)
        self.screen.blit(txt, (x - txt.get_width() // 2, y - 20))
        unit = self.font_sm.render("km/h", True, self.COL_TEXT_DIM)
        self.screen.blit(unit, (x - unit.get_width() // 2, y + 35))

    def _draw_steering_gauge(self, x, y, radius, steer_val):
        """Draw a steering angle visualization."""
        rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        pygame.draw.arc(self.screen, self.COL_BORDER, rect, math.radians(0), math.radians(180), 8)
        
        # Steering indicator needle-like arc
        steer_angle = 90 - (steer_val * 2) # multiplier for visualization
        start_r = math.radians(steer_angle - 10)
        end_r = math.radians(steer_angle + 10)
        pygame.draw.arc(self.screen, self.COL_ACCENT, rect, start_r, end_r, 12)

        txt = self.font_lg.render(f"STEER", True, self.COL_TEXT_DIM)
        self.screen.blit(txt, (x - txt.get_width() // 2, y - 20))
        val_txt = self.font.render(f"{int(steer_val)}°", True, self.COL_TEXT)
        self.screen.blit(val_txt, (x - val_txt.get_width() // 2, y + 10))

    def _draw_header(self, adas_state):
        # Background bar
        header_h = 50
        # Glassmorphism effect (simulated with semi-transp)
        hdr_surf = pygame.Surface((self.width, header_h), pygame.SRCALPHA)
        pygame.draw.rect(hdr_surf, (30, 35, 45, 200), (0, 0, self.width, header_h))
        self.screen.blit(hdr_surf, (0, 0))
        
        # Title
        title = self.font_lg.render("BMW ADAS | INTELLIGENT VISION", True, self.COL_ACCENT)
        self.screen.blit(title, (20, 10))
        
        # Time
        curr_time = time.strftime("%H:%M:%S")
        time_txt = self.font.render(curr_time, True, self.COL_TEXT)
        self.screen.blit(time_txt, (self.width - 120, 10))

    def _draw_bottom_bar(self, adas_state):
        bar_h = 40
        bar_y = self.height - bar_h
        pygame.draw.rect(self.screen, (20, 22, 28), (0, bar_y, self.width, bar_h))
        
        # FPS
        fps = adas_state.get('fps', 0)
        fps_txt = self.font_sm.render(f"SYSTEM FPS: {fps:.1f}", True, self.COL_GREEN)
        self.screen.blit(fps_txt, (20, bar_y + 10))
        
        # TTC Info
        fcw = adas_state.get('fcw', {})
        ttc = fcw.get('ttc')
        ttc_str = f"TTC: {ttc:.1f}s" if ttc is not None else "TTC: --"
        ttc_col = self.COL_RED if ttc and ttc < 2 else self.COL_TEXT
        ttc_txt = self.font_sm.render(ttc_str, True, ttc_col)
        self.screen.blit(ttc_txt, (200, bar_y + 10))
        
        # System Info & Shortcuts
        shrt = self.font_sm.render("[F] Fullscreen | [G] Grid | [V] Cycle Layer", True, self.COL_TEXT_DIM)
        self.screen.blit(shrt, (self.width // 2 - shrt.get_width() // 2, bar_y + 10))
        
        # Hardware Status
        h_status = "HW: CONNECTED" if adas_state.get('hardware_ok', True) else "HW: ERROR"
        h_col = self.COL_ACCENT if "CONNECTED" in h_status else self.COL_RED
        hw_txt = self.font_sm.render(h_status, True, h_col)
        self.screen.blit(hw_txt, (self.width - 150, bar_y + 10))

    def _render_warnings(self):
        now = time.time()
        self._warnings = [(t, c, e) for t, c, e in self._warnings if e > now]
        
        if not self._warnings:
            return

        # Show only the latest critical warning as a large banner
        text, colour, _ = self._warnings[-1]
        
        banner_h = 70
        banner_y = 120
        # Pulsing effect
        alpha = int(180 + 75 * math.sin(time.time() * 10))
        
        surf = pygame.Surface((self.width, banner_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, colour + (alpha,), (0, 0, self.width, banner_h))
        self.screen.blit(surf, (0, banner_y))
        
        msg = self.font_lg.render(f"⚠  {text}  ⚠", True, (255, 255, 255))
        self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, banner_y + 15))

    def cleanup(self):
        if PYGAME_AVAILABLE and self.enabled:
            pygame.quit()
            print("[UI] Pygame closed.")

    # ---- helpers ----

    def _cv2_to_surface(self, frame):
        """Convert BGR numpy image to a Pygame surface."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.rot90(rgb)
        rgb = np.flipud(rgb)
        return pygame.surfarray.make_surface(rgb)

    def _draw_text(self, text, x, y, font, colour):
        """Draw text and return new y position below it."""
        if font is None:
            return y + 20
        surface = font.render(text, True, colour)
        self.screen.blit(surface, (x, y))
        return y + surface.get_height() + 4
