from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List


@dataclass
class SymbolicUI:
    # App behavior
    print_hz: float = 30.0
    mode: str = "library"   # library | outdoors | home
    muted: bool = False

    # internal pygame state
    _pygame_inited: bool = False
    _screen: Optional[object] = None
    _clock: Optional[object] = None
    _font_sm: Optional[object] = None
    _font_md: Optional[object] = None
    _font_lg: Optional[object] = None
    _last_frame: float = 0.0

    # app state
    _top_label: str = "other"
    _top_conf: float = 0.0
    _triggered: Optional[str] = None
    _rms: float = 0.0

    # haptic display
    _last_haptic_label: Optional[str] = None
    _last_haptic_ts: float = 0.0

    # recent alerts feed
    _feed: List[Tuple[float, str]] = field(default_factory=list)

    # UI layout regions
    _btn_mode: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    _btn_mute: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def _ensure_pygame(self) -> None:
        if self._pygame_inited:
            return
        try:
            import pygame

            pygame.init()
            self._screen = pygame.display.set_mode((900, 520))
            pygame.display.set_caption("HearCue")
            self._clock = pygame.time.Clock()
            self._font_sm = pygame.font.SysFont(None, 22)
            self._font_md = pygame.font.SysFont(None, 32)
            self._font_lg = pygame.font.SysFont(None, 64)

            self._btn_mode = {
                "library": (20, 20, 150, 52),
                "outdoors": (190, 20, 170, 52),
                "home": (380, 20, 130, 52),
            }
            self._btn_mute = (740, 20, 140, 52)

            self._pygame_inited = True
        except Exception as e:
            print("UI disabled (pygame not available):", e)
            self._pygame_inited = False

    def _hit(self, pos, rect) -> bool:
        x, y = pos
        rx, ry, rw, rh = rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    def _pump_events(self) -> None:
        if not self._pygame_inited:
            return
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.muted = not self.muted
                if event.key == pygame.K_1:
                    self.mode = "library"
                if event.key == pygame.K_2:
                    self.mode = "outdoors"
                if event.key == pygame.K_3:
                    self.mode = "home"
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for name, rect in self._btn_mode.items():
                    if self._hit(pos, rect):
                        self.mode = name
                        return
                if self._hit(pos, self._btn_mute):
                    self.muted = not self.muted
                    return

    def note_haptic(self, label: str) -> None:
        self._last_haptic_label = label
        self._last_haptic_ts = time.time()

    def show(
        self,
        *,
        top_label: str,
        top_conf: float,
        margin: float,
        rms: float,
        spec_min: float | None,
        spec_max: float | None,
        spec_std: float | None,
        triggered: str | None,
    ) -> None:
        self._ensure_pygame()
        if not self._pygame_inited:
            return

        self._top_label = top_label
        self._top_conf = float(top_conf)
        self._rms = float(rms)

        if triggered:
            self._triggered = triggered
            self._feed.append((time.time(), triggered))
            self._feed = self._feed[-5:]

        now = time.time()
        if now - self._last_frame < (1.0 / max(self.print_hz, 1.0)):
            return
        self._last_frame = now

        self._render()

    def _render(self) -> None:
        import pygame

        self._pump_events()

        BG = (14, 16, 20)
        CARD = (24, 28, 36)
        TEXT = (240, 240, 240)
        SUBT = (180, 190, 205)
        ACCENT = (0, 170, 255)
        MUTED = (100, 100, 110)

        ALERT = (255, 70, 70)
        OK = (60, 220, 140)

        self._screen.fill(BG)

        # Top buttons
        for name, rect in self._btn_mode.items():
            x, y, w, h = rect
            is_active = (name == self.mode)
            color = ACCENT if is_active else CARD
            pygame.draw.rect(self._screen, color, rect, border_radius=14)
            self._screen.blit(self._font_md.render(name.capitalize(), True, TEXT), (x + 16, y + 14))

        mx, my, mw, mh = self._btn_mute
        pygame.draw.rect(self._screen, (ALERT if self.muted else CARD), self._btn_mute, border_radius=14)
        self._screen.blit(self._font_md.render("Muted" if self.muted else "Mute", True, TEXT), (mx + 22, my + 14))

        # Main card
        pygame.draw.rect(self._screen, CARD, (20, 90, 860, 260), border_radius=22)

        if self.muted:
            state_title = "Muted"
            state_color = MUTED
            subtitle = "No alerts will be issued."
            big_label = "—"
        else:
            if self._rms < 0.03:
                state_title = "Quiet"
                state_color = SUBT
                subtitle = "Listening…"
                big_label = "—"
            else:
                if self._feed:
                    last_ts, last_lbl = self._feed[-1]
                    if (time.time() - last_ts) < 1.2:
                        state_title = "Alert"
                        state_color = ALERT
                        subtitle = "Sound detected"
                        big_label = last_lbl.upper()
                    else:
                        state_title = "Listening"
                        state_color = OK
                        subtitle = "Monitoring environment"
                        big_label = self._top_label.upper()
                else:
                    state_title = "Listening"
                    state_color = OK
                    subtitle = "Monitoring environment"
                    big_label = self._top_label.upper()

        self._screen.blit(self._font_md.render(state_title, True, state_color), (50, 120))
        self._screen.blit(self._font_sm.render(subtitle, True, SUBT), (50, 155))
        self._screen.blit(self._font_lg.render(big_label, True, TEXT), (50, 190))

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 50, 315, 520, 14
        pygame.draw.rect(self._screen, (40, 45, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=8)
        fill_w = int(bar_w * max(0.0, min(1.0, self._top_conf)))
        pygame.draw.rect(self._screen, ACCENT, (bar_x, bar_y, fill_w, bar_h), border_radius=8)
        self._screen.blit(self._font_sm.render(f"Confidence: {self._top_conf:.2f}", True, SUBT), (bar_x, bar_y - 24))

        # Haptic indicator
        if self._last_haptic_label and (time.time() - self._last_haptic_ts) < 1.2:
            htxt = f"Vibration sent: {self._last_haptic_label.upper()}"
            self._screen.blit(self._font_md.render(htxt, True, ALERT), (590, 300))

        # Recent feed
        pygame.draw.rect(self._screen, CARD, (20, 370, 860, 130), border_radius=22)
        self._screen.blit(self._font_md.render("Recent", True, TEXT), (50, 395))

        x0, y0 = 50, 435
        if not self._feed:
            self._screen.blit(self._font_sm.render("No alerts yet.", True, SUBT), (x0, y0))
        else:
            for i, (ts, lbl) in enumerate(reversed(self._feed[-5:])):
                t_ago = time.time() - ts
                s = f"{lbl.upper():10s}  •  {t_ago:4.1f}s ago"
                self._screen.blit(self._font_sm.render(s, True, SUBT), (x0, y0 + i * 22))

        pygame.display.flip()
        if self._clock:
            self._clock.tick(60)
