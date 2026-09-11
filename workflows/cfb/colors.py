"""Pairing two teams' brand colours so both stay legible and tell apart.

Two problems have to be solved together:

1. **Legibility.** The UI paints on a near-black background, and 111 of the
   138 FBS primaries are navy, maroon or dark green — they fail a 3:1 contrast
   ratio there. Each colour is lifted in OKLCH just far enough to clear the
   floor, which keeps the hue (Georgia stays crimson rather than turning pink).

2. **Telling the two sides apart.** Plenty of schools share a colour — Air
   Force and Pittsburgh are the same navy to the byte, and Tennessee's orange
   sits a mere 12.6 ΔE from Clemson's. The home team keeps its primary; a too-
   close away team falls back to its alternate, then to a neutral grey.

Similarity is judged on the *lifted* colours, because those are what a viewer
actually sees: two different navies that lift to the same blue still collide.
"""
import math
from typing import Optional, Tuple

# The UI's chart background (sports-ui src/index.css --background).
BACKGROUND = "#201D1D"

# WCAG 2.1 non-text contrast minimum for graphical objects.
CONTRAST_FLOOR = 3.0

# CIELAB ΔE below which two colours read as "the same". Tennessee/Clemson is
# 12.6; Tennessee/Georgia Tech is 61.
SIMILARITY_DELTA_E = 15.0

# Shown when a team has no usable colour left. Matches the UI's away-grey.
NEUTRAL = "#8a8a94"

Rgb = Tuple[float, float, float]


def _srgb_to_linear(u: float) -> float:
    return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(u: float) -> float:
    return 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1 / 2.4) - 0.055


def _parse(hex_color: str) -> Rgb:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"not a hex colour: {hex_color!r}")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _format(rgb: Rgb) -> str:
    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in rgb)


def relative_luminance(hex_color: str) -> float:
    r, g, b = (_srgb_to_linear(c) for c in _parse(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _to_lab(hex_color: str) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in _parse(hex_color))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    """CIE76 colour difference. Good enough to answer 'same colour?'."""
    la, lb = _to_lab(a), _to_lab(b)
    return math.dist(la, lb)


def _to_oklch(hex_color: str) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in _parse(hex_color))
    l = (0.4122 * r + 0.5364 * g + 0.0514 * b) ** (1 / 3)
    m = (0.2119 * r + 0.6807 * g + 0.1074 * b) ** (1 / 3)
    s = (0.0883 * r + 0.2818 * g + 0.6300 * b) ** (1 / 3)
    L = 0.2105 * l + 0.7936 * m - 0.0040 * s
    A = 1.9780 * l - 2.4286 * m + 0.4506 * s
    B = 0.0259 * l + 0.7828 * m - 0.8087 * s
    return L, math.hypot(A, B), math.atan2(B, A)


def _from_oklch(L: float, C: float, H: float) -> str:
    A, B = C * math.cos(H), C * math.sin(H)
    l = (L + 0.3963 * A + 0.2158 * B) ** 3
    m = (L - 0.1055 * A - 0.0638 * B) ** 3
    s = (L - 0.0894 * A - 1.2914 * B) ** 3
    r = 4.0767 * l - 3.3077 * m + 0.2309 * s
    g = -1.2684 * l + 2.6097 * m - 0.3413 * s
    b = -0.0041 * l - 0.7034 * m + 1.7076 * s
    return _format(tuple(_linear_to_srgb(min(1.0, max(0.0, c))) for c in (r, g, b)))  # type: ignore[arg-type]


def ensure_legible(hex_color: str, floor: float = CONTRAST_FLOOR) -> str:
    """Raise lightness only as far as the contrast floor demands, keeping hue.

    Binary search rather than a fixed target band: lifting Penn State's navy to
    the palette's brightness would turn it into a pale periwinkle, whereas the
    minimum lift leaves it recognisably navy-blue.
    """
    if contrast_ratio(hex_color, BACKGROUND) >= floor:
        return hex_color
    L, C, H = _to_oklch(hex_color)
    lo, hi = L, 1.0
    if contrast_ratio(_from_oklch(hi, C, H), BACKGROUND) < floor:
        return NEUTRAL
    for _ in range(40):
        mid = (lo + hi) / 2
        if contrast_ratio(_from_oklch(mid, C, H), BACKGROUND) >= floor:
            hi = mid
        else:
            lo = mid
    return _from_oklch(hi, C, H)


TeamColors = Tuple[Optional[str], Optional[str]]


def _usable(hex_color: Optional[str]) -> Optional[str]:
    if not hex_color:
        return None
    try:
        lifted = ensure_legible(hex_color)
    except ValueError:
        return None
    return None if lifted == NEUTRAL else lifted


def resolve_pair(home: TeamColors, away: TeamColors) -> Tuple[str, str]:
    """Pick the rendered colour for each side.

    The home team keeps its primary. The away team keeps its own unless that
    would read as the same colour, in which case it tries its alternate and
    then gives up and goes neutral.
    """
    home_color = _usable(home[0]) or _usable(home[1]) or NEUTRAL

    for candidate in (away[0], away[1]):
        resolved = _usable(candidate)
        if resolved and delta_e(resolved, home_color) >= SIMILARITY_DELTA_E:
            return home_color, resolved
    return home_color, NEUTRAL
