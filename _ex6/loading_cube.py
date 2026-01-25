import ex6
import time
import math

# Save original before overriding
_orig_render_work_mode = ex6.OVERRIDES['render_work_mode']

# Cube vertices (centered at origin, size 1)
VERTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]
# Edges as vertex index pairs
EDGES = [
    (0,1), (1,2), (2,3), (3,0),  # back face
    (4,5), (5,6), (6,7), (7,4),  # front face
    (0,4), (1,5), (2,6), (3,7),  # connecting edges
]

def rotate(v, ax, ay, az):
    x, y, z = v
    # Rotate around X
    y, z = y*math.cos(ax) - z*math.sin(ax), y*math.sin(ax) + z*math.cos(ax)
    # Rotate around Y
    x, z = x*math.cos(ay) + z*math.sin(ay), -x*math.sin(ay) + z*math.cos(ay)
    # Rotate around Z
    x, y = x*math.cos(az) - y*math.sin(az), x*math.sin(az) + y*math.cos(az)
    return (x, y, z)

def project(v, scale, ox, oy):
    x, y, z = v
    z = z + 3  # push back so z is always positive
    sx = x / z * scale
    sy = y / z * scale * 0.5  # 0.5 for terminal aspect
    return (int(ox + sx), int(oy + sy))

def draw_line(buf, x0, y0, x1, y1, char='*', color='cyan'):
    # Bresenham's line
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        buf.put(x0, y0, char, txt_color=color)
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy

def draw_cube(buf, cx, cy, size, t):
    ay = t * 3.0  # Y-axis only, faster spin
    scale = size * 0.35

    # Rotate and project all vertices
    pts = [project(rotate(v, 0, ay, 0), scale, cx, cy) for v in VERTS]

    # Draw edges
    for i, j in EDGES:
        draw_line(buf, pts[i][0], pts[i][1], pts[j][0], pts[j][1], char='#', color='cyan')

@ex6.override
def render_work_mode(buf, inpt, r):
    _orig_render_work_mode(buf, inpt, r)

    ctx = ex6.state.current
    if not ctx.is_running():
        return

    # Draw cube centered, filling the screen
    x, y, w, h = r
    cx = x + w // 2
    cy = y + h // 2
    size = min(w, h * 2)  # *2 to account for terminal aspect

    draw_cube(buf, cx, cy, size, time.time())
