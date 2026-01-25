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

def draw_line(buf, x0, y0, x1, y1, char='*', color='cyan', bg=None):
    # Bresenham's line
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        buf.put(x0, y0, char, txt_color=color, bg_color=bg)
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy

def draw_cube(buf, cx, cy, size, t, bg='black'):
    ax = t * 2.1
    ay = t * 3.0
    az = t * 0.9
    scale = size * 0.35

    # Rotate and project all vertices
    pts = [project(rotate(v, ax, ay, az), scale, cx, cy) for v in VERTS]

    # Draw edges
    for i, j in EDGES:
        draw_line(buf, pts[i][0], pts[i][1], pts[j][0], pts[j][1], char='#', color='cyan', bg=bg)

@ex6.override
def render_work_mode(buf, inpt, r):
    _orig_render_work_mode(buf, inpt, r)

    ctx = ex6.state.current
    if not ctx.is_running():
        return

    x, y, w, h = r

    # Cube dimensions
    cx = x + w // 2
    cy = y + h // 2
    size = min(w, h * 2) * 0.8  # smaller cube
    scale = size * 0.35

    # Cube projected extent: max at z=2 gives 0.5*scale, so full width=scale, height=scale*0.5
    # Add 2 for 1px padding on each side
    cube_w = int(scale) + 2
    cube_h = int(scale * 0.5) + 2

    # Background behind cube
    bx = cx - cube_w // 2
    by = cy - cube_h // 2
    buf.fill((bx, by, cube_w, cube_h), char=' ', bg_color='black')

    # Header with elapsed time, 1 unit above cube bg
    elapsed = time.time() - ctx.last_invoke_time_start
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    header = f" {mins}:{secs:02d} "
    buf.puts(cx - len(header) // 2, by - 1, header, txt_color='white', bg_color='red')

    draw_cube(buf, cx, cy, size, time.time(), bg='black')
