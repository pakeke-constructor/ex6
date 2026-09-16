import ex6
import time
import math


VERTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]
EDGES = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7),
]


def rotate(v, ax, ay, az):
    x, y, z = v
    y, z = y*math.cos(ax) - z*math.sin(ax), y*math.sin(ax) + z*math.cos(ax)
    x, z = x*math.cos(ay) + z*math.sin(ay), -x*math.sin(ay) + z*math.cos(ay)
    x, y = x*math.cos(az) - y*math.sin(az), x*math.sin(az) + y*math.cos(az)
    return (x, y, z)


def project(v, scale, ox, oy):
    x, y, z = v
    z = z + 3
    return (int(ox + x / z * scale), int(oy + y / z * scale * 0.5))


def draw_line(buf, x0, y0, x1, y1, char='*', color='cyan', bg=None):
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
    pts = [project(rotate(v, t * 2.1, t * 3.0, t * 0.9), size * 0.35, cx, cy) for v in VERTS]
    for i, j in EDGES:
        draw_line(buf, pts[i][0], pts[i][1], pts[j][0], pts[j][1], char='#', color='cyan', bg=bg)


@ex6.override
def render_work_mode(tui, buf, inpt, r):
    tui.app.get_implementation('render_work_mode', default=True)(tui, buf, inpt, r)
    ctx = tui.current
    if not ctx.is_running():
        return

    x, y, w, h = r
    cx, cy = x + w // 2, y + h // 2
    size = min(w, h * 2) * 0.8
    scale = size * 0.35
    cube_w, cube_h = int(scale) + 2, int(scale * 0.5) + 2
    bx, by = cx - cube_w // 2, cy - cube_h // 2
    buf.fill((bx, by, cube_w, cube_h), char=' ', bg_color='black')
    elapsed = time.time() - ctx.last_invoke_time_start
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    header = f" {mins}:{secs:02d} "
    buf.puts(cx - len(header) // 2, by - 1, header, txt_color='white', bg_color='red')
    draw_cube(buf, cx, cy, size, time.time(), bg='black')
