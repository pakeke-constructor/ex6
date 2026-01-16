

from typing import Union, Tuple, List, Optional

RegionLike = Union['Region', Tuple[int, int, int, int]]

def ensure_region(r: RegionLike) -> 'Region':
    if isinstance(r, Region):
        return r
    if isinstance(r, tuple) and len(r) == 4:
        return Region(*r)
    raise TypeError(f"Expected Region or (x,y,w,h) tuple")


class Region(tuple):
    def __new__(cls, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        return super().__new__(cls, (int(x), int(y), max(0, int(w)), max(0, int(h))))
    
    def __repr__(self):
        return f"Region{super().__repr__()}"
    
    def split_vertical(self, *ratios: float) -> List['Region']:
        norm = [r / sum(ratios) for r in ratios]
        regions = []
        accum_y = self[1]
        for ratio in norm:
            h = int(self[3] * ratio)
            regions.append(Region(self[0], accum_y, self[2], h))
            accum_y += h
        return regions
    
    def split_horizontal(self, *ratios: float) -> List['Region']:
        norm = [r / sum(ratios) for r in ratios]
        regions = []
        accum_x = self[0]
        for ratio in norm:
            w = int(self[2] * ratio)
            regions.append(Region(accum_x, self[1], w, self[3]))
            accum_x += w
        return regions
    
    def grid(self, cols: int, rows: int) -> List['Region']:
        cell_w = self[2] // cols
        cell_h = self[3] // rows
        regions = []
        for row in range(rows):
            for col in range(cols):
                x = self[0] + cell_w * col
                y = self[1] + cell_h * row
                regions.append(Region(x, y, cell_w, cell_h))
        return regions
    
    def shrink(self, left: int, top: Optional[int] = None, right: Optional[int] = None, bottom: Optional[int] = None) -> 'Region':
        top = top if top is not None else left
        right = right if right is not None else left
        bottom = bottom if bottom is not None else top
        return Region(
            self[0] + left,
            self[1] + top,
            self[2] - left - right,
            self[3] - top - bottom
        )
    
    def center(self, other: RegionLike) -> 'Region':
        other = ensure_region(other)
        cx = other[0] + (other[2] - self[2]) // 2
        cy = other[1] + (other[3] - self[3]) // 2
        return Region(cx, cy, self[2], self[3])
    
    def move(self, dx: int, dy: int) -> 'Region':
        return Region(self[0] + dx, self[1] + dy, self[2], self[3])

