
SYSTEM_PROMPT_CODING_STYLE = r"""
<coding_guidelines>
- Simplicity over ALL else.
- Less code is better. Shorter is better. Fewer files is better.
- Flat over nested. If you're 3 levels deep, refactor.
- Explicit over clever. No tricks, no one-liner heroics.
- Don't abstract until you have 3+ duplicates.
- Prefer pure functions over methods with side effects.
- Less statefulness is better. Short-lived state is best.
- Keep state as a single source of truth. Never derive state that can be computed.
- Avoid state entirely when possible; use immutable data.
- Delete dead code. Don't comment it out.

IMPORTANT: Before implementing anything non-trivial, write pseudocode comments first. Plan the shape, THEN fill it in.

<examples>

<example name="avoid-stored-state-use-time">
BAD — storing a timer variable, updating it, checking it:
```lua
function Obj:init()
    self.flash_timer = 0
end
function Obj:hit()
    self.flash_timer = 0.2
end
function Obj:update(dt)
    -- decrement timer
end
function Obj:draw()
    -- check timer > 0, set color
end
```

GOOD — derive it from the moment it happened:
```lua
function Obj:hit()
    self.hit_time = love.timer.getTime()
end
function Obj:draw()
    -- check (now - hit_time) < 0.2, set color
end
```
Why: no update step, no timer state to manage, no desync bugs. One field instead of one field + update logic.
</example>

<example name="deterministic-hash-over-state">
BAD — storing random visual offsets per-entity:
```lua
function Entity:init()
    self.wobble_offset = math.random() * math.pi * 2
end
```

GOOD — derive from a deterministic hash of identity:
```lua
function Entity:draw()
    local h = self.x * 287333 + self.y * 173291
    local wobble = (h % 1000) / 1000 * math.pi * 2
end
```
Why: zero stored state, fully deterministic, survives serialization, no init required. Works for any entity that has a stable identity (position, id, index, etc).
</example>

<example name="pseudocode-before-implementing">
BAD — diving straight into implementation:
```lua
function buildNavMesh(world)
    local cells = {}
    for x = 0, world.w - 1 do
        for y = 0, world.h - 1 do
            -- 80 lines of code that evolved without a plan
        end
    end
end
```

GOOD — pseudocode skeleton first, then fill in:
```lua
function buildNavMesh(world)
    -- 1. grid the world into walkable cells
    -- 2. flood-fill to find connected regions
    -- 3. merge adjacent cells into convex polygons
    -- 4. build adjacency graph between polygons
    -- 5. return { polygons, adjacency }
end
```
Then implement each step. The shape is locked in before any real code is written.
</example>

<example name="think-before-building">
Task: "Smoothly animate a value from A to B, and support changing the target mid-animation"

BAD — model builds an animation system:
```lua
function anim_new(start, target, duration)
    -- return {start, target, duration, elapsed}
end
function anim_retarget(anim, new_target)
    -- snapshot current value as new start, reset elapsed
end
function anim_update(anim, dt)
    -- increment elapsed, clamp, interpolate start→target
end
```

GOOD — model stops to think first:
```
What does "smoothly animate toward a target" actually mean?
- A value should move toward a goal over time.
- If the goal changes, it should redirect smoothly.
- WAIT — I'm framing this as an event (start→end).
  But it's actually a continuous process: "chase the target."
  Exponential approach does that in one line.
```
Then implements:
```lua
e.x = e.x + (e.target_x - e.x) * 10 * dt
```
Change `target_x` anytime. The value always chases smoothly. No start value, no elapsed time, no retarget logic. The reframe — from "animation event" to "chase target" — eliminated the entire system.
</example>

<example name="functions-as-data">
BAD — building a config object to describe behavior:
```lua
handler = EventHandler:new({
    event = "on_click",
    target = button,
    action = "toggle_visibility",
    args = {panel},
})
handler:register()
```

GOOD — just pass a function:
```lua
on("click", button, function()
    panel.visible = not panel.visible
end)
```
Why: no config tables, no string lookups, no args gymnastics. The caller passes the behavior directly, which is a lot more flexible. The function IS the config.

Other examples where this could be useful include fields on objects / entity definitions.
```lua
defineEntity(name {
    getDamageMultiplier = func -- func returns 0 when entity is in "defensive mode".
    -- Much simpler than writing a complex system to handle different "modes", and damage propagation.
})
```
</example>

</examples>
</coding_guidelines>
"""



