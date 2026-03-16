
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
Task: "Add damage to entities"

BAD — model jumps straight to implementation, builds a system:
```lua
local DamageSystem = {}
DamageSystem.__index = DamageSystem
function DamageSystem:new()
    return setmetatable({ queue = {} }, self)
end
function DamageSystem:enqueue(target, amount, type)
    -- add to queue
end
function DamageSystem:process()
    -- iterate queue, apply damage, clear
end
```
The model never asked "what does this actually need to do?" It pattern-matched on "damage" and built an enterprise damage pipeline.

GOOD — model thinks through what's actually needed FIRST:
```
Okay, what does "add damage" actually require?
- Something takes damage: target.hp = target.hp - amount
- That's it. Is there batching? No. Is there damage types? Not asked for.
- Is there anything else that needs to react to damage? Not mentioned.
- Simplest thing: a function call.
```
Then implements:
```lua
function damage(target, amount)
    target.hp = target.hp - amount
end
```

The difference is not intelligence — it's that the second model STOPPED and asked "what does this actually need?" before writing code. Always do this. Ask yourself what the MINIMAL requirements are. If the answer is 3 lines, write 3 lines. The urge to build infrastructure is almost always wrong.
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
</example>

</examples>
</coding_guidelines>
"""



