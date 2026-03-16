


SYSTEM_PROMPT_CODING_STYLE = r"""
<coding_guidelines>
Simplicity over ALL else.
Less code is better. Shorter is better. Fewer files is better.
Flat over nested. If you're 3 levels deep, refactor.
Explicit over clever. No tricks, no one-liner heroics.
Don't abstract until you have 3+ duplicates.
Prefer pure functions over methods with side effects.
Less statefulness is better. Short-lived state is best.
Keep state as a single source of truth. Never derive state that can be computed.
Avoid state entirely when possible; use immutable data.
Delete dead code. Don't comment it out.

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
    self.flash_timer = math.max(0, self.flash_timer - dt)
end
function Obj:draw()
    if self.flash_timer > 0 then
        love.graphics.setColor(1,1,1)
    end
end
```

GOOD — derive time elapsed from the moment it happened:
```lua
function Obj:hit()
    self.hit_time = love.timer.getTime()
end
function Obj:draw()
    local elapsed = love.timer.getTime() - (self.hit_time or 0)
    if elapsed < 0.2 then
        love.graphics.setColor(1,1,1)
    end
end
```
Why: no update step, no timer state to manage, no desync bugs. One field instead of one field + update logic.
</example>

<example name="deterministic-hash-over-state">
BAD — storing random visual offsets per-entity:
```lua
function Entity:init()
    self.wobble_offset = math.random() * math.pi * 2
    self.color_variation = math.random() * 0.2 - 0.1
    self.size_jitter = math.random() * 0.1
end
```

GOOD — derive from a deterministic hash of identity:
```lua
function Entity:draw()
    local h = self.x * 287333 + self.y * 173291
    local wobble = (h % 1000) / 1000 * math.pi * 2
    local color_var = (h % 500) / 500 * 0.2 - 0.1
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
    table.insert(self.queue, { target=target, amount=amount, type=type })
end
function DamageSystem:process()
    for _, dmg in ipairs(self.queue) do
        dmg.target.hp = dmg.target.hp - dmg.amount
    end
    self.queue = {}
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

<example name="pass-a-function-not-a-config">
BAD — building an object to describe behavior:
```lua
local Tween = {}
Tween.__index = Tween
function Tween:new(target, field, start, finish, duration, easing)
    return setmetatable({
        target=target, field=field,
        start=start, finish=finish,
        duration=duration, elapsed=0,
        easing=easing or "linear"
    }, self)
end
function Tween:update(dt)
    self.elapsed = math.min(self.elapsed + dt, self.duration)
    local t = self.elapsed / self.duration
    if self.easing == "quadOut" then t = 1-(1-t)^2
    elseif self.easing == "quadIn" then t = t^2 end
    self.target[self.field] = self.start + (self.finish - self.start) * t
end
```

GOOD — just pass a function:
```lua
function tween(duration, fn)
    local start = love.timer.getTime()
    return function()
        local t = math.min((love.timer.getTime() - start) / duration, 1)
        fn(t)
        return t >= 1
    end
end

-- usage: caller decides what happens
local tw = tween(0.3, function(t)
    enemy.scale = 1 + t * 0.5
    enemy.color[4] = 1 - t
end)
```
Why: no config tables, no easing string lookup, no field-name gymnastics. The caller passes the behavior directly. If they want quad easing, they just write `fn(t*t)`. The function IS the config.
</example>

</examples>
</coding_guidelines>
"""

