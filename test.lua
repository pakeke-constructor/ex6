

---@class g.BlessingInfo
---@field id string
---@field image string
---@field handlers table<string, function>


---@class g.PerkInfo
---@field id string
---@field image string
---@field handlers table<string, function>


---@class g
local g = {}

local AutoAtlas = require("lib.AutoAtlas.AutoAtlas")

local atlas = AutoAtlas(2048, 2048)

local nameToQuad = {}

local richtext = require("src.modules.richtext.exports")

local sceneManager = require("src.scenes.sceneManager")

local Run = require("src.Run")
local Entity = require("src.ecs.Entity")

local currentRun

function g.newRun()
    currentRun = Run()
    return currentRun
end


---@return ecs.ECSWorld?
function g.getBattleECS()
    -- gets the ECS for battle
    local scene, name = g.getCurrentScene()
    if name == "battle_scene" then
        return scene.ecs
    end
end


function g.hasRun()
    return currentRun ~= nil
end

function g.getRun()
    return assert(currentRun, "run not loaded")
end

function g.delRun()
    currentRun = nil
end

function g.saveRun()
    if not currentRun or not currentRun.serialize then
        return
    end
    local data = currentRun:serialize()
    local contents = json.encode(data)
    love.filesystem.write("saves/run1.json", contents)
end

function g.loadRun(path)
    local contents = assert(love.filesystem.read(path))
    local data = json.decode(contents)
    currentRun = Run.deserialize(data)
end

function g.saveAndInvalidateRun()
    if not currentRun or not currentRun.serialize then
        return
    end
    g.saveRun()
    g.delRun()
end


---@return love.Texture
function g.getAtlas()
    return atlas:getTexture()
end

---@param imageName string
function g.getImageQuad(imageName)
    local quad = nameToQuad[imageName]
    if not quad then
        error("Invalid quad: " .. tostring(imageName))
    end
    return quad
end

---@param imageName any
---@return boolean
function g.isImage(imageName)
    return (nameToQuad[imageName] and true) or false
end

---@param imageName string|love.Quad
---@param x number
---@param y number
---@param r number?
---@param sx number?
---@param sy number?
---@param kx number?
---@param ky number?
function g.drawImage(imageName, x, y, r, sx, sy, kx, ky)
    return g.drawImageOffset(imageName, x, y, r, sx, sy, 0.5, 0.5, kx, ky)
end

---@param imageName string|love.Quad
---@param x number
---@param y number
---@param r number?
---@param sx number?
---@param sy number?
---@param ox number?
---@param oy number?
---@param kx number?
---@param ky number?
function g.drawImageOffset(imageName, x, y, r, sx, sy, ox, oy, kx, ky)
    local quad
    if type(imageName) == "string" then
        quad = g.getImageQuad(imageName)
    else
        if not (imageName.typeOf and imageName:typeOf("Quad")) then
            error("Expected quad, got: " .. type(imageName) .. " " .. tostring(imageName))
        end
        quad = imageName
    end
    local _,_,w,h = quad:getViewport()
    atlas:draw(quad, x, y, r, sx, sy, (ox or 0.5) * w, (oy or 0.5) * h, kx, ky)
end

---@param imageName string
---@param x number
---@param y number
---@param w number
---@param h number
---@param rot number?
function g.drawImageContained(imageName, x, y, w, h, rot)
    local quad = g.getImageQuad(imageName)
    local _,_,qw,qh = quad:getViewport()
    local scaleX = w / qw
    local scaleY = h / qh
    local scale = math.min(scaleX, scaleY)
    local scaledW = qw * scale
    local scaledH = qh * scale
    local centerX = x + (w - scaledW) / 2
    local centerY = y + (h - scaledH) / 2
    atlas:draw(quad, centerX + scaledW/2, centerY + scaledH/2, rot or 0, scale, scale, qw/2, qh/2)
end

---@param path string
---@param func fun(path: string)
function g.walkDirectory(path, func)
    local info = love.filesystem.getInfo(path)
    if not info then return end

    if info.type == "file" then
        func(path)
    elseif info.type == "directory" then
        local dirItems = love.filesystem.getDirectoryItems(path)
        for _, pth in ipairs(dirItems) do
            g.walkDirectory(path .. "/" .. pth, func)
        end
    end
end

local validExtensions = {
    [".png"] = true,
    [".jpg"] = true,
}

local function loadImage(path)
    local ext = path:sub(-4):lower()
    if validExtensions[ext] then
        local name = path:match("([^/]+)%.%w+$")
        local quad = atlas:add(love.image.newImageData(path))
        if nameToQuad[name] then
            error("Duplicate image: " .. name)
        end
        nameToQuad[name] = quad
        if richtext and richtext.defineImage then
            pcall(richtext.defineImage, name, atlas:getTexture(), quad)
        end
    end
end

function g.loadImagesFrom(path)
    g.walkDirectory(path, loadImage)
end


-- Define 1x1 white image
do
    -- Add padding around to prevent bleeding
    local id = love.image.newImageData(3, 3, "rgba8")
    id:mapPixel(function() return 1, 1, 1, 0 end) -- fill transparent white
    id:setPixel(1, 1, 1, 1, 1, 1) -- set middle pixel
    local q = assert(atlas:add(id))
    local x, y = q:getViewport()
    -- Now define it to be 1x1 instead of 3x3
    q:setViewport(x + 1, y + 1, 1, 1, g.getAtlas():getDimensions())
    nameToQuad["1x1"] = q
end


-- Forward-declare perk tables (used by both squad and perk systems below)
local PERK_DEFS = {}
local PERK_LIST = {}

-- Squad system
local SQUAD_DEFS = {}
local SQUAD_LIST = {}

---@class g.SquadInfo
---@field id string
---@field entityId string
---@field count number
---@field image string
---@field perks string[]
---@field onDeploy (fun(squad: g.SquadInfo, entities: table[]))?

---@param id string
---@param info g.SquadInfo
function g.defineSquad(id, info)
    assert(not SQUAD_DEFS[id], "Duplicate squad: " .. id)
    info.id = id
    info.perks = info.perks or {}
    info.count = info.count or 1
    SQUAD_DEFS[id] = info
    SQUAD_LIST[#SQUAD_LIST + 1] = id
end

function g.getSquadInfo(id)
    return assert(SQUAD_DEFS[id], "Unknown squad: " .. tostring(id))
end

function g.getSquadList()
    return SQUAD_LIST
end

--- Add a perk to a squad definition (applied to entities on spawn)
function g.addPerkToSquad(squadId, perkId)
    local info = assert(SQUAD_DEFS[squadId], "Unknown squad: " .. tostring(squadId))
    assert(PERK_DEFS[perkId], "Unknown perk: " .. tostring(perkId))
    info.perks[#info.perks + 1] = perkId
end

--- Remove a perk from a squad definition
function g.removePerkFromSquad(squadId, perkId)
    local info = assert(SQUAD_DEFS[squadId], "Unknown squad: " .. tostring(squadId))
    for i = #info.perks, 1, -1 do
        if info.perks[i] == perkId then
            table.remove(info.perks, i)
            return true
        end
    end
    return false
end

--- Spawn all entities for a squad, applying perks and calling onDeploy
function g.spawnSquad(squadId, x, y, ...)
    local info = assert(SQUAD_DEFS[squadId], "Unknown squad: " .. tostring(squadId))
    local squadScope = g.newScope()
    for j = 1, #info.perks do
        local perkInfo = g.getPerkInfo(info.perks[j])
        squadScope:addHandler(perkInfo.handlers)
    end
    local entities = {}
    for i = 1, info.count do
        local ent = g.spawnEntity(info.entityId, x, y, ...)
        ent.squadScope = squadScope
        entities[i] = ent
    end
    if info.onDeploy then
        info.onDeploy(info, entities)
    end
    return entities
end

-- Blessing system
local BLESSING_DEFS = {}
local BLESSING_LIST = {}

---@param id string
---@param info g.BlessingInfo
function g.defineBlessing(id, info)
    assert(not BLESSING_DEFS[id], "Duplicate blessing: " .. id)
    info.id = id
    BLESSING_DEFS[id] = info
    BLESSING_LIST[#BLESSING_LIST + 1] = id
end

---@return g.BlessingInfo
function g.getBlessingInfo(id)
    return assert(BLESSING_DEFS[id], "Unknown blessing: " .. tostring(id))
end

function g.getBlessingList()
    return BLESSING_LIST
end

function g.addBlessing(id)
    assert(BLESSING_DEFS[id], "Unknown blessing: " .. tostring(id))
    local run = g.getRun()
    run.blessings[#run.blessings + 1] = id
end

function g.removeBlessing(id)
    local run = g.getRun()
    for i = #run.blessings, 1, -1 do
        if run.blessings[i] == id then
            table.remove(run.blessings, i)
            return true
        end
    end
    return false
end

-- Perk system

---@param id string
---@param info g.PerkInfo
function g.definePerk(id, info)
    assert(not PERK_DEFS[id], "Duplicate perk: " .. id)
    info.id = id
    PERK_DEFS[id] = info
    PERK_LIST[#PERK_LIST + 1] = id
end

function g.getPerkInfo(id)
    return assert(PERK_DEFS[id], "Unknown perk: " .. tostring(id))
end

function g.addPerk(ent, id)
    local info = assert(PERK_DEFS[id], "Unknown perk: " .. tostring(id))
    assert(info.handlers, "Perk has no handlers: " .. id)
    if not ent.scope then ent.scope = g.newScope() end
    ent.scope:addHandler(info.handlers)
end

function g.removePerk(ent, id)
    if not ent.scope then return false end
    local info = assert(PERK_DEFS[id], "Unknown perk: " .. tostring(id))
    return ent.scope:removeHandler(info.handlers)
end

-- Entity system
local ENTITY_DEFS = {}
local ENTITY_LIST = {}
local currentEntityId = 0
function g.defineEntity(id, def)
    assert(not ENTITY_DEFS[id], "Duplicate entity type: " .. id)
    assert(def.x == nil and def.y == nil and def.type == nil and def._world == nil, "x/y/type/_world are reserved")
    for k in pairs(Entity) do
        assert(def[k] == nil, "Entity def '" .. id .. "' cannot override base method: " .. k)
    end
    def.type = id
    def.image = def.image or id
    for k, v in pairs(Entity) do
        def[k] = v
    end
    local mt = {__index = def}
    ENTITY_DEFS[id] = mt
    ENTITY_LIST[#ENTITY_LIST + 1] = id
end

function g.spawnEntity(id, x, y, ...)
    local mt = ENTITY_DEFS[id]
    assert(mt, "Unknown entity type: " .. tostring(id))
    local ecs = g.getBattleECS()
    assert(ecs, "g.spawnEntity called outside of battle")
    currentEntityId = currentEntityId + 1
    local ent = setmetatable({
        id = currentEntityId,
        x = x, y = y, type = id,
        _world = ecs,
    }, mt)
    if ent.init then
        ent:init(...)
    end
    ecs:addEntity(ent)
    return ent
end

function g.drawEntity(ent, x, y)
    local sx, sy = ent.sx or 1, ent.sy or 1
    if ent.draw then
        ent:draw(x, y)
        return
    end
    if ent.image then
        love.graphics.setColor(1, 1, 1, ent.alpha or 1)
        g.drawImage(ent.image, x + (ent.ox or 0), y + (ent.oy or 0), ent.rot or 0, sx, sy)
    end
end

function g.getEntityDef(id)
    local mt = ENTITY_DEFS[id]
    return mt and mt.__index
end

local suffixes = {
    {1e12, "t"},
    {1e9,  "b"},
    {1e6,  "m"},
    {1e3,  "k"},
}

local bigCache = {}
local smolCache = {}
local fbCache = {}

local function getFallbackFonts(size)
    if not fbCache[size] then
        fbCache[size] = love.graphics.newFont("assets/fonts/unifont-17.0.03.otf", size, "mono", size / 16)
    end
    return fbCache[size]
end

---@param size number
function g.getBigFont(size)
    assert(size % 16 == 0, "Size must by divisible by 16")
    if not bigCache[size] then
        local f = love.graphics.newFont("assets/fonts/Smart 9h.ttf", size, "mono", 1)
        f:setFallbacks(getFallbackFonts(size))
        bigCache[size] = f
    end
    return bigCache[size]
end

---@param size number
function g.getSmallFont(size)
    assert(size % 16 == 0, "Size must by divisible by 16")
    if not smolCache[size] then
        local f = love.graphics.newFont("assets/fonts/Match 7h.ttf", size, "mono", 1)
        f:setFallbacks(getFallbackFonts(size))
        smolCache[size] = f
    end
    return smolCache[size]
end

---@param path string
function g.requireFolder(path)
    local results = {}
    g.walkDirectory(path:gsub("%.", "/"), function(pth)
        if pth:sub(-4, -1) == ".lua" then
            pth = pth:sub(1, -5)
            results[pth] = require(pth:gsub("%/", "."))
        end
    end)
    return results
end

---@param num number
function g.formatNumber(num)
    local isNegative = num < 0
    num = math.abs(num)
    local prefix = (isNegative and "-" or "")

    if num < 1000 then
        if num == math.floor(num) then
            return prefix .. ("%d"):format(num)
        elseif num < 1 then
            return prefix .. ("%.2f"):format(num)
        elseif num < 3 then
            return prefix .. ("%.1f"):format(num)
        end
        return prefix .. tostring(math.floor(num))
    end

    for _, suffix in ipairs(suffixes) do
        if num >= suffix[1] then
            local scaled = num / suffix[1]
            local formatted
            if scaled >= 100 then
                formatted = string.format("%.0f", math.floor(scaled))
            elseif scaled >= 10 then
                formatted = string.format("%.14g", math.floor(scaled * 10) / 10)
            else
                formatted = string.format("%.14g", math.floor(scaled * 100) / 100)
            end
            return prefix .. formatted .. suffix[2]
        end
    end
    return prefix .. tostring(num)
end

function g.gotoScene(sceneName)
    return sceneManager.gotoScene(sceneName)
end

function g.gotoLastScene()
    return sceneManager.gotoLastScene()
end

function g.getCurrentScene()
    return sceneManager.getCurrentScene()
end


function g.getWorldTime()
    -- todo: add a proper counter here; allows for faster game-speed
    return love.timer.getTime()
end


-- Event Bus / Question Bus
local reducers = require("src.modules.reducers")

local definedEvents = {}
local questions = {}
-- Scene-level handler caches: name -> {func1, func2, ...}
-- Built incrementally by g.addHandler, wiped by g.clearHandlers.
local table_clear = require("table.clear")
local handlerCache = {} -- [eventOrQuestionName] -> {func, func, ...}

function g.defineEvent(ev)
    assert(not definedEvents[ev], "Event already defined: " .. ev)
    definedEvents[ev] = true
    handlerCache[ev] = {}
end

function g.isEvent(ev)
    return definedEvents[ev] == true
end

function g.defineQuestion(question, reducer, defaultValue)
    assert(not questions[question], "Question already defined: " .. question)
    questions[question] = {
        reducer = reducer,
        defaultValue = defaultValue,
    }
    handlerCache[question] = {}
end

function g.getQuestionInfo(q)
    return questions[q]
end

-- Add a handler table for this frame only.
-- Must be re-added every frame (e.g. in scene:preUpdate).
function g.addHandler(handler)
    for key, func in pairs(handler) do
        local list = handlerCache[key]
        assert(list, "Unknown event/question: " .. tostring(key))
        list[#list + 1] = func
    end
end

-- Called once per frame to clear all ephemeral handlers.
function g.clearHandlers()
    for _, list in pairs(handlerCache) do
        table_clear(list)
    end
end


---@class g.Scope: objects.Class
local Scope = objects.Class("g:Scope")

function Scope:init()
    self.handlers = {}
    self.expiry = {} -- [handler] -> expire time
    self.cache = {} -- [eventOrQuestionName] -> {func, func, ...}
    self.lastPrune = 0
end

function Scope:_rebuild()
    local cache = self.cache
    for k in pairs(cache) do
        table_clear(cache[k])
    end
    local now = love.timer.getTime()
    local expiry = self.expiry
    for _, handler in ipairs(self.handlers) do
        if not expiry[handler] or expiry[handler] > now then
            for key, func in pairs(handler) do
                if definedEvents[key] or questions[key] then
                    if not cache[key] then cache[key] = {} end
                    local list = cache[key]
                    list[#list + 1] = func
                end
            end
        end
    end
end

function Scope:_pruneIfNeeded()
    local now = love.timer.getTime()
    if now - self.lastPrune < 0.2 then return end
    self.lastPrune = now
    local dirty = false
    local expiry = self.expiry
    for i = #self.handlers, 1, -1 do
        local h = self.handlers[i]
        if expiry[h] and expiry[h] <= now then
            table.remove(self.handlers, i)
            expiry[h] = nil
            dirty = true
        end
    end
    if dirty then self:_rebuild() end
end

function Scope:addHandler(handler, duration)
    for key in pairs(handler) do
        assert(definedEvents[key] or questions[key], "Unknown event/question: " .. tostring(key))
    end
    if duration then
        self.expiry[handler] = love.timer.getTime() + duration
    end
    self.handlers[#self.handlers + 1] = handler
    self:_rebuild()
end

function Scope:removeHandler(handler)
    for i = #self.handlers, 1, -1 do
        if self.handlers[i] == handler then
            table.remove(self.handlers, i)
            self.expiry[handler] = nil
            self:_rebuild()
            return true
        end
    end
    return false
end

function Scope:call(event, ...)
    self:_pruneIfNeeded()
    local list = self.cache[event]
    if not list then return end
    for i = 1, #list do
        list[i](...)
    end
end

function Scope:ask(question, ...)
    self:_pruneIfNeeded()
    local t = questions[question]
    if not t then
        error("Invalid question: " .. tostring(question))
    end
    local reducer, val = t.reducer, t.defaultValue
    local list = self.cache[question]
    if not list then return val end
    for i = 1, #list do
        val = reducer(val, list[i](...))
    end
    return val
end


---@return g.Scope
function g.newScope()
    return Scope()
end


-- Fire an event. No return value.
-- Order: scene-level handlers, then ent[ev], then ent.scope
function g.call(ev, arg1, ...)
    -- 1. scene-level handlers
    local list = handlerCache[ev]
    for i = 1, #list do
        list[i](arg1, ...)
    end

    if type(arg1) ~= "table" then return end

    -- 2. direct entity handler
    if arg1[ev] then
        arg1[ev](arg1, ...)
    end

    -- 3. entity scope (perks etc)
    local scope = arg1.scope
    if scope then
        scope:call(ev, arg1, ...)
    end

    -- 4. squad scope (shared squad perks)
    local squadScope = arg1.squadScope
    if squadScope then
        squadScope:call(ev, arg1, ...)
    end
end

-- Ask a question. Returns reduced value.
-- Order: scene-level handlers, then ent[q], then ent.scope
function g.ask(q, arg1, ...)
    local t = questions[q]
    if not t then
        error("Invalid question: " .. tostring(q))
    end
    local reducer, val = t.reducer, t.defaultValue

    -- 1. scene-level handlers
    local list = handlerCache[q]
    for i = 1, #list do
        val = reducer(val, list[i](arg1, ...))
    end

    if type(arg1) == "table" then
        -- 2. direct entity handler
        if arg1[q] then
            val = reducer(val, arg1[q](arg1, ...))
        end

        -- 3. entity scope (perks etc)
        if arg1.scope then
            val = reducer(val, arg1.scope:ask(q, arg1, ...))
        end

        -- 4. squad scope (shared squad perks)
        if arg1.squadScope then
            val = reducer(val, arg1.squadScope:ask(q, arg1, ...))
        end
    end

    return val
end

return g
