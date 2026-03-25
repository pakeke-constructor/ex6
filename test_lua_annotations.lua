-- TEST FILE FOR LUA ANNOTATION HANDLING
-- ======================================
--
-- Agent testing instructions:
--
-- 1. read_headers("test_lua_annotations.lua", line_numbers=True)
--    - Verify line numbers point to the START of annotations, not the function line.
--    - hello should start at line 11 (the ---@param line), not line 13.
--    - greet should start at line 22, not line 25.
--    - bare_func should show its actual line with no annotations.
--    - M.method should start at line 34 (annotation), not line 35.
--    - CONSTANT should show its actual line.
--
-- 2. read_body("test_lua_annotations.lua", "hello", line_numbers=True)
--    - Output should include the ---@param annotations as part of the body.
--    - The ±1 context lines should also be snapshotted.
--    - Line numbers shown should be correct.
--
-- 3. read_body("test_lua_annotations.lua", "hello", line_numbers=True)
--    then edit_file_lines("test_lua_annotations.lua", <first_line_shown>, <last_line_shown>, <new_content>)
--    - This MUST NOT fail with "Line N not in any snapshot".
--    - The boundary context lines (±1) should be in the snapshot.
--
-- 4. read_body("test_lua_annotations.lua", "bare_func", line_numbers=True)
--    - Should work fine with no annotations. No off-by-one errors.
--
-- 5. read_body("test_lua_annotations.lua", "greet", line_numbers=True)
--    - Should include all three ---@ annotations.
--
-- 6. read_headers("test_lua_annotations.lua", line_numbers=False)
--    - Should return signatures with annotations but no line numbers.


---@param w any
---@param h any
function hello(w, h)
    print("hello " .. w .. " " .. h)
end
-- this is a regular comment, not an annotation
-- it should NOT be attached to greet

---@param name string
---@param greeting string
---@return string
function greet(name, greeting)
    return greeting .. ", " .. name .. "!"
end

function bare_func()
    return 42
end

local M = {}

---@param self table
function M.method(self)
    return self
end

---@type number
local CONSTANT = 42

return M
