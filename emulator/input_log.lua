-- input_log.lua — log P1 inputs + game state every frame
-- Drop in FBNeo directory, run with:  fcadefbneo.exe ... -lua input_log.lua
--
-- Output format (one line per frame):
--   <frame> <bits> <p1_hp> <p2_hp> <p1_stun> <p2_stun> <timer> <p1_wins> <p2_wins>

local out = io.open("input_log.txt", "w")
if not out then print("ERROR: cannot open input_log.txt") return end

-- CPS3 memory addresses (sfiii3 / sfiii3nr1)
local P1_HEALTH = 0x2068D0B
local P2_HEALTH = 0x20691A3
local ROUND_TIMER = 0x20683E4     -- frames remaining (0-99)
local P1_WINS    = 0x206A2DB     -- rounds won
local P2_WINS    = 0x206A2DD     -- rounds won
-- stun: not easily available as a single byte; approximate via guard meter or skip

local MAX_HEALTH = 0xA0  -- 160

local frame = 0

local function log_frame()
    -- Inputs
    local j = joypad.get()
    local bits = 0
    if j["P1 Up"]        then bits = bits | (1 << 0) end
    if j["P1 Down"]      then bits = bits | (1 << 1) end
    if j["P1 Left"]      then bits = bits | (1 << 2) end
    if j["P1 Right"]     then bits = bits | (1 << 3) end
    if j["P1 Button 1"]  then bits = bits | (1 << 4) end
    if j["P1 Button 2"]  then bits = bits | (1 << 5) end
    if j["P1 Button 3"]  then bits = bits | (1 << 6) end
    if j["P1 Button 4"]  then bits = bits | (1 << 7) end
    if j["P1 Button 5"]  then bits = bits | (1 << 8) end
    if j["P1 Button 6"]  then bits = bits | (1 << 9) end
    if j["P1 Start"]     then bits = bits | (1 << 10) end

    -- Game state from memory
    local p1_hp  = memory.readbyte(P1_HEALTH) / MAX_HEALTH
    local p2_hp  = memory.readbyte(P2_HEALTH) / MAX_HEALTH
    local timer  = memory.readbyte(ROUND_TIMER)
    local p1_w   = memory.readbyte(P1_WINS)
    local p2_w   = memory.readbyte(P2_WINS)

    out:write(string.format("%d %d %.3f %.3f %d %d %d\n",
        frame, bits, p1_hp, p2_hp, timer, p1_w, p2_w))
    out:flush()
    frame = frame + 1
end

print("input_log.lua: logging (frame inputs p1hp p2hp timer p1wins p2wins)")
while true do
    log_frame()
    emu.frameadvance()
end
