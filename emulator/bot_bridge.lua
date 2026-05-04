-- bot_bridge.lua — apply bot inputs from a command file and log game state.
--
-- Drop in the FBNeo directory and run with:
--   fcadefbneo.exe ... -lua bot_bridge.lua
--
-- Python writes a CPS3 bitfield integer to bot_input.txt. This script reads it
-- once per frame, applies it to P1, then logs inputs/state like input_log.lua.

local INPUT_PATH = "bot_input.txt"
local LOG_PATH = "input_log.txt"

local out = io.open(LOG_PATH, "w")
if not out then
    print("ERROR: cannot open " .. LOG_PATH)
    os.exit(1)
end

local P1_HEALTH = 0x2068D0B
local P2_HEALTH = 0x20691A3
local ROUND_TIMER = 0x2011377
local P1_WINS    = 0x206A2DB
local P2_WINS    = 0x206A2DD
local P1_COMBO   = 0x20696C5
local P2_COMBO   = 0x206961D
local MAX_HEALTH = 0xA0

local frame = 0
local last_bits = 0

local function read_command_bits()
    local f = io.open(INPUT_PATH, "r")
    if not f then
        return last_bits
    end
    local text = f:read("*l")
    f:close()
    local bits = tonumber(text)
    if bits then
        last_bits = bits
    end
    return last_bits
end

local function bit(bits, n)
    return math.floor(bits / (2 ^ n)) % 2 == 1
end

local function apply_p1_bits(bits)
    joypad.set({
        ["P1 Up"] = bit(bits, 0),
        ["P1 Down"] = bit(bits, 1),
        ["P1 Left"] = bit(bits, 2),
        ["P1 Right"] = bit(bits, 3),
        ["P1 Weak Punch"] = bit(bits, 4),
        ["P1 Medium Punch"] = bit(bits, 5),
        ["P1 Strong Punch"] = bit(bits, 6),
        ["P1 Weak Kick"] = bit(bits, 7),
        ["P1 Medium Kick"] = bit(bits, 8),
        ["P1 Strong Kick"] = bit(bits, 9),
        ["P1 Start"] = bit(bits, 10),
    })
end

local function read_bits(prefix)
    local j = joypad.get()
    local bits = 0
    if j[prefix .. " Up"] then bits = bits + 1 end
    if j[prefix .. " Down"] then bits = bits + 2 end
    if j[prefix .. " Left"] then bits = bits + 4 end
    if j[prefix .. " Right"] then bits = bits + 8 end
    if j[prefix .. " Weak Punch"] then bits = bits + 16 end
    if j[prefix .. " Medium Punch"] then bits = bits + 32 end
    if j[prefix .. " Strong Punch"] then bits = bits + 64 end
    if j[prefix .. " Weak Kick"] then bits = bits + 128 end
    if j[prefix .. " Medium Kick"] then bits = bits + 256 end
    if j[prefix .. " Strong Kick"] then bits = bits + 512 end
    if j[prefix .. " Start"] then bits = bits + 1024 end
    return bits
end

local function log_frame()
    local p1_bits = read_bits("P1")
    local p2_bits = read_bits("P2")
    local p1_hp = memory.readbyte(P1_HEALTH) / MAX_HEALTH
    local p2_hp = memory.readbyte(P2_HEALTH) / MAX_HEALTH
    local p1_stun = memory.readbyte(P2_COMBO) ~= 0 and 1 or 0
    local p2_stun = memory.readbyte(P1_COMBO) ~= 0 and 1 or 0
    local timer = memory.readbyte(ROUND_TIMER)
    local p1_w = memory.readbyte(P1_WINS)
    local p2_w = memory.readbyte(P2_WINS)

    out:write(string.format("%d %d %d %.3f %.3f %d %d %d %d %d\n",
        frame, p1_bits, p2_bits, p1_hp, p2_hp, p1_stun, p2_stun, timer, p1_w, p2_w))
    out:flush()
    frame = frame + 1
end

print("bot_bridge.lua: reading " .. INPUT_PATH .. " and logging " .. LOG_PATH)
while true do
    apply_p1_bits(read_command_bits())
    log_frame()
    emu.frameadvance()
end
