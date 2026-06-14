-- bot_bridge.lua — apply bot inputs from a command file and report game state.
--
-- Drop in the FBNeo directory and run with:
--   fcadefbneo.exe ... -lua bot_bridge.lua
--
-- Python writes a CPS3 bitfield integer to bot_input.txt. This script reads it
-- once per frame, applies it to P1, then writes the resulting game state to
-- bot_state.txt so Python can compute reward from real memory values instead
-- of pixel-peeping.

local INPUT_PATH = "bot_input.txt"
local STATE_PATH = "bot_state.txt"

-- CPS3 memory addresses for sfiii3 / sfiii3nr1
local P1_HEALTH  = 0x2068D0B
local P2_HEALTH  = 0x20691A3
local P1_WINS    = 0x206A2DB
local P2_WINS    = 0x206A2DD
local P1_COMBO   = 0x20696C5
local P2_COMBO   = 0x206961D
local ROUND_TIMER = 0x2011377
local MAX_HEALTH = 0xA0

local frame = 0
local last_bits = 0

-- ── helpers ──────────────────────────────────────────────────────────

local function read_command_bits()
    local f = io.open(INPUT_PATH, "r")
    if not f then return last_bits end
    local text = f:read("*l")
    f:close()
    local bits = tonumber(text)
    if bits then last_bits = bits end
    return last_bits
end

local function bit(bits, n)
    return math.floor(bits / (2 ^ n)) % 2 == 1
end

local function apply_p1_bits(bits)
    joypad.set({
        ["P1 Up"]          = bit(bits, 0),
        ["P1 Down"]        = bit(bits, 1),
        ["P1 Left"]        = bit(bits, 2),
        ["P1 Right"]       = bit(bits, 3),
        ["P1 Weak Punch"]  = bit(bits, 4),
        ["P1 Medium Punch"] = bit(bits, 5),
        ["P1 Strong Punch"] = bit(bits, 6),
        ["P1 Weak Kick"]   = bit(bits, 7),
        ["P1 Medium Kick"] = bit(bits, 8),
        ["P1 Strong Kick"] = bit(bits, 9),
        ["P1 Start"]       = bit(bits, 10),
    })
end

-- Fixed width: %.4f is 6 chars (byte/0xA0 stays < 2.0) and %3d covers the
-- full 0-255 byte range, so seek(0)+rewrite leaves no trailing bytes from a
-- longer prior line.  Do NOT relax these without truncation support.
local STATE_FMT = "%.4f %.4f %3d %3d %3d %3d %3d\n"

-- One persistent handle, rewritten in place each frame.  Binary mode dodges
-- wine's text-mode fwrite/fflush offset desync (see input_log.lua).  The
-- reader reopens fresh each call and tolerates the torn-read window.
local state_out = io.open(STATE_PATH, "wb")
if not state_out then
    print("ERROR: cannot open " .. STATE_PATH)
end

local function write_state()
    if not state_out then return end
    local p1_hp  = memory.readbyte(P1_HEALTH) / MAX_HEALTH
    local p2_hp  = memory.readbyte(P2_HEALTH) / MAX_HEALTH
    local p1_stun = memory.readbyte(P2_COMBO) ~= 0 and 1 or 0
    local p2_stun = memory.readbyte(P1_COMBO) ~= 0 and 1 or 0
    local timer  = memory.readbyte(ROUND_TIMER)
    local p1_w   = memory.readbyte(P1_WINS)
    local p2_w   = memory.readbyte(P2_WINS)
    state_out:seek("set", 0)
    state_out:write(string.format(STATE_FMT,
        p1_hp, p2_hp, p1_w, p2_w, p1_stun, p2_stun, timer))
    state_out:flush()
end

-- ── main ─────────────────────────────────────────────────────────────

print("bot_bridge.lua: reading " .. INPUT_PATH .. " -> writing " .. STATE_PATH)
while true do
    apply_p1_bits(read_command_bits())
    emu.frameadvance()
    write_state()
end
