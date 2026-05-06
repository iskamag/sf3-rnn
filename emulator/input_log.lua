-- input_log.lua — log P1/P2 inputs + game state every frame
-- Drop in FBNeo directory, run with:  fcadefbneo.exe ... -lua input_log.lua
--
-- Output format (one line per frame):
--   <frame> <p1_bits> <p2_bits> <p1_hp> <p2_hp> <p1_stun> <p2_stun> <timer> <p1_wins> <p2_wins> <reset>

local out_path = os.getenv("ST_RNN_INPUT_LOG") or "input_log.txt"
local out = io.open(out_path, "w")
if not out then
    print("ERROR: cannot open " .. out_path)
    os.exit(1)
end

-- CPS3 memory addresses for sfiii3 / sfiii3nr1 (verified against fbneo-training-mode)
local P1_HEALTH = 0x2068D0B
local P2_HEALTH = 0x20691A3
local ROUND_TIMER = 0x2011377
local P1_COMBO   = 0x20696C5
local P2_COMBO   = 0x206961D

local MAX_HEALTH = 0xA0
local KO_THRESHOLD = 0.01
local FULL_HEALTH_THRESHOLD = 0.98

local frame = 0
local p1_wins = 0
local p2_wins = 0
local round_over = false

-- Replay-end detection: after the quark stream ends, the emulator
-- either hangs in frameadvance or keeps returning without advancing
-- the frame counter.  Three consecutive dead frames = replay done.
local last_fc = -1
local dead_fc = 0

local function end_replay()
    out:write("REPLAY_END\n")
    out:flush()
    out:close()
    os.exit(0)
end

local function read_bits(prefix)
    local j = joypad.get()
    local bits = 0
    if j[prefix .. " Up"]          then bits = bits + 1    end
    if j[prefix .. " Down"]        then bits = bits + 2    end
    if j[prefix .. " Left"]        then bits = bits + 4    end
    if j[prefix .. " Right"]       then bits = bits + 8    end
    if j[prefix .. " Weak Punch"]  then bits = bits + 16   end
    if j[prefix .. " Medium Punch"] then bits = bits + 32  end
    if j[prefix .. " Strong Punch"] then bits = bits + 64  end
    if j[prefix .. " Weak Kick"]   then bits = bits + 128  end
    if j[prefix .. " Medium Kick"] then bits = bits + 256  end
    if j[prefix .. " Strong Kick"] then bits = bits + 512  end
    if j[prefix .. " Start"]       then bits = bits + 1024 end
    return bits
end

local function is_round_start(p1_hp, p2_hp)
    return p1_hp >= FULL_HEALTH_THRESHOLD and p2_hp >= FULL_HEALTH_THRESHOLD
end

local function round_winner(p1_hp, p2_hp, timer)
    if p1_hp <= KO_THRESHOLD and p2_hp > KO_THRESHOLD then return 2 end
    if p2_hp <= KO_THRESHOLD and p1_hp > KO_THRESHOLD then return 1 end
    if timer == 0 then
        if p1_hp > p2_hp + KO_THRESHOLD then return 1 end
        if p2_hp > p1_hp + KO_THRESHOLD then return 2 end
        return -1
    end
    return 0
end

local function update_round_wins(p1_hp, p2_hp, timer)
    if is_round_start(p1_hp, p2_hp) then
        if p1_wins >= 2 or p2_wins >= 2 then p1_wins = 0; p2_wins = 0 end
        round_over = false
    end
    if round_over then return end
    local winner = round_winner(p1_hp, p2_hp, timer)
    if winner == 1 then p1_wins = math.min(p1_wins + 1, 2); round_over = true
    elseif winner == 2 then p2_wins = math.min(p2_wins + 1, 2); round_over = true
    elseif winner == -1 then round_over = true end
end

local function detect_console_reset(p1_bits, p2_bits)
    local j = joypad.get()
    local system_key = (j["Service"] or j["Test"] or j["Reset"]
                        or j["Diagnostic"] or j["Service Mode"]
                        or j["Service 1"] or j["Service 2"])
    local p1_service = (p1_bits >= 2048)
    local p2_service = (p2_bits >= 2048)
    return system_key or p1_service or p2_service
end

local function log_frame()
    local p1_bits = read_bits("P1")
    local p2_bits = read_bits("P2")
    local p1_hp  = memory.readbyte(P1_HEALTH) / MAX_HEALTH
    local p2_hp  = memory.readbyte(P2_HEALTH) / MAX_HEALTH
    local p1_stun = memory.readbyte(P2_COMBO) ~= 0 and 1 or 0
    local p2_stun = memory.readbyte(P1_COMBO) ~= 0 and 1 or 0
    local timer   = memory.readbyte(ROUND_TIMER)
    local reset = detect_console_reset(p1_bits, p2_bits) and 1 or 0
    update_round_wins(p1_hp, p2_hp, timer)
    out:write(string.format("%d %d %d %.3f %.3f %d %d %d %d %d %d\n",
        frame, p1_bits, p2_bits, p1_hp, p2_hp, p1_stun, p2_stun, timer, p1_wins, p2_wins, reset))
    out:flush()
    frame = frame + 1
end

print("input_log.lua: logging")
while true do
    log_frame()
    emu.frameadvance()

    local fc = emu.framecount()
    if fc == last_fc then
        dead_fc = dead_fc + 1
        if dead_fc >= 3 then end_replay() end
    else
        dead_fc = 0
        last_fc = fc
    end
end
