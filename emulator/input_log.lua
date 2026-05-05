-- input_log.lua — log P1/P2 inputs + game state every frame
-- Drop in FBNeo directory, run with:  fcadefbneo.exe ... -lua input_log.lua
--
-- Output format (one line per frame):
--   <frame> <p1_bits> <p2_bits> <p1_hp> <p2_hp> <p1_stun> <p2_stun> <timer> <p1_wins> <p2_wins>

local out_path = os.getenv("ST_RNN_INPUT_LOG") or "input_log.txt"
local out = io.open(out_path, "w")
if not out then
    print("ERROR: cannot open " .. out_path)
    os.exit(1)
end

-- CPS3 memory addresses for sfiii3 / sfiii3nr1 (verified against fbneo-training-mode)
local P1_HEALTH = 0x2068D0B
local P2_HEALTH = 0x20691A3
local ROUND_TIMER = 0x2011377     -- frames remaining (0-99)
local P1_COMBO   = 0x20696C5     -- P1 combo counter (non-zero → P2 in stun)
local P2_COMBO   = 0x206961D     -- P2 combo counter (non-zero → P1 in stun)

local MAX_HEALTH = 0xA0  -- 160
local KO_THRESHOLD = 0.01
local FULL_HEALTH_THRESHOLD = 0.98

local frame = 0
local p1_wins = 0
local p2_wins = 0
local round_over = false

local function read_bits(prefix)
    -- CPS3/SF3 uses per-game button names, not generic "P1 Button N" (cf. fbneo-training-mode)
    local j = joypad.get()
    local bits = 0
    if j[prefix .. " Up"]          then bits = bits + 1    end  -- bit 0
    if j[prefix .. " Down"]        then bits = bits + 2    end  -- bit 1
    if j[prefix .. " Left"]        then bits = bits + 4    end  -- bit 2
    if j[prefix .. " Right"]       then bits = bits + 8    end  -- bit 3
    if j[prefix .. " Weak Punch"]  then bits = bits + 16   end  -- bit 4  (LP)
    if j[prefix .. " Medium Punch"] then bits = bits + 32  end  -- bit 5  (MP)
    if j[prefix .. " Strong Punch"] then bits = bits + 64  end  -- bit 6  (HP)
    if j[prefix .. " Weak Kick"]   then bits = bits + 128  end  -- bit 7  (LK)
    if j[prefix .. " Medium Kick"] then bits = bits + 256  end  -- bit 8  (MK)
    if j[prefix .. " Strong Kick"] then bits = bits + 512  end  -- bit 9  (HK)
    if j[prefix .. " Start"]       then bits = bits + 1024 end  -- bit 10
    return bits
end

local function is_round_start(p1_hp, p2_hp)
    return p1_hp >= FULL_HEALTH_THRESHOLD and p2_hp >= FULL_HEALTH_THRESHOLD
end

local function round_winner(p1_hp, p2_hp, timer)
    if p1_hp <= KO_THRESHOLD and p2_hp > KO_THRESHOLD then
        return 2
    end
    if p2_hp <= KO_THRESHOLD and p1_hp > KO_THRESHOLD then
        return 1
    end
    if timer == 0 then
        if p1_hp > p2_hp + KO_THRESHOLD then
            return 1
        end
        if p2_hp > p1_hp + KO_THRESHOLD then
            return 2
        end
        return -1
    end
    return 0
end

local function update_round_wins(p1_hp, p2_hp, timer)
    if is_round_start(p1_hp, p2_hp) then
        if p1_wins >= 2 or p2_wins >= 2 then
            p1_wins = 0
            p2_wins = 0
        end
        round_over = false
    end

    if round_over then
        return
    end

    local winner = round_winner(p1_hp, p2_hp, timer)
    if winner == 1 then
        p1_wins = math.min(p1_wins + 1, 2)
        round_over = true
    elseif winner == 2 then
        p2_wins = math.min(p2_wins + 1, 2)
        round_over = true
    elseif winner == -1 then
        round_over = true
    end
end

local function log_frame()
    local p1_bits = read_bits("P1")
    local p2_bits = read_bits("P2")

    -- Game state from memory
    local p1_hp  = memory.readbyte(P1_HEALTH) / MAX_HEALTH
    local p2_hp  = memory.readbyte(P2_HEALTH) / MAX_HEALTH
    -- P1 in stun = P2's combo counter is non-zero (P2 is hitting P1)
    local p1_stun = memory.readbyte(P2_COMBO) ~= 0 and 1 or 0
    -- P2 in stun = P1's combo counter is non-zero (P1 is hitting P2)
    local p2_stun = memory.readbyte(P1_COMBO) ~= 0 and 1 or 0
    local timer   = memory.readbyte(ROUND_TIMER)

    update_round_wins(p1_hp, p2_hp, timer)

    out:write(string.format("%d %d %d %.3f %.3f %d %d %d %d %d\n",
        frame, p1_bits, p2_bits, p1_hp, p2_hp, p1_stun, p2_stun, timer, p1_wins, p2_wins))
    out:flush()
    frame = frame + 1
end

print("input_log.lua: logging (frame p1bits p2bits p1hp p2hp p1stun p2stun timer p1wins p2wins)")
while true do
    log_frame()
    emu.frameadvance()
end
