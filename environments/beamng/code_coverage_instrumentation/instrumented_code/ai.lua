-- This Source Code Form is subject to the terms of the bCDDL, v. 1.1.
-- If a copy of the bCDDL was not distributed with this
-- file, You can obtain one at http://beamng.com/bCDDL-1.1.txt


-- [[ Coverage functions ]] --
local control_variables = {}
control_variables['line_count']         = 756
control_variables['branches_count']     = 471
control_variables['current_max_length'] = 10000000
control_variables['current_path_index'] = 1
control_variables['start_coverage']     = false

local coverage = {}
coverage['line'] = {}
coverage['branch'] = {}
coverage['path'] = {}

for i=0, control_variables['line_count'] do
  coverage['line'][i] = 0
end

for i=0, control_variables['branches_count'] do
  coverage['branch'][i] = 0
end

for i=0, control_variables['current_max_length'] do
  coverage['path'][i] = 0
end

local function getLineCoverageArray()
  local cov_out = ''
  for i, line in ipairs(coverage['line']) do
    cov_out = cov_out ..' ' .. tostring(line)
  end
  return cov_out
end

local function getBranchCoverageArray()
  local branch_out = ''
  for i, branch in ipairs(coverage['branch']) do
    branch_out = branch_out ..' ' .. tostring(branch)
  end
  return branch_out
end

local function getPathCoverageArray()
  local path_out = table.concat(coverage['path'], ', ')
  return path_out
end

local function resetCoverageArrays()
for i=0, control_variables['line_count'] do
coverage['line'][i] = 0
end
for i= 0, control_variables['branches_count'] do
coverage['branch'][i] = 0
end
for i=0, control_variables['current_max_length'] do
coverage['path'][i] = 0
end
control_variables['current_path_index'] = 1
-- Turn off coverage until its started again
control_variables['start_coverage'] = false
end

local function startCoverage()
  control_variables['start_coverage'] = true
end

local function update_path_coverage(line_branch_number)
  if control_variables['start_coverage'] then
    if control_variables['current_path_index'] < control_variables['current_max_length'] then
      coverage['path'][control_variables['current_path_index']] = line_branch_number
      control_variables['current_path_index'] = control_variables['current_path_index'] + 1
    end
  end
end

local function enter_or_exit_function(e_string, func_name)
  if control_variables['start_coverage'] then
    if e_string == 'enter' then
      coverage['path'][control_variables['current_path_index']] = 'enter_' .. func_name
      control_variables['current_path_index'] = control_variables['current_path_index'] + 1
    end
    if e_string == 'exit' then
      coverage['path'][control_variables['current_path_index']] = 'exit_' .. func_name
      control_variables['current_path_index'] = control_variables['current_path_index'] + 1
    end
  end
end

local function process_line(line_number)
  if control_variables['start_coverage'] then
    coverage['line'][line_number] = 1;
  end
end

local function predicate_analyzer(predicate, t_branch, f_branch)
  if control_variables['start_coverage'] then
    if (predicate) then
      coverage['branch'][t_branch] = 1
      update_path_coverage(t_branch)
    else
      coverage['branch'][f_branch] = 1
      update_path_coverage(f_branch)
    end
  end
  return predicate
end

local function for_loop_pair(i, data, t_branch, f_branch)
  if control_variables['start_coverage'] then
    local count = 0
    for _ in pairs(data) do count = count + 1 end
    if (i < count) then
      coverage['branch'][t_branch] = 1
      update_path_coverage(t_branch)
    else
      coverage['branch'][f_branch] = 1
      update_path_coverage(f_branch)
    end
  end
end

local function for_loop_range(data1, data2, t_branch, f_branch)
  if control_variables['start_coverage'] then
    if (data1 < data2) then
      coverage['branch'][t_branch] = 1
      update_path_coverage(t_branch)
    else
      coverage['branch'][f_branch] = 1
      update_path_coverage(f_branch)
    end
  end
end
---------------------------------


-- [[ STORE FREQUENTLY USED FUNCTIONS IN UPVALUES ]] --
-- local max = math.max
-- local min = math.min
-- local sin = math.sin
-- local asin = math.asin
-- local pi = math.pi
-- local abs = math.abs
-- local sqrt = math.sqrt
-- local floor = math.floor
-- local tableInsert = table.insert
-- local tableRemove = table.remove
-- local strFormat = string.format
---------------------------------
local scriptai = nil

local M = {}

M.mode = 'disabled' -- this is the main mode
M.manualTargetName = nil
M.debugMode = 'off'
M.speedMode = nil
M.routeSpeed = nil
M.extAggression = 0.3
M.cutOffDrivability = 0

-- [[ Simulation time step]] --
local dt

-- [[ ENVIRONMENT VARIABLES ]] --
local g = math.abs(obj:getGravity())
local gravityDir = vec3(0, 0, -1)
local gravityVec = gravityDir * g
----------------------------------

-- [[ PERFORMANCE RELATED ]] --
local aggression = 1
local aggressionMode
--------------------------------------------------------------------------

-- [[ AI DATA: POSITION, CONTROL, STATE ]] --
local aiPos = vec3(obj:getFrontPosition())
--aiPos.z = BeamEngine:getSurfaceHeightBelow(aiPos)
local aiDirVec = vec3(obj:getDirectionVector())
local aiVel = vec3(obj:getVelocity())
local aiSpeed = aiVel:length()
local ai = {prevDirVec = vec3(aiDirVec), upVec = vec3(obj:getDirectionVectorUp()), rightVec = vec3(), width = nil, length = nil}
local targetSpeedSmoother = nil -- = newTemporalSmoothingNonLinear(math.huge, 0.2)
local aiDeviation = 0
local aiDeviationSmoother = newTemporalSmoothing(1)
local smoothTcs = newTemporalSmoothingNonLinear(0.1, 0.9)
local aiCannotMoveTime = 0
local aiForceGoFrontTime = 0
local staticFrictionCoef = 1
local threewayturn = {state = 0, speedDifInt = 0}

local forces = {}

local lastCommand = {steering = 0, throttle = 0, brake = 0, parkingbrake = 0}

local driveInLaneFlag = false
local internalState = 'onroad'

local validateInput = nop
------------------------------

-- [[ CRASH DETECTION ]] --
local crash = {time = 0, manoeuvre = 0, dir = nil}

-- [[ OPPONENT DATA ]] --
local player
local plPrevVel
local chaseData = {suspectState = nil, suspectStoppedTimer = 0, distanceToSuspect = nil, playerSeg = nil, playerXnormOnSeg = nil, playerRoad = nil}

-- [[ SETTINGS, PARAMETERS, AUXILIARY DATA ]] --
local mapData -- map data including node connections (edges and edge data), node positions and node radii
local signalsData -- traffic intersection and signals data
local currentRoute
local minPlanCount = 3
local targetWPName

local wpList, manualPath, speedList
local race, noOfLaps

local targetObjectSelectionMode

local edgeDict

------------------------------

-- [[ TRAFFIC ]] --
local currPathIdx = 0
local damageFlag = false
local hornFlag = false
local trafficTable = {}
local trafficBlock = {timer = 0, coef = 0, limit = 6, horn = 0}
local trafficSide = {timer = 0, cTimer = 0, side = 1, timerRange = 6}
local trafficSignal = {hTimer = 0, hLimit = 1}
local intersection = {stopTime = 0, timer = 0, turn = 0}
local avoidCars = 'on'
M.avoidCarsMaster = 'auto'
local changePlanTimer = 0

-----------------------

-- [[ HEAVY DEBUG MODE ]] --
local speedRecordings = {}
local trajecRec = {last = 0}
local routeRec = {last = 0}
local labelRenderDistance = 10
------------------------------

local function aistatus(status, category)
  guihooks.trigger("AIStatusChange", {status=status, category=category})
end

local function getState()
  enter_or_exit_function('enter', 'getState')
  enter_or_exit_function('exit', 'getState')
  process_line(0); return M
end

local function stateChanged()
  enter_or_exit_function('enter', 'stateChanged')
  if (predicate_analyzer(playerInfo.anyPlayerSeated, 0, 1)) then
    process_line(1); guihooks.trigger("AIStateChange", getState())
  end
  enter_or_exit_function('exit', 'stateChanged')
end

local function setSpeed(speed)
  enter_or_exit_function('enter', 'setSpeed')
  if (predicate_analyzer(type(speed) ~= 'number', 2, 3)) then
    process_line(2); M.routeSpeed = nil
  else 
    process_line(3); M.routeSpeed = speed
  end
  enter_or_exit_function('exit', 'setSpeed')
end

local function setSpeedMode(speedMode)
  enter_or_exit_function('enter', 'setSpeedMode')
  if (predicate_analyzer(speedMode == 'set' or speedMode == 'limit' or speedMode == 'legal' or speedMode == 'off', 4, 5)) then
    process_line(4); M.speedMode = speedMode
  else
    process_line(5); M.speedMode = nil
  end
  enter_or_exit_function('exit', 'setSpeedMode')
end

local function resetSpeedModeAndValue()
  M.speedMode = nil -- maybe this should be 'off'
  M.routeSpeed = nil
end

local function setAggressionInternal(v)
  enter_or_exit_function('enter', 'setAggressionInternal')
  process_line(6); aggression = v and v or M.extAggression
  enter_or_exit_function('exit', 'setAggressionInternal')
end

local function setAggressionExternal(v)
  enter_or_exit_function('enter', 'setAggressionExternal')
  process_line(7); M.extAggression = v or M.extAggression
  process_line(8); setAggressionInternal()
  process_line(9); stateChanged()
  enter_or_exit_function('exit', 'setAggressionExternal')
end

local function setAggressionMode(aggrmode)
  if aggrmode == 'rubberBand' or aggrmode == 'manual' then
    aggressionMode = aggrmode
  else
    aggressionMode = nil
  end
end

local function resetAggression()
  enter_or_exit_function('enter', 'resetAggression')
  process_line(10); setAggressionInternal()
  enter_or_exit_function('exit', 'resetAggression')
end

local function setTargetObjectID(id)
  enter_or_exit_function('enter', 'setTargetObjectID')
  process_line(11); M.targetObjectID = M.targetObjectID ~= objectId and id or -1
  if (predicate_analyzer(M.targetObjectID ~= -1, 6, 7)) then
    process_line(12); targetObjectSelectionMode = 'manual'
  end
  enter_or_exit_function('exit', 'setTargetObjectID')
end

local function updatePlayerData()
  if mapmgr.objects[M.targetObjectID] and targetObjectSelectionMode == 'manual' then
    player = mapmgr.objects[M.targetObjectID]
    player.id = M.targetObjectID
  elseif tableSize(mapmgr.objects) == 2 then
    if player ~= nil then
      player = mapmgr.objects[player.id]
    else
      for k, v in pairs(mapmgr.objects) do
        if k ~= objectId then
          M.targetObjectID = k
          player = v
          break
        end
      end
      targetObjectSelectionMode = 'auto'
    end
  else
    if player ~= nil and player.active == true then
      player = mapmgr.objects[player.id]
    else
      for k, v in pairs(mapmgr.objects) do
        if k ~= objectId and v.active == true then
          M.targetObjectID = k
          player = v
          break
        end
      end
      targetObjectSelectionMode = 'targetActive'
    end
  end
  mapmgr.objects[objectId] = mapmgr.objects[objectId] or {pos = aiPos, dirVec = aiDirVec}
end

local function driveCar(steering, throttle, brake, parkingbrake)
  enter_or_exit_function('enter', 'driveCar')
  process_line(13); input.event("steering", steering, 1)
  process_line(14); input.event("throttle", throttle, 2)
  process_line(15); input.event("brake", brake, 2)
  process_line(16); input.event("parkingbrake", parkingbrake, 2)

  process_line(17); lastCommand.steering = steering
  process_line(18); lastCommand.throttle = throttle
  process_line(19); lastCommand.brake = brake
  process_line(20); lastCommand.parkingbrake = parkingbrake
  enter_or_exit_function('exit', 'driveCar')
end

local function driveToTarget(targetPos, throttle, brake, targetSpeed)
  enter_or_exit_function('enter', 'driveToTarget')
  if (predicate_analyzer(not targetPos, 8, 9)) then
    enter_or_exit_function('exit', 'driveToTarget')
    process_line(21); return
  end

  process_line(22); local brakeCoef = 1
  process_line(23); local throttleCoef = 1

  process_line(24); local targetVec = (targetPos - aiPos):normalized()

  process_line(25); local dirAngle = math.asin(ai.rightVec:dot(targetVec))
  process_line(26); local dirVel = aiVel:dot(aiDirVec)
  process_line(27); local absAiSpeed = math.abs(dirVel)
  process_line(28); local plan = currentRoute and currentRoute.plan
  process_line(29); targetSpeed = targetSpeed or plan and plan.targetSpeed
  if (predicate_analyzer(not targetSpeed, 10, 11)) then
    enter_or_exit_function('exit', 'driveToTarget')
    process_line(30); return
  end

  -- oversteer
  if (predicate_analyzer(aiSpeed > 1, 12, 13)) then
    process_line(31); local rightVel = ai.rightVec:dot(aiVel)
    if (predicate_analyzer(rightVel * ai.rightVec:dot(targetPos - aiPos) > 0, 14, 15)) then
      process_line(32); local rotVel = math.min(1, (ai.prevDirVec:projectToOriginPlane(ai.upVec):normalized()):distance(aiDirVec) * dt * 10000)
      process_line(33); throttleCoef = throttleCoef * math.max(0, 1 - math.abs(rightVel * aiSpeed * 0.05) * math.min(1, dirAngle * dirAngle * aiSpeed * 6) * rotVel)
    end
  end

  if (predicate_analyzer(plan and plan[3] and dirVel > 3, 16, 17)) then
    process_line(34); local p1, p2 = plan[1].pos, plan[2].pos
    process_line(35); local p2p1 = p2 - p1
    process_line(36); local turnRight = p2p1:cross(ai.upVec):normalized()
    process_line(37); local tp2
    if (predicate_analyzer(plan.targetSeg and plan.targetSeg > 1, 18, 19)) then
      process_line(38); tp2 = (targetPos - p2):normalized():dot(turnRight)
    else
      process_line(39); tp2 = (plan[3].pos - p2):normalized():dot(turnRight)
    end

    process_line(40); local outDeviation = aiDeviationSmoother:value() - aiDeviation * sign(tp2)
    process_line(41); outDeviation = sign(outDeviation) * math.min(1, math.abs(outDeviation))
    process_line(42); aiDeviationSmoother:set(outDeviation)
    process_line(43); aiDeviationSmoother:getUncapped(0, dt)

    if (predicate_analyzer(outDeviation > 0 and absAiSpeed > 3, 20, 21)) then
      process_line(44); local steerCoef = outDeviation * absAiSpeed * absAiSpeed * math.min(1, dirAngle * dirAngle * 4)
      process_line(45); local understeerCoef = math.max(0, steerCoef) * math.min(1, math.abs(aiVel:dot(p2p1:normalized()) * 3))
      process_line(46); local noUndersteerCoef = math.max(0, 1 - understeerCoef)
      process_line(47); throttleCoef = throttleCoef * noUndersteerCoef
      process_line(48); brakeCoef = math.min(brakeCoef, math.max(0, 1 - understeerCoef * understeerCoef))
    end
  else
    process_line(49); aiDeviationSmoother:set(0)
  end

  -- wheel speed
  if (predicate_analyzer(absAiSpeed > 0.05, 22, 23)) then
    if (predicate_analyzer(sensors.gz <= 0, 24, 25)) then
      process_line(50); local totalSlip = 0
      process_line(51); local totalDownForce = 0
      process_line(52); local propSlip = 0
      process_line(53); local propDownForce = 0
      process_line(54); local lwheels = wheels.wheels
      for_loop_range(0, tableSizeC(lwheels) - 1, 26, 27)
      for i = 0, tableSizeC(lwheels) - 1 do
        for_loop_range(i, tableSizeC(lwheels) - 1, 26, 27)
        process_line(55); local wd = lwheels[i]
        if (predicate_analyzer(not wd.isBroken, 28, 29)) then
          process_line(56); local lastSlip = wd.lastSlip
          process_line(57); totalSlip = totalSlip + lastSlip
          if (predicate_analyzer(wd.isPropulsed, 30, 31)) then
            process_line(58); propSlip = math.max(propSlip, lastSlip)
          end
        end
      end

      -- math.abs
      process_line(59); brakeCoef = brakeCoef * square(math.max(0, absAiSpeed - totalSlip) / absAiSpeed)

      -- tcs
      process_line(60); local tcsCoef = math.max(0, absAiSpeed - propSlip * propSlip) / absAiSpeed
      process_line(61); throttleCoef = throttleCoef * math.min(tcsCoef, smoothTcs:get(tcsCoef, dt))
    else
      process_line(62); brakeCoef = 0
      process_line(63); throttleCoef = 0
    end
  end

  process_line(64); local dirTarget = aiDirVec:dot(targetVec)
  process_line(65); local dirDiff = math.asin(aiDirVec:cross(ai.upVec):normalized():dot(targetVec))

  if (predicate_analyzer(crash.manoeuvre == 1 and dirTarget < aiDirVec:dot(crash.dir), 32, 33)) then
    process_line(66); driveCar(-fsign(dirDiff), brake * brakeCoef, throttle * throttleCoef, 0)
    enter_or_exit_function('exit', 'driveToTarget')
    process_line(67); return
  else
    process_line(68); crash.manoeuvre = 0
  end

  process_line(69); aiForceGoFrontTime = math.max(0, aiForceGoFrontTime - dt)
  if (predicate_analyzer(threewayturn.state == 1 and aiCannotMoveTime > 1 and aiForceGoFrontTime == 0, 34, 35)) then
    process_line(70); threewayturn.state = 0
    process_line(71); aiCannotMoveTime = 0
    process_line(72); aiForceGoFrontTime = 2
  end

  if (predicate_analyzer(aiForceGoFrontTime > 0 and dirTarget < 0, 36, 37)) then
    process_line(73); dirTarget = -dirTarget
    process_line(74); dirDiff = -dirDiff
  end

  if (predicate_analyzer((dirTarget < 0 or (dirTarget < 0.15 and threewayturn.state == 1)) and currentRoute and not damageFlag, 38, 39)) then
    process_line(75); local n1, n2, n3 = plan[1], plan[2], plan[3]
    process_line(76); local edgeDist = math.min((n2 or n1).radiusOrig, n1.radiusOrig) - aiPos:z0():distanceToLine((n3 or n2).posOrig:z0(), n2.posOrig:z0())
    if (predicate_analyzer(edgeDist > ai.width and threewayturn.state == 0, 40, 41)) then
      process_line(77); driveCar(fsign(dirDiff), 0.5 * throttleCoef, 0, math.min(math.max(aiSpeed - 3, 0), 1))
    else
      if (predicate_analyzer(threewayturn.state == 0, 42, 43)) then
        process_line(78); threewayturn.state = 1
        process_line(79); threewayturn.speedDifInt = 0
      end
      process_line(80); local angleModulation = math.min(math.max(0, -(dirTarget-0.15)), 1)
      process_line(81); local speedDif = (10 * aggression * angleModulation) - aiSpeed
      process_line(82); threewayturn.speedDifInt = threewayturn.speedDifInt + speedDif * dt
      process_line(83); local pbrake = clamp(sign2(aiDirVec:dot(gravityDir) - 0.17), 0, 1) -- apply parking brake if reversing on an incline >~ 10 deg
      process_line(84); driveCar(-sign2(dirDiff), 0, clamp(0.05 * speedDif + 0.01 * threewayturn.speedDifInt, 0, 1), pbrake)
    end
  else
    process_line(85); threewayturn.state = 0
    process_line(86); local pbrake
    if (predicate_analyzer(aiVel:dot(aiDirVec) * math.max(aiSpeed*aiSpeed - 1e-2, 0) < 0, 44, 45)) then
      if (predicate_analyzer(aiSpeed < 0.15 and targetSpeed <= 1e-5, 46, 47)) then
        process_line(87); pbrake = 1
      else
        process_line(88); pbrake = 0
      end
      process_line(89); driveCar(dirDiff, 0.5 * throttleCoef, 0, pbrake)
    else
      if (predicate_analyzer((aiSpeed > 4 and aiSpeed < 30 and math.abs(dirDiff) > 0.97 and brake == 0) or (aiSpeed < 0.15 and targetSpeed <= 1e-5), 48, 49)) then
        process_line(90); pbrake = 1
      else
        process_line(91); pbrake = 0
      end
      process_line(92); driveCar(dirDiff, throttle * throttleCoef, brake * brakeCoef, pbrake)
    end
  end
  enter_or_exit_function('exit', 'driveToTarget')
end

local function posOnPlan(pos, plan, dist)
  if not plan then
    return
  end
  dist = dist or 4
  dist = dist * dist
  local bestSeg, bestXnorm, bestDist
  for i = 1, #plan-2 do
    local p0, p1 = plan[i].pos, plan[i+1].pos
    local xnorm1 = pos:xnormOnLine(p0, p1)
    if xnorm1 > 0 then
      local p2 = plan[i+2].pos
      local xnorm2 = pos:xnormOnLine(p1, p2)
      if xnorm1 < 1 then -- contained in segment i
        if xnorm2 > 0 then -- also partly contained in segment i+1
          local sqDistFromP1 = pos:squaredDistance(p1)
          if sqDistFromP1 <= dist then
            bestSeg = i
            bestXnorm = 1
            break -- break inside conditional
          end
        else
          local sqDistFromLine = pos:squaredDistance(p0 + (p1 - p0) * xnorm1)
          if sqDistFromLine <= dist then
            bestSeg = i
            bestXnorm = xnorm1
          end
          break -- break should be outside above conditional
        end
      elseif xnorm2 < 0 then
        local sqDistFromP1 = pos:squaredDistance(p1)
        if sqDistFromP1 <= dist then
          bestSeg = i
          bestXnorm = 1
        end
        break -- break outside conditional
      end
    else
      break
    end
  end

  return bestSeg, bestXnorm
end

local function aiPosOnPlan(plan, planCount)
  enter_or_exit_function('enter', 'aiPosOnPlan')
  process_line(93); local aiSeg = 1
  process_line(94); local aiXnormOnSeg = 0
  for_loop_range(1, planCount-1, 50, 51)
  for i = 1, planCount-1 do
    for_loop_range(i, planCount-1, 50, 51)
    process_line(95); local p0Pos, p1Pos = plan[i].pos, plan[i+1].pos
    process_line(96); local xnorm = aiPos:xnormOnLine(p0Pos, p1Pos)
    if (predicate_analyzer(xnorm < 1, 52, 53)) then
      if (predicate_analyzer(i < planCount - 2, 54, 55)) then
        process_line(97); local nextXnorm = aiPos:xnormOnLine(p1Pos, plan[i+2].pos)
        if (predicate_analyzer(nextXnorm >= 0, 56, 57)) then
          process_line(98); local p1Radius = plan[i+1].radius
          if (predicate_analyzer(aiPos:squaredDistance(linePointFromXnorm(p1Pos, plan[i+2].pos, nextXnorm)) < square(ai.width + lerp(p1Radius, plan[i+2].radius, math.min(1, nextXnorm))), 58, 59)) then
            process_line(99); aiXnormOnSeg = nextXnorm
            process_line(100); aiSeg = i + 1
            process_line(101); break
          end
        end
      end
      process_line(102); aiXnormOnSeg = xnorm
      process_line(103); aiSeg = i
      process_line(104); break
    end
  end

  for_loop_range(1, aiSeg-1, 60, 61)
  for _ = 1, aiSeg-1 do
    for_loop_range(_, aiSeg-1, 60, 61)
    process_line(105); table.remove(plan, 1)
    process_line(106); planCount = planCount-1
  end

  enter_or_exit_function('exit', 'aiPosOnPlan')
  process_line(107); return aiXnormOnSeg, planCount
end

local function calculateTarget(plan, planCount)
  enter_or_exit_function('enter', 'calculateTarget')
  process_line(108); planCount = planCount or #plan
  process_line(109); local aiXnormOnSeg
  process_line(110); aiXnormOnSeg, planCount = aiPosOnPlan(plan, planCount)
  process_line(111); local targetLength = math.max(aiSpeed * 0.65, 4.5)

  if (predicate_analyzer(planCount >= 3, 62, 63)) then
    process_line(112); local xnorm = clamp(aiXnormOnSeg, 0, 1)
    process_line(113); local p2Pos = plan[2].pos
    process_line(114); targetLength = math.max(targetLength, plan[1].pos:distance(p2Pos) * (1-xnorm), p2Pos:distance(plan[3].pos) * xnorm)
  end

  process_line(115); local remainder = targetLength

  process_line(116); local targetPos = vec3(plan[planCount].pos)
  process_line(117); local targetSeg = math.max(1, planCount-1)
  process_line(118); local prevPos = linePointFromXnorm(plan[1].pos, plan[2].pos, aiXnormOnSeg) -- aiPos

  process_line(119); local segVec = vec3()
  for_loop_range(2, planCount, 64, 65)
  for i = 2, planCount do
    for_loop_range(i, planCount, 64, 65)
    process_line(120); local pos = plan[i].pos
    process_line(121); segVec:set(pos)
    process_line(122); segVec:setSub(prevPos)
    process_line(123); local segLen = segVec:length()

    if (predicate_analyzer(remainder <= segLen, 66, 67)) then
      process_line(124); targetSeg = i - 1
      process_line(125); targetPos:set(segVec)
      process_line(126); targetPos:setScaled(remainder / (segLen + 1e-25))
      process_line(127); targetPos:setAdd(prevPos)

      -- smooth target
      process_line(128); local xnorm = clamp(targetPos:xnormOnLine(prevPos, pos), 0, 1)
      process_line(129); local lp_n1n2 = linePointFromXnorm(prevPos, pos, xnorm * 0.5 + 0.25)
      if (predicate_analyzer(xnorm <= 0.5, 68, 69)) then
        if (predicate_analyzer(i >= 3, 70, 71)) then
          process_line(130); targetPos = linePointFromXnorm(linePointFromXnorm(plan[i-2].pos, prevPos, xnorm * 0.5 + 0.75), lp_n1n2, xnorm + 0.5)
        end
      else
        if (predicate_analyzer(i <= planCount - 2, 72, 73)) then
          process_line(131); targetPos = linePointFromXnorm(lp_n1n2, linePointFromXnorm(pos, plan[i+1].pos, xnorm * 0.5 - 0.25), xnorm - 0.5)
        end
      end
      process_line(132); break
    end
    process_line(133); prevPos = pos
    process_line(134); remainder = remainder - segLen
  end

  enter_or_exit_function('exit', 'calculateTarget')
  process_line(135); return targetPos, targetSeg, aiXnormOnSeg, planCount
end

local function targetsCompatible(baseRoute, newRoute)
  local baseTvec = baseRoute.plan.targetPos - aiPos
  local newTvec = newRoute.plan.targetPos - aiPos
  if aiSpeed < 2 then
    return true
  end
  if newTvec:dot(aiDirVec) * baseTvec:dot(aiDirVec) <= 0 then
    return false
  end
  local baseTargetRight = baseTvec:cross(ai.upVec):normalized()
  return math.abs(newTvec:normalized():dot(baseTargetRight)) * aiSpeed < 2
end

local function getMinPlanLen(limLow, speed, accelg)
  enter_or_exit_function('enter', 'getMinPlanLen')
  -- given current speed, distance required to come to a stop if I can decelerate at 0.2g
  process_line(136); limLow = limLow or 150
  process_line(137); speed = speed or aiSpeed
  process_line(138); accelg = math.max(0.2, accelg or 0.2)
  enter_or_exit_function('exit', 'getMinPlanLen')
  process_line(139); return math.min(550, math.max(limLow, 0.5 * speed * speed / (accelg * g)))
end

local function pickAiWp(wp1, wp2, dirVec)
  dirVec = dirVec or aiDirVec
  local vec1 = mapData.positions[wp1] - aiPos
  local vec2 = mapData.positions[wp2] - aiPos
  local dot1 = vec1:dot(dirVec)
  local dot2 = vec2:dot(dirVec)
  if (dot1 * dot2) <= 0 then
    if dot1 < 0 then
      return wp2
    end
  else
    if vec2:squaredLength() < vec1:squaredLength() then
      return wp2
    end
  end
  return wp1
end

local function pathExtend(path, newPath)
  enter_or_exit_function('enter', 'pathExtend')
  if (predicate_analyzer(newPath == nil, 74, 75)) then
    enter_or_exit_function('exit', 'pathExtend')
    process_line(140); return
  end
  process_line(141); local pathCount = #path
  if (predicate_analyzer(path[pathCount] ~= newPath[1], 76, 77)) then
    enter_or_exit_function('exit', 'pathExtend')
    process_line(142); return
  end
  process_line(143); pathCount = pathCount - 1
  for_loop_range(2, #newPath, 78, 79)
  for i = 2, #newPath do
    for_loop_range(i, #newPath, 78, 79)
    process_line(144); path[pathCount+i] = newPath[i]
  end
  enter_or_exit_function('exit', 'pathExtend')
end

-- http://cnx.org/contents/--TzKjCB@8/Projectile-motion-on-an-inclin
local function projectileSqSpeedToRangeRatio(pos1, pos2, pos3)
  local sinTheta = (pos2.z - pos1.z) / pos1:distance(pos2)
  local sinAlpha = (pos3.z - pos2.z) / pos2:distance(pos3)
  local cosAlphaSquared = math.max(1 - sinAlpha * sinAlpha, 0)
  local cosTheta = math.sqrt(math.max(1 - sinTheta * sinTheta, 0)) -- in the interval theta = {-math.pi/2, math.pi/2} cosTheta is always positive
  return 0.5 * g * cosAlphaSquared / math.max(cosTheta * (sinTheta*math.sqrt(cosAlphaSquared) - cosTheta*sinAlpha), 0)
end

local function inCurvature(vec1, vec2)
  enter_or_exit_function('enter', 'inCurvature')
  process_line(145); local vec1Sqlen, vec2Sqlen = vec1:squaredLength(), vec2:squaredLength()
  process_line(146); local dot12 = vec1:dot(vec2)
  process_line(147); local cos8sq = math.min(1, dot12 * dot12 / math.max(1e-30, vec1Sqlen * vec2Sqlen))

  if (predicate_analyzer(dot12 < 0, 80, 81)) then
    process_line(148); local minDsq = math.min(vec1Sqlen, vec2Sqlen)
    process_line(149); local maxDsq = minDsq / math.max(1e-30, cos8sq)
    if (predicate_analyzer(math.max(vec1Sqlen, vec2Sqlen) > (minDsq + maxDsq) * 0.5, 82, 83)) then
      if (predicate_analyzer(vec1Sqlen > vec2Sqlen, 84, 85)) then
        process_line(150); vec1, vec2 = vec2, vec1
        process_line(151); vec1Sqlen, vec2Sqlen = vec2Sqlen, vec1Sqlen
      end
      process_line(152); vec2 = math.sqrt(0.5 * (minDsq + maxDsq) / math.max(1e-30, vec2Sqlen)) * vec2
    end
  end

  enter_or_exit_function('exit', 'inCurvature')
  process_line(153); return 2 * math.sqrt((1 - cos8sq) / math.max(1e-30, (vec1 + vec2):squaredLength()))
end

local function getPathLen(path, startIdx, stopIdx)
  enter_or_exit_function('enter', 'getPathLen')
  if (predicate_analyzer(not path, 86, 87)) then
    enter_or_exit_function('exit', 'getPathLen')
    process_line(154); return
  end
  process_line(155); startIdx = startIdx or 1
  process_line(156); stopIdx = stopIdx or #path
  process_line(157); local positions = mapData.positions
  process_line(158); local pathLen = 0
  for_loop_range(startIdx+1, stopIdx, 88, 89)
  for i = startIdx+1, stopIdx do
    for_loop_range(i, stopIdx, 88, 89)
    process_line(159); pathLen = pathLen + positions[path[i-1]]:distance(positions[path[i]])
  end

  enter_or_exit_function('exit', 'getPathLen')
  process_line(160); return pathLen
end

local function waypointInPath(path, waypoint, startIdx, stopIdx)
  enter_or_exit_function('enter', 'waypointInPath')
  if (predicate_analyzer(not path or not waypoint, 90, 91)) then
    enter_or_exit_function('exit', 'waypointInPath')
    process_line(161); return
  end
  process_line(162); startIdx = startIdx or 1
  process_line(163); stopIdx = stopIdx or #path
  for_loop_range(startIdx, stopIdx, 92, 93)
  for i = startIdx, stopIdx do
    for_loop_range(i, stopIdx, 92, 93)
    if (predicate_analyzer(path[i] == waypoint, 94, 95)) then
      enter_or_exit_function('exit', 'waypointInPath')
      process_line(164); return i
    end
  end
  enter_or_exit_function('exit', 'waypointInPath')
end

local function doTrafficActions(path, plan)
  -- hazard lights
  if mapmgr.objects[objectId] and mapmgr.objects[objectId].damage >= 1000 then
    if not damageFlag then
      trafficSignal.hTimer = trafficSignal.hTimer + dt
      if trafficSignal.hTimer > trafficSignal.hLimit then
        electrics.set_warn_signal(1)
        trafficSignal.hTimer = 0
        damageFlag = true
      end
    end
    return
  else
    damageFlag = false
  end

  -- horn
  if not hornFlag then
    if not damageFlag and trafficBlock.horn > 0 and trafficBlock.timer >= trafficBlock.horn then
      electrics.horn(true)
      hornFlag = true
    end
  else
    if trafficBlock.timer < trafficBlock.horn then
      electrics.horn(false)
      hornFlag = false
    end
  end

  if damageFlag then
    return
  end

  -- intersections & turn signals
  if not intersection.node then
    local dist = mapData.positions[path[plan[1].pathidx]]:distance(aiPos) -- limit the distance to look ahead for intersections
    local minLen = getMinPlanLen(80)

    for i = plan[1].pathidx, #path - 1 do
      local nid1, nid2 = path[i], path[i + 1]
      local n1Pos, n2Pos = mapData.positions[nid1], mapData.positions[nid2]
      local prevNode = i > 1 and path[i - 1] -- use previous path node if it exists
      local nDir = prevNode and (n1Pos - mapData.positions[prevNode]):normalized() or vec3(aiDirVec)

      if dist > minLen then
        break
      end

      if signalsData then -- defined intersections
        local sNodes = signalsData.nodes
        if sNodes[nid1] then -- node from current path was found in the signals dict
          local rootNode = sNodes[nid1].alias -- node is part of a multi-node intersection
          local mainNode = rootNode and sNodes[rootNode] or sNodes[nid1]

          for j, node in ipairs(mainNode.signalNodes) do
            local validNode = (i == 1 and path[i] == node.edgeNodes[1]) or arrayFindValueIndex(path, node.edgeNodes[2]) -- two steps just to be safe
            if validNode and nDir:dot(mapData.positions[node.edgeNodes[1]] - mapData.positions[node.edgeNodes[2]]) > 0 then -- if intersection signal direction matches
              intersection.node = nid1
              intersection.pos = vec3(mainNode.pos)
              intersection.radius = math.max(mapData.radius[nid1], mainNode.radius)
              intersection.dir = nDir
              intersection.linkDir = (n2Pos - n1Pos):z0():normalized()
              intersection.ref = node
              intersection.stopPos = vec3(node.pos)
              break
            end
          end
        end
      end

      if not intersection.node then -- auto intersections
        if tableSize(mapData.graph[nid1]) > 2 then
          for k, v in pairs(mapData.graph[nid1]) do
            if k == nid2 then -- child node matches next path node
              local drivability = prevNode and mapData.graph[nid1][prevNode][2] or 1
              local linkDir = (n2Pos - n1Pos):z0():normalized()

              if math.abs(nDir:dot(linkDir)) < 0.8 or v[2] - drivability > 0 then -- junction turn or drivability difference
                intersection.node = nid1
                intersection.pos = vec3(n1Pos)
                intersection.radius = mapData.radius[nid1]
                intersection.dir = nDir
                intersection.linkDir = linkDir
                intersection.action = 0.1 -- default stop & go
                intersection.stopPos = n1Pos - nDir * math.max(4, intersection.radius) * 1.5
                break
              end
            end
          end
        end
      end

      if intersection.node then
        intersection.stopTime = 2.5
        if math.abs(intersection.dir:dot(intersection.linkDir)) < 0.8 then
          intersection.turn = -sign2(intersection.dir:cross(gravityDir):dot(intersection.linkDir)) -- turn detection
        end
        break
      end

      dist = dist + n2Pos:distance(n1Pos)
    end
  else
    if intersection.stopPos then
      if intersection.ref then
        intersection.action = intersection.ref.action or 1
      end -- get action from referenced table

      if false then -- debug testing
        local sColor = (intersection.action and intersection.action <= 0.1) and color(255,0,0,160) or color(0,255,0,160)
        obj.debugDrawProxy:drawSphere(1, (intersection.pos + aiDirVec):toFloat3(), sColor)
      end

      local bestDist = math.huge
      local bestIdx = 1
      local distSq = aiPos:squaredDistance(intersection.stopPos)
      intersection.planStop = math.huge

      for i = 1, #plan - 1 do -- get best plan node to set as a stopping point
        if (plan[i].pos - intersection.stopPos):dot(intersection.dir) <= 0 then
          local dist = plan[i].pos:squaredDistance(intersection.stopPos)
          if dist < bestDist then
            bestDist = dist
            bestIdx = i
          end
        end
      end

      local brakeDist = square(aiSpeed) / (2 * g * aggression)

      if ((intersection.stopPos + aiDirVec * ai.length) - aiPos):dot(intersection.dir) >= 0 then -- vehicle position is behind the stop pos
        if intersection.action == 0.5 then -- yellow light
          if not intersection.stopFlag then
            -- TODO: check if vehicle is turning at intersection
            if square(brakeDist) <= distSq then
              intersection.planStop = bestIdx -- plan index for AI stop
              intersection.stopFlag = true -- force the vehicle to plan to stop
            else
              intersection.stopPos = nil -- go through intersection
            end
          else
            intersection.planStop = bestIdx
          end
        end

        if intersection.action <= 0.1 then -- red light or other stop signal
          intersection.planStop = bestIdx
        end

        if intersection.action == 0.1 then
          if distSq <= math.max(bestDist, square(ai.length)) and aiSpeed <= 1 then
            intersection.timer = intersection.timer + dt
          end
          if intersection.timer >= intersection.stopTime then
            intersection.ref = nil
            intersection.action = 1
          end
        end
      else
        intersection.stopPos = nil -- clear stop pos, but do not search for next intersection yet
      end

      if distSq < square(brakeDist * 1.5) then -- approaching intersection
        if intersection.turn < 0 and electrics.values.turnsignal >= 0 then
          electrics.toggle_left_signal()
        elseif intersection.turn > 0 and electrics.values.turnsignal <= 0 then
          electrics.toggle_right_signal()
        end
      end
    else
      if aiDirVec:dot(intersection.pos - aiPos) < 0 and aiPos:squaredDistance(intersection.pos) > square(intersection.radius + ai.length) then -- vehicle is clear of intersection
        intersection = {stopTime = 0, timer = 0, turn = 0}
      end
    end
  end
end

local function getPlanLen(plan)
  enter_or_exit_function('enter', 'getPlanLen')
  process_line(165); local planLen, planCount = 0, #plan
  for_loop_range(2, planCount, 96, 97)
  for i = 2, planCount do
    for_loop_range(i, planCount, 96, 97)
    process_line(166); planLen = planLen + plan[i].pos:distance(plan[i-1].pos)
  end
  enter_or_exit_function('exit', 'getPlanLen')
  process_line(167); return planLen, planCount
end

local function buildNextRoute(plan, planCount, path)
  enter_or_exit_function('enter', 'buildNextRoute')
  process_line(168); local nextPathIdx = (plan[planCount].pathidx or 0) + 1 -- if the first plan node is the aiPos it does not have a pathIdx value yet

  if (predicate_analyzer(race == true and noOfLaps and noOfLaps > 1 and not path[nextPathIdx], 98, 99)) then -- in case the path loops
    process_line(169); local loopPathId
    process_line(170); local pathCount = #path
    process_line(171); local lastWayPoint = path[pathCount]
    for_loop_range(1, pathCount, 100, 101)
    for i = 1, pathCount do
      for_loop_range(i, pathCount, 100, 101)
      if (predicate_analyzer(lastWayPoint == path[i], 102, 103)) then
        process_line(172); loopPathId = i
        process_line(173); break
      end
    end
    process_line(174); nextPathIdx = 1 + loopPathId -- nextPathIdx % #path
    process_line(175); noOfLaps = noOfLaps - 1
  end

  process_line(176); local nextNodeName = path[nextPathIdx]
  if (predicate_analyzer(not nextNodeName, 104, 105)) then
    enter_or_exit_function('exit', 'buildNextRoute')
    process_line(177); return
  end

  process_line(178); local n1Pos, oneWay
  process_line(179); local n2 = mapData.graph[nextNodeName]
  if (predicate_analyzer(not n2, 106, 107)) then
    enter_or_exit_function('exit', 'buildNextRoute')
    process_line(180); return
  end
  process_line(181); local n2Pos = mapData.positions[nextNodeName]
  process_line(182); local n2Radius = mapData.radius[nextNodeName]
  process_line(183); local validOneWayLink = true -- test for if oneWay road does not merge into a not oneWay road

  if (predicate_analyzer(path[nextPathIdx-1], 108, 109)) then
    process_line(184); n1Pos = mapData.positions[path[nextPathIdx-1]]
    process_line(185); local link = mapData.graph[path[nextPathIdx-1]][path[nextPathIdx]]
    process_line(186); local legalSpeed
    if (predicate_analyzer(link, 110, 111)) then
      process_line(187); oneWay = link[3]
      if (predicate_analyzer(oneWay, 112, 113)) then
        for_loop_pair(0, mapData.graph[path[nextPathIdx]], 114, 115)
        for k, v in pairs(mapData.graph[path[nextPathIdx]]) do
          if (predicate_analyzer(k ~= path[nextPathIdx - 1] and not v[3] and ((n2Pos - n1Pos):normalized()):dot((mapData.positions[k] - n2Pos):normalized()) > 0.2, 116, 117)) then
            process_line(188); validOneWayLink = false
          end
        end
      end

      if (predicate_analyzer(nextPathIdx > 2, 118, 119)) then
        process_line(189); legalSpeed = math.min(link[4], mapData.graph[path[nextPathIdx-1]][path[nextPathIdx-2]][4])
      else
        process_line(190); legalSpeed = link[4]
      end
    end
    process_line(191); plan[planCount].legalSpeed = legalSpeed
  elseif (predicate_analyzer(path[nextPathIdx+1], 120, 121)) then
    process_line(192); n1Pos = vec3(aiPos)
    process_line(193); local link = mapData.graph[path[nextPathIdx]][path[nextPathIdx+1]]
    process_line(194); oneWay = link and link[3] -- why do we need the link check here?
  end

  if (predicate_analyzer(driveInLaneFlag, 122, 123)) then
    process_line(195); local lane = 1
    process_line(196); plan.lane = plan.lane or 0
    if (predicate_analyzer(oneWay and validOneWayLink, 124, 125)) then
      if (predicate_analyzer(plan.lane ~= 0, 126, 127)) then
        process_line(197); lane = plan.lane
      else
        if (predicate_analyzer(path[2], 128, 129)) then
          process_line(198); local curPathIdx = (plan[1] and plan[1].pathidx) and math.max(2, plan[1].pathidx) or 2
          process_line(199); local p1Pos = mapData.positions[path[curPathIdx-1]]
          process_line(200); lane = sign((mapData.positions[path[curPathIdx]] - p1Pos):z0():cross(gravityDir):dot(p1Pos - aiPos))
          process_line(201); plan.lane = lane
        end
      end
    else
      process_line(202); plan.lane = 0
    end

    process_line(203); local nVec1
    if (predicate_analyzer(path[nextPathIdx-1], 130, 131)) then
      process_line(204); nVec1 = (n1Pos - n2Pos):z0():cross(gravityDir):normalized()
    else
      process_line(205); nVec1 = vec3()
    end

    process_line(206); local nVec2
    if (predicate_analyzer(path[nextPathIdx+1], 132, 133)) then
      process_line(207); nVec2 = (n2Pos - mapData.positions[path[nextPathIdx+1]]):z0():cross(gravityDir):normalized()
    else
      process_line(208); nVec2 = vec3()
    end

    process_line(209); local width = math.max(n2Radius * 0.5, ai.width * 0.7)
    process_line(210); local displacement = math.max(0, n2Radius - width) -- provide a bit more space in narrow roads so other vehicles can overtake
    process_line(211); n2Pos = n2Pos + displacement * lane * (1 - nVec1:dot(nVec2) * 0.5) * (nVec1 + nVec2)
    process_line(212); n2Radius = width
  end

  process_line(213); local lastPlanPos = plan[planCount] and plan[planCount].pos or aiPos
  process_line(214); local vec = (lastPlanPos - n2Pos):z0()
  process_line(215); local manSpeed = speedList and speedList[nextNodeName]

  enter_or_exit_function('exit', 'buildNextRoute')
  process_line(216); return {pos = vec3(n2Pos), posOrig = vec3(n2Pos), radius = n2Radius, radiusOrig = n2Radius,  posz0 = n2Pos:z0(), vec = vec, dirVec = vec:normalized(), turnDir = vec3(0,0,0), manSpeed = manSpeed, pathidx = nextPathIdx}
end

local function mergePathPrefix(source, dest, srcStart)
  enter_or_exit_function('enter', 'mergePathPrefix')
  process_line(217); srcStart = srcStart or 1
  process_line(218); local sourceCount = #source
  process_line(219); local dict = table.new(0, sourceCount-(srcStart-1))
  for_loop_range(srcStart, sourceCount, 134, 135)
  for i = srcStart, sourceCount do
    for_loop_range(i, sourceCount, 134, 135)
    process_line(220); dict[source[i]] = i
  end

  process_line(221); local destCount = #dest
  for_loop_range(1, destCount, 136, 137)
  for i = destCount, 1, -1 do
    for_loop_range(1, i, 136, 137)
    process_line(222); local srci = dict[dest[i]]
    if (predicate_analyzer(srci ~= nil, 138, 139)) then
      process_line(223); local res = table.new(destCount, 0)
      process_line(224); local resi = 1
      for_loop_range(srcStart, srci - 1, 140, 141)
      for i1 = srcStart, srci - 1 do
        for_loop_range(i1, srci - 1, 140, 141)
        process_line(225); res[resi] = source[i1]
        process_line(226); resi = resi + 1
      end
      for_loop_range(i, destCount, 142, 143)
      for i1 = i, destCount do
        for_loop_range(i1, destCount, 142, 143)
        process_line(227); res[resi] = dest[i1]
        process_line(228); resi = resi + 1
      end

      enter_or_exit_function('exit', 'mergePathPrefix')
      process_line(229); return res, srci
    end
  end

  enter_or_exit_function('exit', 'mergePathPrefix')
  process_line(230); return dest, 0
end

local function planAhead(route, baseRoute)
  enter_or_exit_function('enter', 'planAhead')
  if (predicate_analyzer(route == nil, 144, 145)) then
    enter_or_exit_function('exit', 'planAhead')
    process_line(231); return
  end
  if (predicate_analyzer(route.path == nil, 146, 147)) then
    process_line(232); route.path = {}
    for_loop_range(1, #route, 148, 149)
    for i = 1, #route do
      for_loop_range(i, #route, 148, 149)
      process_line(233); route.path[i] = route[i]
      process_line(234); route[i] = nil
    end
    process_line(235); route.plan = {}
  end

  process_line(236); local plan = route.plan

  if (predicate_analyzer(baseRoute and not plan[1], 150, 151)) then
    -- merge from base plan
    process_line(237); local bsrPlan = baseRoute.plan
    if (predicate_analyzer(bsrPlan[2], 152, 153)) then
      process_line(238); local commonPathEnd
      process_line(239); route.path, commonPathEnd = mergePathPrefix(baseRoute.path, route.path, bsrPlan[2].pathidx)
      if (predicate_analyzer(commonPathEnd >= 1, 154, 155)) then
        process_line(240); local refpathidx = bsrPlan[2].pathidx - 1
        for_loop_range(1, #bsrPlan, 156, 157)
        for i = 1, #bsrPlan do
          for_loop_range(i, #bsrPlan, 156, 157)
          process_line(241); local n = bsrPlan[i]
          if (predicate_analyzer(n.pathidx > commonPathEnd, 158, 159)) then
            process_line(242); break
          end
          process_line(243); plan[i] = {pos = vec3(n.pos), posOrig = vec3(n.posOrig), posz0 = vec3(n.posz0), vec = vec3(n.vec), dirVec = vec3(n.dirVec), turnDir = vec3(n.turnDir),  radius = n.radius, radiusOrig = n.radiusOrig, pathidx = math.max(1, n.pathidx-refpathidx), legalSpeed = n.legalSpeed}
        end
        if (predicate_analyzer(plan[bsrPlan.targetSeg+1], 160, 161)) then
          process_line(244); plan.targetSeg = bsrPlan.targetSeg
          process_line(245); plan.targetPos = vec3(bsrPlan.targetPos)
          process_line(246); plan.aiSeg = bsrPlan.aiSeg
        end
      end
    end
  end

  if (predicate_analyzer(not plan[1], 162, 163)) then
    process_line(247); plan[1] = {posOrig = vec3(aiPos), pos = vec3(aiPos), posz0 = aiPos:z0(), vec = (-8) * aiDirVec,  dirVec = -aiDirVec, turnDir = vec3(0,0,0), radiusOrig = 2, radius = 2}
  end

  process_line(248); local planLen, planCount = getPlanLen(plan)
  process_line(249); local minPlanLen = getMinPlanLen()
  process_line(250); while(predicate_analyzer(not plan[minPlanCount] or planLen < minPlanLen, 164, 165)) do
    process_line(251); local n = buildNextRoute(plan, planCount, route.path)
    if (predicate_analyzer(not n, 166, 167)) then
      process_line(252); break
    end
    process_line(253); planCount = planCount + 1
    process_line(254); plan[planCount] = n
    process_line(255); planLen = planLen + n.pos:distance(plan[planCount-1].pos)
  end

  if (predicate_analyzer(not plan[2], 168, 169)) then
    enter_or_exit_function('exit', 'planAhead')
    process_line(256); return
  end
  if (predicate_analyzer(not plan[1].pathidx, 170, 171)) then
    process_line(257); plan[1].pathidx = plan[2].pathidx
  end

  process_line(258); do
    process_line(259); local segmentSplitDelay = plan.segmentSplitDelay or 0
    process_line(260); local distOnPlan = 0
    for_loop_range(1, planCount-1, 172, 173)
    for i = 1, planCount-1 do
      for_loop_range(i, planCount-1, 172, 173)
      process_line(261); local curDist = plan[i].posOrig:squaredDistance(plan[i+1].posOrig)
      process_line(262); local xSq = square(distOnPlan)
      if (predicate_analyzer(curDist > square(math.min(220, (25e-8 * xSq + 1e-5) * xSq + 6)) and distOnPlan < 550, 174, 175)) then
        if (predicate_analyzer(segmentSplitDelay == 0, 176, 177)) then
          process_line(263); local n1, n2 = plan[i], plan[i+1]
          process_line(264); local pos = (n1.pos + n2.pos) * 0.5
          process_line(265); local vec = (n1.pos - pos):z0()
          process_line(266); n2.vec = (pos - n2.pos):z0()
          process_line(267); n2.dirVec = n2.vec:normalized()
          process_line(268); local legalSpeed
          if (predicate_analyzer(n2.pathidx > 1, 178, 179)) then
            process_line(269); legalSpeed = mapData.graph[route.path[n2.pathidx]][route.path[n2.pathidx-1]][4]
          else
            process_line(270); legalSpeed = n2.legalSpeed
          end
          process_line(271); table.insert(plan, i+1, {posOrig = (n1.posOrig + n2.posOrig) * 0.5, pos = pos, posz0 = pos:z0(),  vec = vec, dirVec = vec:normalized(), turnDir = vec3(0, 0, 0), radiusOrig = (n1.radiusOrig + n2.radiusOrig) * 0.5, radius = (n1.radius + n2.radius) * 0.5, pathidx = n2.pathidx, legalSpeed = legalSpeed})
          process_line(272); planCount = planCount + 1
          process_line(273); segmentSplitDelay = math.min(5, math.floor(90/aiSpeed))
        else
          process_line(274); segmentSplitDelay = segmentSplitDelay - 1
        end
        process_line(275); break
      end
      process_line(276); distOnPlan = distOnPlan + math.sqrt(curDist)
    end
    process_line(277); plan.segmentSplitDelay = segmentSplitDelay
  end

  if (predicate_analyzer(plan.targetSeg == nil, 180, 181)) then
    process_line(278); local aiXnormOnSeg
    process_line(279); plan.targetPos, plan.targetSeg, aiXnormOnSeg, planCount = calculateTarget(plan, planCount)
    process_line(280); plan.aiSeg = 1
  end

  for_loop_range(0, planCount, 182, 183)
  for i = 0, planCount do
    for_loop_range(i, planCount, 182, 183)
    if (predicate_analyzer(forces[i], 184, 185)) then
      process_line(281); forces[i]:set(0,0,0)
    else
      process_line(282); forces[i] = vec3(0,0,0)
    end
  end

  -- calculate spring forces
  for_loop_range(1, planCount-1, 186, 187)
  for i = 1, planCount-1 do
    for_loop_range(i, planCount-1, 186, 187)
    process_line(283); local n1 = plan[i]
    process_line(284); local n2 = plan[i+1]
    process_line(285); local v1 = n1.dirVec
    process_line(286); local v2 = -n2.dirVec
    process_line(287); local turnDir = (v1 + v2):normalized()

    process_line(288); local nforce = (1-threewayturn.state) * math.max(1 + v1:dot(v2), 0) * 2 * turnDir
    process_line(289); forces[i+1]:setSub(nforce)
    process_line(290); forces[i-1]:setSub(nforce)
    process_line(291); nforce:setScaled(2)
    process_line(292); forces[i]:setAdd(nforce)

    process_line(293); n1.turnDir:set(turnDir)
    process_line(294); n1.speed = 0
  end

  if (predicate_analyzer(M.mode == 'traffic', 188, 189)) then
    process_line(295); doTrafficActions(route.path, plan)
  end

  -- other vehicle awareness
  process_line(296); plan.trafficMinProjSpeed = math.huge
  process_line(297); plan.trafficBlockCoef = nil
  if (predicate_analyzer(avoidCars ~= 'off', 190, 191)) then
    process_line(298); table.clear(trafficTable)
    process_line(299); local trafficTableLen = 0
    for_loop_pair(0, mapmgr.objects, 192, 193)
    for plID, v in pairs(mapmgr.objects) do
      for_loop_pair(plID, mapmgr.objects, 192, 193)
      if (predicate_analyzer(plID ~= objectId and (M.mode ~= 'chase' or plID ~= player.id or chaseData.suspectState == 'stopped'), 194, 195)) then
        process_line(300); v.length = obj:getObjectInitialLength(plID) + 0.3
        process_line(301); v.width = obj:getObjectInitialWidth(plID)
        process_line(302); v.targetType = (player and plID == player.id) and M.mode
        if (predicate_analyzer(v.targetType == 'follow', 196, 197)) then
          process_line(303); v.width = v.width * 4
        end
        process_line(304); local posFront = obj:getObjectFrontPosition(plID)
        process_line(305); local dirVec = v.dirVec
        process_line(306); v.posFront = dirVec * 0.3 + posFront
        process_line(307); v.posRear = dirVec * (-v.length) + posFront
        process_line(308); v.lightbar = ((v.states and v.states.lightbar) and v.states.lightbar > 0) and true or false
        process_line(309); table.insert(trafficTable, v)
        process_line(310); trafficTableLen = trafficTableLen + 1
      end
    end

    process_line(311); local trafficMinSpeedSq = math.huge
    process_line(312); local distanceT = 0
    process_line(313); local aiPathVel = aiVel:dot((plan[2].pos-plan[1].pos):normalized())
    process_line(314); local aiPathVelInv = 1 / math.abs(aiPathVel + 1e-30)
    process_line(315); local minTrafficDir = 1

    for_loop_range(2, planCount-1, 198, 199)
    for i = 2, planCount-1 do
      for_loop_range(i, planCount-1, 198, 199)
      process_line(316); local n1, n2 = plan[i], plan[i+1]
      process_line(317); local n1pos, n2pos = n1.pos, n2.pos
      process_line(318); local n1n2 = n2pos - n1pos
      process_line(319); local n1n2len = n1n2:length()
      process_line(320); local nDir = n1n2 / (n1n2len + 1e-30)
      process_line(321); n1.trafficSqVel = math.huge
      process_line(322); local arrivalT = distanceT * aiPathVelInv

      if (predicate_analyzer(damageFlag or (intersection.planStop and intersection.planStop < i), 200, 201)) then
        process_line(323); n1.trafficSqVel = 0
      else
        for_loop_range(1, trafficTableLen, 202, 203)
        for j = trafficTableLen, 1, -1 do
          for_loop_range(1, j, 202, 203)
          process_line(324); local v = trafficTable[j]
          process_line(325); local plPosFront, plPosRear, plWidth = v.posFront, v.posRear, v.width
          process_line(326); local ai2PlVec = plPosFront - aiPos
          process_line(327); local ai2PlDir = ai2PlVec:dot(aiDirVec)
          process_line(328); local ai2PlSqDist = ai2PlVec:squaredLength()
          if (predicate_analyzer(ai2PlDir > 0, 204, 205)) then
            process_line(329); local velDisp = arrivalT * v.vel
            process_line(330); plPosFront = plPosFront + velDisp
            process_line(331); plPosRear = plPosRear + velDisp
          end
          process_line(332); local extVec = nDir * (math.max(ai.width, plWidth) * 0.5)
          process_line(333); local n1ext, n2ext = n1pos - extVec, n2pos + extVec
          process_line(334); local rnorm, vnorm = closestLinePoints(n1ext, n2ext, plPosFront, plPosRear)

          if (predicate_analyzer(M.mode == 'traffic' and v.lightbar and ai2PlSqDist <= 10000, 206, 207)) then -- lightbar awareness
            process_line(335); local tmpVec = ai.rightVec * 2
            process_line(336); forces[i]:setAdd(tmpVec)
            process_line(337); forces[i + 1]:setAdd(tmpVec)
            process_line(338); trafficSide.cTimer = math.max(5, trafficSide.cTimer)
            process_line(339); n1.trafficSqVel = clamp(math.sqrt(ai2PlSqDist) * 2 - 25, 0, 200)
          end

          process_line(340); local minSqDist = math.huge
          if (predicate_analyzer(rnorm > 0 and rnorm < 1 and vnorm > 0 and vnorm < 1, 208, 209)) then
            process_line(341); minSqDist = 0
          else
            process_line(342); local rlen = n1n2len + plWidth
            process_line(343); local xnorm = plPosFront:xnormOnLine(n1ext, n2ext) * rlen
            if (predicate_analyzer(xnorm > 0 and xnorm < rlen, 210, 211)) then
              process_line(344); minSqDist = math.min(minSqDist, (n1ext + nDir * xnorm):squaredDistance(plPosFront))
            end

            process_line(345); xnorm = plPosRear:xnormOnLine(n1ext, n2ext) * rlen
            if (predicate_analyzer(xnorm > 0 and xnorm < rlen, 212, 213)) then
              process_line(346); minSqDist = math.min(minSqDist, (n1ext + nDir * xnorm):squaredDistance(plPosRear))
            end

            process_line(347); rlen = v.length + ai.width
            process_line(348); local v1 = vec3(n1ext)
            process_line(349); v1:setSub(plPosRear)
            process_line(350); local v1dot = v1:dot(v.dirVec)
            if (predicate_analyzer(v1dot > 0 and v1dot < rlen, 214, 215)) then
              process_line(351); minSqDist = math.min(minSqDist, v1:squaredDistance(v1dot * v.dirVec))
            end

            process_line(352); v1:set(n2ext)
            process_line(353); v1:setSub(plPosRear)
            process_line(354); v1dot = v1:dot(v.dirVec)
            if (predicate_analyzer(v1dot > 0 and v1dot < rlen, 216, 217)) then
              process_line(355); minSqDist = math.min(minSqDist, v1:squaredDistance(v1dot * v.dirVec))
            end
          end

          if (predicate_analyzer(minSqDist < square((ai.width + plWidth) * 0.8), 218, 219)) then
            process_line(356); local velProjOnSeg = math.max(0, v.vel:dot(nDir))
            process_line(357); local middlePos = (plPosFront + plPosRear) * 0.5
            process_line(358); local forceCoef = trafficSide.side * 0.5 * math.max(0, math.max(aiSpeed - velProjOnSeg, sign(-(nDir:dot(v.dirVec))) * trafficSide.cTimer)) / ((1 + minSqDist) * (1 + distanceT * math.min(0.1, 1 / (2 * math.max(0, aiPathVel - v.vel:dot(nDir)) + 1e-30))))

            if (predicate_analyzer(intersection.planStop or v.targetType ~= 'follow', 220, 221)) then
              process_line(359); forces[i]:setSub((sign(n1.turnDir:dot(middlePos - n1.posOrig)) * forceCoef) * n1.turnDir)
              process_line(360); forces[i+1]:setSub((sign(n2.turnDir:dot(middlePos - n2.posOrig)) * forceCoef) * n2.turnDir)
            end

            if (predicate_analyzer(avoidCars == 'on' and M.mode ~= 'flee' and M.mode ~= 'random', 222, 223)) then
              if (predicate_analyzer(minSqDist < square((ai.width + plWidth) * 0.51) , 224, 225)) then
                -- obj.debugDrawProxy:drawSphere(0.25, v.posFront:toFloat3(), color(0,0,255,255))
                -- obj.debugDrawProxy:drawSphere(0.25, plPosFront:toFloat3(), color(0,0,255,255))
                process_line(361); table.remove(trafficTable, j)
                process_line(362); trafficTableLen = trafficTableLen - 1
                process_line(363); plan.trafficMinProjSpeed = math.min(plan.trafficMinProjSpeed, velProjOnSeg)

                process_line(364); n1.trafficSqVel = math.min(n1.trafficSqVel, velProjOnSeg * velProjOnSeg)
                process_line(365); trafficMinSpeedSq = math.min(trafficMinSpeedSq, v.vel:squaredLength())
                process_line(366); minTrafficDir = math.min(minTrafficDir, v.dirVec:dot(nDir))
              end

              if (predicate_analyzer(i == 2 and minSqDist < square((ai.width + plWidth) * 0.6) and ai2PlDir > 0 and v.vel:dot(ai.rightVec) * ai2PlVec:dot(ai.rightVec) < 0, 226, 227)) then
                process_line(367); n1.trafficSqVel = math.max(0, n1.trafficSqVel - math.abs(1 - v.vel:dot(aiDirVec)) * (v.vel:length()))
              end
            end
          end
        end
      end
      process_line(368); distanceT = distanceT + n1n2len
    end

    if (predicate_analyzer(math.max(trafficMinSpeedSq, aiSpeed*aiSpeed) < 0.25, 228, 229)) then
      process_line(369); plan.trafficBlockCoef = clamp((1 - minTrafficDir) * 0.5, 0, 1)
    else
      process_line(370); plan.trafficBlockCoef = 0
    end
    process_line(371); plan[1].trafficSqVel = plan[2].trafficSqVel
  end

  -- spring force integrator
  for_loop_range(2, planCount, 230, 231)
  for i = 2, planCount do
    for_loop_range(i, planCount, 230, 231)
    process_line(372); local n = plan[i]
    process_line(373); local k = n.turnDir:dot(forces[i])
    process_line(374); local nodeDisplVec = n.pos + fsign(k) * math.min(math.abs(k), 0.5) * n.turnDir - n.posOrig
    process_line(375); local nodeDisplLen = nodeDisplVec:length()
    process_line(376); local maxDispl = math.max(0, n.radiusOrig - ai.width * (0.35 + 0.3 / (1 + trafficSide.cTimer * 0.1))) * math.min(1, aggression * (1 + trafficSide.cTimer * 0.3))
    process_line(377); local nodeDisplLenLim = clamp(nodeDisplLen, 0, maxDispl)
    --n.radius = math.max(n.radiusOrig - fsign(nodeDisplVec:dot(n.turnDir)) * nodeDisplLenLim - distFromEdge, distFromEdge)
    process_line(378); n.radius = math.max(0, n.radiusOrig - nodeDisplLenLim)
    process_line(379); n.pos = n.posOrig + (nodeDisplLenLim / (nodeDisplLen + 1e-30)) * nodeDisplVec
    process_line(380); n.posz0:set(n.pos)
    process_line(381); n.posz0.z = 0
    process_line(382); n.vec = plan[i-1].posz0 - n.posz0
    process_line(383); n.dirVec:set(n.vec)
    process_line(384); n.dirVec:normalize()
  end

  process_line(385); local targetSeg = plan.targetSeg
  -- smoothly distribute error from planline onto the front segments
  if (predicate_analyzer(targetSeg ~= nil and planCount > targetSeg and plan.targetPos and threewayturn.state == 0, 232, 233)) then
    process_line(386); local dTotal = 0
    process_line(387); local sumLen = {}
    for_loop_range(2, targetSeg - 1, 234, 235)
    for i = 2, targetSeg - 1  do
      for_loop_range(i, targetSeg - 1, 234, 235)
      process_line(388); sumLen[i] = dTotal
      process_line(389); dTotal = dTotal + plan[i+1].pos:distance(plan[i].pos)
    end
    process_line(390); dTotal = dTotal + plan.targetPos:distance(plan[targetSeg].pos)

    process_line(391); dTotal = math.max(1, dTotal)
    process_line(392); local p1, p2 = plan[1].pos, plan[2].pos
    process_line(393); local dispVec = (aiPos - linePointFromXnorm(p1, p2, aiPos:xnormOnLine(p1, p2)))
    process_line(394); dispVec:setScaled(0.5 * dt)
    process_line(395); aiDeviation = dispVec:dot((p2-p1):cross(ai.upVec):normalized())

    process_line(396); local dispVecRatio = dispVec / dTotal
    for_loop_range(3, targetSeg - 1, 236, 237)
    for i = targetSeg - 1, 3, -1 do
      for_loop_range(3, i, 236, 237)
      process_line(397); plan[i].pos:setAdd((dTotal - sumLen[i]) * dispVecRatio)
      process_line(398); plan[i].posz0 = plan[i].pos:z0()
      process_line(399); plan[i+1].vec = plan[i].posz0 - plan[i+1].posz0
      process_line(400); plan[i+1].dirVec = plan[i+1].vec:normalized()
    end

    process_line(401); plan[1].pos:setAdd(dispVec)
    process_line(402); plan[1].posz0 = plan[1].pos:z0()
    process_line(403); plan[2].pos:setAdd(dispVec)
    process_line(404); plan[2].posz0 = plan[2].pos:z0()
    if (predicate_analyzer(plan[3], 238, 239)) then
      process_line(405); plan[3].vec = plan[2].posz0 - plan[3].posz0
      process_line(406); plan[3].dirVec = plan[3].vec:normalized()
    end
  end

  process_line(407); plan.targetPos, plan.targetSeg, plan.aiXnormOnSeg, planCount = calculateTarget(plan)
  process_line(408); plan.aiSeg = 1
  process_line(409); plan.planCount = planCount

  -- plan speeds
  process_line(410); local totalAccel = math.min(aggression, staticFrictionCoef) * g

  process_line(411); local rLast = plan[planCount]
  if (predicate_analyzer(route.path[rLast.pathidx+1] or (race and noOfLaps and noOfLaps > 1), 240, 241)) then
    process_line(412); rLast.speed = rLast.manSpeed or math.sqrt(2 * 550 * totalAccel) -- shouldn't this be calculated based on the path length remaining?
  else
    process_line(413); rLast.speed = rLast.manSpeed or 0
  end

  -- speed planning
  process_line(414); local tmpEdgeVec = vec3()
  for_loop_range(1, planCount-1, 242, 243)
  for i = planCount-1, 1, -1 do
    for_loop_range(1, i, 242, 243)
    process_line(415); local n1 = plan[i]
    process_line(416); local n2 = plan[i+1]

    -- inclination calculation
    process_line(417); tmpEdgeVec:set(n2.pos) -- = n2.pos - n1.pos
    process_line(418); tmpEdgeVec:setSub(n1.pos)
    process_line(419); local dist = tmpEdgeVec:length() + 1e-30
    process_line(420); tmpEdgeVec:setScaled(1 / dist)

    process_line(421); local Gf = gravityVec:dot(tmpEdgeVec) -- acceleration due to gravity parallel to road segment, positive when downhill
    process_line(422); local Gt = gravityVec:distance(tmpEdgeVec * Gf) / g -- gravity vec normal to road segment

    process_line(423); local n2SpeedSq = square(n2.speed)

    process_line(424); local n0vec = plan[math.max(1, i-2)].posz0 - n1.posz0
    process_line(425); local n3vec = n1.posz0 - plan[math.min(planCount, i + 2)].posz0
    process_line(426); local curvature = math.min(math.min(inCurvature(n1.vec, n2.vec), inCurvature(n0vec, n3vec)), math.min(inCurvature(n0vec, n2.vec), inCurvature(n1.vec, n3vec))) + 1.6e-7

    process_line(427); local turnSpeedSq = totalAccel * Gt / curvature -- available centripetal acceleration * radius

    -- https://physics.stackexchange.com/questions/312569/non-uniform-circular-motion-velocity-optimization
    --local deltaPhi = 2 * math.asin(0.5 * n2.vec:length() * curvature) -- phi = phi2 - phi1 = 2 * math.asin(halfcord / radius)
    process_line(428); local n1SpeedSq = turnSpeedSq * math.sin(math.min(math.asin(math.min(1, n2SpeedSq / turnSpeedSq)) + 2*curvature*dist, math.pi*0.5))

    process_line(429); n1SpeedSq = math.min(n1SpeedSq, n1.trafficSqVel or math.huge)
    process_line(430); n1.trafficSqVel = math.huge

    process_line(431); n1.speed = n1.manSpeed or (M.speedMode == 'legal' and n1.legalSpeed and math.min(n1.legalSpeed, math.sqrt(n1SpeedSq))) or (M.speedMode == 'limit' and M.routeSpeed and math.min(M.routeSpeed, math.sqrt(n1SpeedSq))) or (M.speedMode == 'set' and M.routeSpeed) or math.sqrt(n1SpeedSq)
  end

  process_line(432); plan.targetSpeed = plan[1].speed + math.max(0, plan.aiXnormOnSeg) * (plan[2].speed - plan[1].speed)

  enter_or_exit_function('exit', 'planAhead')
  process_line(433); return plan
end

local function resetMapAndRoute()
  enter_or_exit_function('enter', 'resetMapAndRoute')
  process_line(434); mapData = nil
  process_line(435); signalsData = nil
  process_line(436); currentRoute = nil
  process_line(437); race = nil
  process_line(438); noOfLaps = nil
  process_line(439); damageFlag = false
  process_line(440); internalState = 'onroad'
  process_line(441); changePlanTimer = 0
  process_line(442); resetAggression()
  enter_or_exit_function('exit', 'resetMapAndRoute')
end

local function getMapEdges(cutOffDrivability, node)
  -- creates a table (edgeDict) with map edges with drivability > cutOffDrivability
  if mapData ~= nil then
    local allSCC = mapData:scc(node) -- An array of dicts containing all strongly connected components reachable from 'node'.
    local maxSccLen = 0
    local sccIdx
    for i, scc in ipairs(allSCC) do
      -- finds the scc with the most nodes
      local sccLen = scc[0] -- position at which the number of nodes in currentSCC is stored
      if sccLen > maxSccLen then
        sccIdx = i
        maxSccLen = sccLen
      end
      scc[0] = nil
    end
    local currentSCC = allSCC[sccIdx]
    local keySet = {}
    local keySetLen = 0
    edgeDict = {}
    for nid, n in pairs(mapData.graph) do
      if currentSCC[nid] or not driveInLaneFlag then
        for lid, data in pairs(n) do
          if (currentSCC[lid] or not driveInLaneFlag) and (data[2] > cutOffDrivability) then
            local inNode = data[3] or nid
            local outNode = inNode == nid and lid or nid
            keySetLen = keySetLen + 1
            keySet[keySetLen] = {inNode, outNode}
            edgeDict[inNode..'\0'..outNode] = 1
            if not data[3] or not driveInLaneFlag then
              edgeDict[outNode..'\0'..inNode] = 1
            end
          end
        end
      end
    end
    if keySetLen == 0 then
      return
    end
    local edge = keySet[math.random(keySetLen)]
    return edge[1], edge[2]
  end
end

local function newManualPath()
  local newRoute, n1, n2, dist
  local offRoad = false

  if manualPath then
    if currentRoute and currentRoute.path then
      pathExtend(currentRoute.path, manualPath)
    else
      newRoute = {plan = {}, path = manualPath}
      currentRoute = newRoute
    end
    manualPath = nil
  elseif wpList then
    if currentRoute and currentRoute.path then
      newRoute = {plan = currentRoute.plan, path = currentRoute.path}
    else
      n1, n2, dist = mapmgr.findClosestRoad(aiPos)

      if n1 == nil or n2 == nil then
        guihooks.message("Could not find a road network, or closest road is too far", 5, "AI debug")
        log('D', "AI", "Could not find a road network, or closest road is too far")
        return
      end

      if dist > 2 * math.max(mapData.radius[n1], mapData.radius[n2]) then
        offRoad = true
        local vec1 = mapData.positions[n1] - aiPos
        local vec2 = mapData.positions[n2] - aiPos

        if aiDirVec:dot(vec1) > 0 and aiDirVec:dot(vec2) > 0 then
          if vec1:squaredLength() > vec2:squaredLength() then
            n1, n2 = n2, n1
          end
        elseif aiDirVec:dot(mapData.positions[n2] - mapData.positions[n1]) > 0 then
          n1, n2 = n2, n1
        end
      elseif aiDirVec:dot(mapData.positions[n2] - mapData.positions[n1]) > 0 then
        n1, n2 = n2, n1
      end

      newRoute = {plan = {}, path = {n1}}
    end

    for i = 0, #wpList-1 do
      local wp1 = wpList[i] or newRoute.path[#newRoute.path]
      local wp2 = wpList[i+1]
      local route = mapData:getPath(wp1, wp2, driveInLaneFlag and 1e4 or 1)
      local routeLen = #route
      if routeLen == 0 or (routeLen == 1 and wp2 ~= wp1) then
        guihooks.message("Path between waypoints '".. wp1 .."' - '".. wp2 .."' Not Found", 7, "AI debug")
        log('D', "AI", "Path between waypoints '".. wp1 .."' - '".. wp2 .."' Not Found")
        return
      end

      for j = 2, routeLen do
        table.insert(newRoute.path, route[j])
      end
    end

    wpList = nil

    if not offRoad and newRoute.path[3] and newRoute.path[2] == n2 then
      table.remove(newRoute.path, 1)
    end

    currentRoute = newRoute
  end
end

local function validateUserInput(list)
  validateInput = nop
  list = list or wpList
  if not list then
    return
  end
  --local isValid = wpList[1] and true or false
  local isValid = list[1] and true or false
  for i = 1, #list do -- #wpList
    local nodeAlias = mapmgr.nodeAliases[list[i]]
    if nodeAlias then
      if mapData.graph[nodeAlias] then
        list[i] = nodeAlias
      else
        if isValid then
          guihooks.message("One or more of the waypoints were not found on the map. Check the game console for more info.", 6, "AI debug")
          log('D', "AI", "The waypoints with the following names could not be found on the Map")
          isValid = false
        end
        -- print(list[i])
      end
    end
  end

  return isValid
end

local function fleePlan()
  local newRoute
  if currentRoute and currentRoute.plan[2].pathidx > #currentRoute.path * 0.7 and currentRoute.path[3] and targetWPName == nil and internalState ~= 'offroad' and (aiPos - player.pos):dot(aiDirVec) >= 0 and currentRoute.plan.trafficMinProjSpeed > 3 then
    local path = currentRoute.path
    local pathCount = #path
    local cr1 = path[pathCount-1]
    local cr2 = path[pathCount]
    pathExtend(path, mapData:getFleePath(cr2, (mapData.positions[cr2] - mapData.positions[cr1]):normalized(), player.pos, getMinPlanLen(), 0.01, 0.01))
  elseif currentRoute == nil or changePlanTimer == 0 then
    local wp1, wp2 = mapmgr.findClosestRoad(aiPos)
    if wp1 == nil or wp2 == nil then
      internalState = 'offroad'
      return
    else
      internalState = 'onroad'
    end

    local dirVec
    if currentRoute and currentRoute.plan.trafficMinProjSpeed < 3 then
      changePlanTimer = 5
      dirVec = -aiDirVec
    else
      dirVec = aiDirVec
    end

    local startnode = pickAiWp(wp1, wp2, dirVec)
    if not targetWPName then -- flee without target
      newRoute = mapData:getFleePath(startnode, dirVec, player.pos, getMinPlanLen(), 0.01, 0.01)
    else -- flee to target
      newRoute = mapData:getPathAwayFrom(startnode, targetWPName, aiPos, player.pos)
      if next(newRoute) == nil then
        targetWPName = nil
      end
    end

    if not newRoute[1] then
      internalState = 'offroad'
      return
    else
      internalState = 'onroad'
    end

    local tempPlan = planAhead(newRoute, currentRoute)
    if tempPlan and (currentRoute == nil or changePlanTimer > 0 or (tempPlan.targetSpeed >= math.min(aiSpeed, currentRoute.plan.targetSpeed) and targetsCompatible(currentRoute, newRoute))) then
      currentRoute = newRoute
    end
  end

  if currentRoute ~= newRoute then
    planAhead(currentRoute)
  else
    changePlanTimer = math.max(1, changePlanTimer)
  end
end

local function chasePlan()
  enter_or_exit_function('enter', 'chasePlan')
  process_line(443); local positions = mapData.positions
  process_line(444); local radii = mapData.radius

  process_line(445); local wp1, wp2, dist1 = mapmgr.findClosestRoad(aiPos)
  if (predicate_analyzer(wp1 == nil or wp2 == nil, 244, 245)) then
    process_line(446); internalState = 'offroad'
    enter_or_exit_function('exit', 'chasePlan')
    process_line(447); return
  end

  if (predicate_analyzer(aiDirVec:dot(positions[wp1] - positions[wp2]) > 0, 246, 247)) then
    process_line(448); wp1, wp2 = wp2, wp1
  end

  process_line(449); local plwp1, plwp2, dist2 = mapmgr.findClosestRoad(player.pos)
  if (predicate_analyzer(plwp1 == nil or plwp2 == nil, 248, 249)) then
    process_line(450); internalState = 'offroad'
    enter_or_exit_function('exit', 'chasePlan')
    process_line(451); return
  end

  if (predicate_analyzer(dist1 > math.max(radii[wp1], radii[wp2]) * 2 and dist2 > math.max(radii[plwp1], radii[plwp2]) * 2, 250, 251)) then
    process_line(452); internalState = 'offroad'
    enter_or_exit_function('exit', 'chasePlan')
    process_line(453); return
  end

  process_line(454); local playerNode = plwp2
  process_line(455); local playerSpeed = player.vel:length()
  process_line(456); local plDriveVel = playerSpeed > 1 and player.vel or player.dirVec
  if (predicate_analyzer(plDriveVel:dot(positions[plwp1] - positions[plwp2]) > 0, 252, 253)) then
    process_line(457); plwp1, plwp2 = plwp2, plwp1
  end

  --chaseData.playerRoad = {plwp1, plwp2}

  process_line(458); local aiPlDist = aiPos:distance(player.pos) -- should this be a signed distance?
  process_line(459); local aggrValue = aggressionMode == 'manual' and M.extAggression or 0.9
  if (predicate_analyzer(M.mode == 'follow', 254, 255)) then
    process_line(460); setAggressionInternal(math.min(aggrValue, 0.3 + 0.002 * aiPlDist))
  else
    process_line(461); setAggressionInternal(aggrValue) -- constant value for better chase experience?
  end

  -- consider calculating the aggression value but then passing it through a smoother so that transitions between chase mode and follow mode are smooth

  if (predicate_analyzer(playerSpeed < 1.5, 256, 257)) then
    process_line(462); chaseData.suspectStoppedTimer = chaseData.suspectStoppedTimer + dt
  else
    process_line(463); chaseData.suspectStoppedTimer = 0
  end

  if (predicate_analyzer(chaseData.suspectStoppedTimer > 5, 258, 259)) then
    process_line(464); chaseData.suspectState = 'stopped'
    if (predicate_analyzer(aiPlDist < 20 and aiSpeed < 0.3, 260, 261)) then
      -- do not plan new route if stopped near player
      process_line(465); currentRoute = nil
      process_line(466); internalState = 'onroad'
      enter_or_exit_function('exit', 'chasePlan')
      process_line(467); return
    end
  else
    process_line(468); chaseData.suspectState = nil
  end

  if (predicate_analyzer(M.mode == 'follow' and aiSpeed < 0.3 and (wp1 == playerNode or wp2 == playerNode), 262, 263)) then
    process_line(469); currentRoute = nil
  end

  if (predicate_analyzer(currentRoute, 264, 265)) then
    process_line(470); local curPath = currentRoute.path
    process_line(471); local curPlan = currentRoute.plan
    process_line(472); local playerNodeInPath = waypointInPath(curPath, playerNode, curPlan[2].pathidx)
    process_line(473); local playerOtherWay = (player.pos - aiPos):dot(positions[wp2] - aiPos) < 0 and plDriveVel:dot(positions[wp2] - aiPos) < 0 -- player is moving other way from next ai wp
    process_line(474); local pathLen = getPathLen(curPath, playerNodeInPath or math.huge) -- curPlan[2].pathidx
    process_line(475); local playerMinPlanLen = getMinPlanLen(0, playerSpeed) -- math.min(curPlan[#curPlan].speed, curPlan[#curPlan-1].speed, aiSpeed)

    if (predicate_analyzer(not playerNodeInPath or playerOtherWay or (M.mode == 'chase' and pathLen < playerMinPlanLen), 266, 267)) then
      process_line(476); local newRoute = mapData:getPath(wp2, playerNode, driveInLaneFlag and 1e3 or 1) -- maybe wp2 should be curPath[curPlan[2].pathidx]
      --pathLen = getPathLen(newRoute)

      if (predicate_analyzer(M.mode == 'chase', 268, 269)) then -- and pathLen < playerMinPlanLen
        --local playerSpeed = player.vel:length() -- * 1.1
        if (predicate_analyzer(playerSpeed > 1, 270, 271)) then -- is this needed?
          --local playerMinPlanLen = getMinPlanLen(0, playerSpeed, staticFrictionCoef * 0.5)
          process_line(477); pathExtend(newRoute, mapData:getFleePath(playerNode, plDriveVel, player.pos, playerMinPlanLen, 0, 0)) -- math.max(minPlanLen-pathLen, playerMinPlanLen) -- mapNodes[plwp1].pos - mapNodes[plwp2].pos
        end
      end

      process_line(478); local tempPlan = planAhead(newRoute, currentRoute)
      if (predicate_analyzer(tempPlan and tempPlan.targetSpeed >= math.min(aiSpeed, curPlan.targetSpeed) and (tempPlan.targetPos-curPlan.targetPos):dot(aiDirVec) >= 0, 272, 273)) then
        process_line(479); currentRoute = newRoute
      else
        process_line(480); planAhead(currentRoute)
      end
    else
      process_line(481); planAhead(currentRoute)
    end

    --chaseData.playerSeg, chaseData.playerXnormOnSeg = posOnPlan(player.pos, currentRoute.plan)

    if (predicate_analyzer(M.mode == 'chase' and plPrevVel and (plwp2 == curPath[curPlan[2].pathidx] or plwp2 == curPath[curPlan[2].pathidx + 1]), 274, 275)) then
    --aiPlDist < math.max(20, aiSpeed * 2.5)
      process_line(482); local playerNodePos1 = positions[plwp2]
      process_line(483); local segDir = (playerNodePos1 - positions[plwp1])
      process_line(484); local targetLineDir = vec3(-segDir.y, segDir.x, 0):normalized()
      process_line(485); local xnorm = closestLinePoints(playerNodePos1, playerNodePos1 + targetLineDir, player.pos, player.pos + player.dirVec)
      process_line(486); local tarPos = playerNodePos1 + targetLineDir * clamp(xnorm, -radii[plwp2], radii[plwp2])

      process_line(487); local p2Target = (tarPos - player.pos):normalized()
      process_line(488); local plVel2Target = playerSpeed > 0.1 and player.vel:dot(p2Target) or 0
      --local plAccel = (plVel2Target - plPrevVel:dot(p2Target)) / dt
      --plAccel = plAccel + sign2(plAccel) * 1e-5
      --local plTimeToTarget = (math.sqrt(math.max(plVel2Target * plVel2Target + 2 * plAccel * (tarPos - player.pos):length(), 0)) - plVel2Target) / plAccel
      process_line(489); local plTimeToTarget = tarPos:distance(player.pos) / (plVel2Target + 1e-30) -- accel maybe not needed; this gives smooth results

      process_line(490); local aiVel2Target = aiSpeed > 0.1 and aiVel:dot((tarPos - aiPos):normalized()) or 0
      process_line(491); local aiTimeToTarget = tarPos:distance(aiPos) / (aiVel2Target + 1e-30)

      if (predicate_analyzer(aiTimeToTarget < plTimeToTarget, 276, 277)) then
        process_line(492); internalState = 'tail'
        -- return
      else
        process_line(493); internalState = 'onroad'
      end
    else
      process_line(494); internalState = 'onroad'
    end
  else
    if (predicate_analyzer(M.mode == 'follow' and aiPlDist < 20, 278, 279)) then
      -- do not plan new route if opponent is stopped and ai has reached opponent
      process_line(495); internalState = 'onroad'
      enter_or_exit_function('exit', 'chasePlan')
      process_line(496); return
    end

    process_line(497); local newRoute = mapData:getPath(wp2, playerNode, driveInLaneFlag and 1e3 or 1)

    process_line(498); local tempPlan = planAhead(newRoute)
    if (predicate_analyzer(tempPlan, 280, 281)) then
      process_line(499); currentRoute = newRoute
    end
  end
  enter_or_exit_function('exit', 'chasePlan')
end

local function warningAIDisabled(message)
  guihooks.message(message, 5, "AI debug")
  M.mode = 'disabled'
  M.updateGFX = nop
  resetMapAndRoute()
  stateChanged()
end

local function offRoadFollowControl()
  enter_or_exit_function('enter', 'offRoadFollowControl')
  if (predicate_analyzer(not player or not player.pos or not aiPos or not aiSpeed, 282, 283)) then
    enter_or_exit_function('exit', 'offRoadFollowControl')
    process_line(500); return 0, 0, 0
  end

  process_line(501); local ai2PlVec = player.pos - aiPos
  process_line(502); local ai2PlDist = ai2PlVec:length()
  process_line(503); local ai2PlDirVec = ai2PlVec / (ai2PlDist + 1e-30)
  process_line(504); local plSpeedFromAI = player.vel:dot(ai2PlDirVec)
  process_line(505); ai2PlDist = math.max(0, ai2PlDist - 12)
  process_line(506); local targetSpeed = math.sqrt(math.max(0, plSpeedFromAI*plSpeedFromAI*plSpeedFromAI / (math.abs(plSpeedFromAI) + 1e-30) + 2 * 9.81 * math.min(1, staticFrictionCoef) * ai2PlDist))
  process_line(507); local speedDif = targetSpeed - aiSpeed
  process_line(508); local throttle = clamp(speedDif, 0, 1)
  process_line(509); local brake = clamp(-speedDif, 0, 1)

  enter_or_exit_function('exit', 'offRoadFollowControl')
  process_line(510); return throttle, brake, targetSpeed
end

M.updateGFX = nop
local function updateGFX(dtGFX)
  enter_or_exit_function('enter', 'updateGFX')
  process_line(511); dt = dtGFX

  if (predicate_analyzer(mapData ~= mapmgr.mapData, 284, 285)) then
    process_line(512); currentRoute = nil
  end

  process_line(513); mapData = mapmgr.mapData
  process_line(514); signalsData = mapmgr.signalsData

  if (predicate_analyzer(mapData == nil, 286, 287)) then
    enter_or_exit_function('exit', 'updateGFX')
    process_line(515); return
  end

  -- local cgPos = obj:calcCenterOfGravity()
  -- aiPos:set(cgPos)
  -- aiPos.z = obj:getSurfaceHeightBelow(cgPos)
  process_line(516); local tmpPos = obj:getFrontPosition()
  process_line(517); aiPos:set(tmpPos)
  process_line(518); aiPos.z = math.max(aiPos.z - 1, obj:getSurfaceHeightBelow(tmpPos))
  process_line(519); ai.prevDirVec:set(aiDirVec)
  process_line(520); aiDirVec:set(obj:getDirectionVector())
  process_line(521); ai.upVec:set(obj:getDirectionVectorUp())
  process_line(522); aiVel:set(obj:getVelocity())
  process_line(523); aiSpeed = aiVel:length()
  process_line(524); ai.width = ai.width or obj:getInitialWidth()
  process_line(525); ai.length = ai.length or obj:getInitialLength()
  process_line(526); ai.rightVec = aiDirVec:cross(ai.upVec):normalized()
  process_line(527); staticFrictionCoef = 0.95 * obj:getStaticFrictionCoef()

  if (predicate_analyzer(trafficBlock.coef > 0, 288, 289)) then
    process_line(528); trafficBlock.timer = trafficBlock.timer + dt * trafficBlock.coef
  else
    process_line(529); trafficBlock.timer = trafficBlock.timer * 0.8
  end

  if (predicate_analyzer(math.max(lastCommand.throttle, lastCommand.throttle) > 0.5 and aiSpeed < 1, 290, 291)) then
    process_line(530); aiCannotMoveTime = aiCannotMoveTime + dt
  else
    process_line(531); aiCannotMoveTime = 0
  end

  if (predicate_analyzer(aiSpeed < 3, 292, 293)) then
    process_line(532); trafficSide.cTimer = trafficSide.cTimer + dt
    process_line(533); trafficSide.timer = (trafficSide.timer + dt) % (2 * trafficSide.timerRange)
    process_line(534); trafficSide.side = sign2(trafficSide.timerRange - trafficSide.timer)
  else
    process_line(535); trafficSide.cTimer = math.max(0, trafficSide.cTimer - dt)
    process_line(536); trafficSide.timer = 0
    process_line(537); trafficSide.side = 1
  end

  process_line(538); changePlanTimer = math.max(0, changePlanTimer - dt)

  ------------------ RANDOM MODE ----------------
  if (predicate_analyzer(M.mode == 'random', 294, 295)) then
    process_line(539); local newRoute
    if (predicate_analyzer(currentRoute == nil or currentRoute.plan[2].pathidx > #currentRoute.path * 0.5, 296, 297)) then
      process_line(540); local wp1, wp2 = mapmgr.findClosestRoad(aiPos)
      if (predicate_analyzer(wp1 == nil or wp2 == nil, 298, 299)) then
        process_line(541); warningAIDisabled("Could not find a road network, or closest road is too far")
        enter_or_exit_function('exit', 'updateGFX')
        process_line(542); return
      end

      if (predicate_analyzer(internalState == 'offroad', 300, 301)) then
        process_line(543); local vec1 = mapData.positions[wp1] - aiPos
        process_line(544); local vec2 = mapData.positions[wp2] - aiPos
        if (predicate_analyzer(aiDirVec:dot(vec1) > 0 and aiDirVec:dot(vec2) > 0, 302, 303)) then
          if (predicate_analyzer(vec1:squaredLength() > vec2:squaredLength(), 304, 305)) then
            process_line(545); wp1, wp2 = wp2, wp1
          end
        elseif (predicate_analyzer(aiDirVec:dot(mapData.positions[wp2] - mapData.positions[wp1]) > 0, 306, 307)) then
          process_line(546); wp1, wp2 = wp2, wp1
        end
      elseif (predicate_analyzer(aiDirVec:dot(mapData.positions[wp2] - mapData.positions[wp1]) > 0, 308, 309)) then
        process_line(547); wp1, wp2 = wp2, wp1
      end

      process_line(548); newRoute = mapData:getRandomPath(wp1, wp2, driveInLaneFlag and 1e4 or 1)

      if (predicate_analyzer(newRoute and newRoute[1], 310, 311)) then
        process_line(549); local tempPlan = planAhead(newRoute, currentRoute)
        if (predicate_analyzer(tempPlan, 312, 313)) then
          if (predicate_analyzer(not currentRoute, 314, 315)) then
            process_line(550); currentRoute = newRoute
          else
            process_line(551); local curPlanIdx = currentRoute.plan[2].pathidx
            process_line(552); local curPathCount = #currentRoute.path
            if (predicate_analyzer(curPlanIdx >= curPathCount * 0.9 or ((tempPlan.targetPos-aiPos):dot(aiDirVec) >= 0 and (curPlanIdx >= curPathCount*0.8 or tempPlan.targetSpeed >= aiSpeed)), 316, 317)) then
              process_line(553); currentRoute = newRoute
            end
          end
        end
      end
    end

    if (predicate_analyzer(currentRoute ~= newRoute, 318, 319)) then
      process_line(554); planAhead(currentRoute)
    end

  ------------------ TRAFFIC MODE ----------------
  elseif (predicate_analyzer(M.mode == 'traffic', 320, 321)) then
    process_line(555); local newRoute
    if (predicate_analyzer(currentRoute == nil or aiPos:squaredDistance(mapData.positions[currentRoute.path[#currentRoute.path]]) < square(getMinPlanLen()) or trafficBlock.timer > trafficBlock.limit, 322, 323)) then -- getPathLen(currentRoute.path, currentRoute.plan[2].pathidx)
      if (predicate_analyzer(currentRoute and currentRoute.path[3] and trafficBlock.timer <= trafficBlock.limit and internalState ~= 'offroad', 324, 325)) then
        process_line(556); local path = currentRoute.path
        process_line(557); local pathCount = #path
        process_line(558); local cr0, cr1, cr2 = path[pathCount-2], path[pathCount-1], path[pathCount]
        process_line(559); local cr2Pos = mapData.positions[cr2]
        process_line(560); local dir1 = (cr2Pos - mapData.positions[cr1]):normalized()
        process_line(561); local vec = cr2Pos - mapData.positions[cr0]
        process_line(562); local mirrorOfVecAboutdir1 = (2 * vec:dot(dir1) * dir1 - vec):normalized()
        process_line(563); pathExtend(path, mapData:getPathT(cr2, cr2Pos, getMinPlanLen(), 1e4, mirrorOfVecAboutdir1))
      else
        process_line(564); local wp1, wp2 = mapmgr.findClosestRoad(aiPos)

        if (predicate_analyzer(wp1 == nil or wp2 == nil, 326, 327)) then
          process_line(565); guihooks.message("Could not find a road network, or closest road is too far", 5, "AI debug")
          process_line(566); currentRoute = nil
          process_line(567); internalState = 'offroad'
          process_line(568); changePlanTimer = 0
          process_line(569); driveCar(0, 0, 0, 1)
          enter_or_exit_function('exit', 'updateGFX')
          process_line(570); return
        end

        process_line(571); local dirVec
        if (predicate_analyzer(trafficBlock.timer > trafficBlock.limit and not mapData.graph[wp1][wp2][3] and (mapData.radius[wp1] + mapData.radius[wp2]) * 0.5 > 2, 328, 329)) then
          process_line(572); dirVec = -aiDirVec
        else
          process_line(573); dirVec = aiDirVec
        end

        process_line(574); wp1 = pickAiWp(wp1, wp2, dirVec)

        -- local newRoute = mapData:getRandomPathG(wp1, aiDirVec, getMinPlanLen(), 0.4, 1 / (aiSpeed + 1e-30))
        --newRoute = mapData:getRandomPathG(wp1, dirVec, getMinPlanLen(), 0.4, math.huge)
        process_line(575); newRoute = mapData:getPathT(wp1, aiPos, getMinPlanLen(), 1e4, aiDirVec)

        if (predicate_analyzer(newRoute and newRoute[1], 330, 331)) then
          process_line(576); local tempPlan = planAhead(newRoute, currentRoute)

          if (predicate_analyzer(tempPlan, 332, 333)) then
            process_line(577); trafficBlock.limit = math.random() * 10 + 5
            process_line(578); trafficBlock.horn = math.random() >= 0.4 and trafficBlock.limit - (math.random() * 2.5) or math.huge -- horn time start
            process_line(579); trafficSignal.hLimit = math.random() * 2
            process_line(580); intersection.turn = 0

            if (predicate_analyzer(not currentRoute or trafficBlock.timer > trafficBlock.limit, 334, 335)) then
              process_line(581); trafficBlock.timer = 0
              process_line(582); currentRoute = newRoute
            elseif (predicate_analyzer(tempPlan.targetSpeed >= aiSpeed and targetsCompatible(currentRoute, newRoute), 336, 337)) then
              process_line(583); currentRoute = newRoute
            end
          end
        end
      end
    end

    if (predicate_analyzer(currentRoute ~= newRoute, 338, 339)) then
      process_line(584); planAhead(currentRoute)
    end

  ------------------ MANUAL MODE ----------------
  elseif (predicate_analyzer(M.mode == 'manual', 340, 341)) then
    if (predicate_analyzer(validateInput(wpList or manualPath), 342, 343)) then
      process_line(585); newManualPath()
    end

    if (predicate_analyzer(aggressionMode == 'rubberBand', 344, 345)) then
      process_line(586); updatePlayerData()
      if (predicate_analyzer(player ~= nil, 346, 347)) then
        if (predicate_analyzer((aiPos - player.pos):dot(aiDirVec) > 0, 348, 349)) then
          process_line(587); setAggressionInternal(math.max(math.min(0.1 + math.max((150 - player.pos:distance(aiPos))/150, 0), 1), 0.5))
        else
          process_line(588); setAggressionInternal()
        end
      end
    end

    process_line(589); planAhead(currentRoute)

  ------------------ SPAN MODE ------------------
  elseif (predicate_analyzer(M.mode == 'span', 350, 351)) then
    if (predicate_analyzer(currentRoute == nil, 352, 353)) then
      process_line(590); local positions = mapData.positions
      process_line(591); local wpAft, wpFore = mapmgr.findClosestRoad(aiPos)
      if (predicate_analyzer(not (wpAft and wpFore), 354, 355)) then
        process_line(592); warningAIDisabled("Could not find a road network, or closest road is too far")
        enter_or_exit_function('exit', 'updateGFX')
        process_line(593); return
      end
      if (predicate_analyzer(aiDirVec:dot(positions[wpFore] - positions[wpAft]) < 0, 356, 357)) then
        process_line(594); wpAft, wpFore = wpFore, wpAft
      end

      process_line(595); local target, targetLink

      if (predicate_analyzer(not edgeDict, 358, 359)) then
        -- creates the edgeDict and returns a random edge
        process_line(596); target, targetLink = getMapEdges(M.cutOffDrivability or 0, wpFore)
        if (predicate_analyzer(not target, 360, 361)) then
          process_line(597); warningAIDisabled("No available target with selected characteristics")
          enter_or_exit_function('exit', 'updateGFX')
          process_line(598); return
        end
      end

      process_line(599); local newRoute = {}

      process_line(600); while(predicate_analyzer(true, 362, 363)) do
        if (predicate_analyzer(not target, 364, 365)) then
          process_line(601); local maxDist = -math.huge
          process_line(602); local lim = 1
          process_line(603); repeat
            -- get most distant non walked edge
            for_loop_pair(0, edgeDict, 366, 367)
            for k, v in pairs(edgeDict) do
              for_loop_pair(k, edgeDict, 366, 367)
              if (predicate_analyzer(v <= lim, 368, 369)) then
                if (predicate_analyzer(lim > 1, 370, 371)) then
                  process_line(604); edgeDict[k] = 1
                end
                process_line(605); local i = string.find(k, '\0')
                process_line(606); local n1id = string.sub(k, 1, i-1)
                process_line(607); local sqDist = positions[n1id]:squaredDistance(aiPos)
                if (predicate_analyzer(sqDist > maxDist, 372, 373)) then
                  process_line(608); maxDist = sqDist
                  process_line(609); target = n1id
                  process_line(610); targetLink = string.sub(k, i+1, #k)
                end
              end
            end
            process_line(611); lim = math.huge -- if the first iteration does not produce a target
          process_line(612); until(predicate_analyzer(target, 374, 375))        end

        process_line(613); local nodeDegree = 1
        for_loop_pair(0, mapData.graph[target], 376, 377)
        for lid, edgeData in pairs(mapData.graph[target]) do
          -- we're looking for neighboring nodes other than the targetLink
          for_loop_pair(lid, mapData.graph[target], 376, 377)
          if (predicate_analyzer(lid ~= targetLink, 378, 379)) then
            process_line(614); nodeDegree = nodeDegree + 1
          end
        end
        if (predicate_analyzer(nodeDegree == 1, 380, 381)) then
          process_line(615); local key = target..'\0'..targetLink
          process_line(616); edgeDict[key] = edgeDict[key] + 1
        end

        process_line(617); newRoute = mapData:spanMap(wpFore, wpAft, target, edgeDict, driveInLaneFlag and 10e7 or 1)

        if (predicate_analyzer(not newRoute[2] and wpFore ~= target, 382, 383)) then
          -- remove edge from edgeDict list and get a new target (while loop will iterate again)
          process_line(618); edgeDict[target..'\0'..targetLink] = nil
          process_line(619); edgeDict[targetLink..'\0'..target] = nil
          process_line(620); target = nil
          if (predicate_analyzer(next(edgeDict) == nil, 384, 385)) then
            process_line(621); warningAIDisabled("Could not find a path to any of the possible targets")
            enter_or_exit_function('exit', 'updateGFX')
            process_line(622); return
          end
        elseif (predicate_analyzer(not newRoute[1], 386, 387)) then
          process_line(623); warningAIDisabled("No Route Found")
          enter_or_exit_function('exit', 'updateGFX')
          process_line(624); return
        else
          -- insert the second edge node in newRoute if it is not already contained
          process_line(625); local newRouteLen = #newRoute
          if (predicate_analyzer(newRoute[newRouteLen-1] ~= targetLink, 388, 389)) then
            process_line(626); newRoute[newRouteLen+1] = targetLink
          end
          process_line(627); break
        end
      end

      if (predicate_analyzer(planAhead(newRoute) == nil, 390, 391)) then
        enter_or_exit_function('exit', 'updateGFX')
        process_line(628); return
      end
      process_line(629); currentRoute = newRoute
    else
      process_line(630); planAhead(currentRoute)
    end

  ------------------ FLEE MODE ------------------
  elseif (predicate_analyzer(M.mode == 'flee', 392, 393)) then
    process_line(631); updatePlayerData()
    if (predicate_analyzer(player, 394, 395)) then
      if (predicate_analyzer(validateInput(), 396, 397)) then
        process_line(632); targetWPName = wpList[1]
        process_line(633); wpList = nil
      end

      process_line(634); local aggrValue = aggressionMode == 'manual' and M.extAggression or 1
      process_line(635); setAggressionInternal(math.max(0.3, aggrValue - 0.002 * player.pos:distance(aiPos)))

      process_line(636); fleePlan()

      if (predicate_analyzer(internalState == 'offroad', 398, 399)) then
        process_line(637); local targetPos = aiPos + (aiPos - player.pos) * 100
        process_line(638); local targetSpeed = math.huge
        process_line(639); driveToTarget(targetPos, 1, 0, targetSpeed)
        enter_or_exit_function('exit', 'updateGFX')
        process_line(640); return
      end
    else
      --guihooks.message("No vehicle to Flee from", 5, "AI debug") -- TODO: this freezes the up because it runs on the gfx step
      enter_or_exit_function('exit', 'updateGFX')
      process_line(641); return
    end

  ------------------ CHASE MODE ------------------
  elseif (predicate_analyzer(M.mode == 'chase' or M.mode == 'follow', 400, 401)) then
    process_line(642); updatePlayerData()
    if (predicate_analyzer(player, 402, 403)) then
      process_line(643); chasePlan()

      if (predicate_analyzer(plPrevVel, 404, 405)) then
        process_line(644); plPrevVel:set(player.vel)
      else
        process_line(645); plPrevVel = vec3(player.vel)
      end

      if (predicate_analyzer(internalState == 'tail', 406, 407)) then
        --internalState = 'onroad'
        --currentRoute = nil
        process_line(646); local plai = player.pos - aiPos
        process_line(647); local relvel = aiVel:dot(plai) - player.vel:dot(plai)
        if (predicate_analyzer(chaseData.suspectState == 'stopped', 408, 409)) then
          process_line(648); driveToTarget(player.pos, 0, 0.5, 0) -- could be better to apply throttle, brake, and targetSpeed values here
        elseif (predicate_analyzer(relvel > 0, 410, 411)) then
          process_line(649); driveToTarget(player.pos + (plai:length() / (relvel + 1e-30)) * player.vel, 1, 0, math.huge)
        else
          process_line(650); driveToTarget(player.pos, 1, 0, math.huge)
        end
        enter_or_exit_function('exit', 'updateGFX')
        process_line(651); return
      elseif (predicate_analyzer(internalState == 'offroad', 412, 413)) then
        if (predicate_analyzer(M.mode == 'follow', 414, 415)) then
          process_line(652); local throttle, brake, targetSpeed = offRoadFollowControl()
          process_line(653); driveToTarget(player.pos, throttle, brake, targetSpeed)
        else
          process_line(654); driveToTarget(player.pos, 1, 0, math.huge)
        end
        enter_or_exit_function('exit', 'updateGFX')
        process_line(655); return
      elseif (predicate_analyzer(currentRoute == nil, 416, 417)) then
        process_line(656); driveCar(0, 0, 0, 1)
        enter_or_exit_function('exit', 'updateGFX')
        process_line(657); return
      end

    else
      --guihooks.message("No vehicle to Chase", 5, "AI debug")
      enter_or_exit_function('exit', 'updateGFX')
      process_line(658); return
    end

  ------------------ STOP MODE ------------------
  elseif (predicate_analyzer(M.mode == 'stop', 418, 419)) then
    if (predicate_analyzer(currentRoute, 420, 421)) then
      process_line(659); planAhead(currentRoute)
      process_line(660); local targetSpeed = math.max(0, aiSpeed - math.sqrt(math.max(0, square(staticFrictionCoef * g) - square(sensors.gx2))) * dt)
      process_line(661); currentRoute.plan.targetSpeed = math.min(currentRoute.plan.targetSpeed, targetSpeed)
    elseif (predicate_analyzer(aiVel:dot(aiDirVec) > 0, 422, 423)) then
      process_line(662); driveCar(0, 0, 0.5, 0)
    else
      process_line(663); driveCar(0, 1, 0, 0)
    end
    if (predicate_analyzer(aiSpeed < 0.08, 424, 425)) then --  or aiVel:dot(aiDirVec) < 0
      process_line(664); driveCar(0, 0, 0, 1) -- only parkingbrake
      process_line(665); M.mode = 'disabled'
      process_line(666); M.manualTargetName = nil
      process_line(667); M.updateGFX = nop
      process_line(668); resetMapAndRoute()
      process_line(669); stateChanged()
      enter_or_exit_function('exit', 'updateGFX')
      process_line(670); return
    end
  end
  -----------------------------------------------

  if (predicate_analyzer(currentRoute, 426, 427)) then
    process_line(671); local plan = currentRoute.plan
    process_line(672); local targetPos = plan.targetPos
    process_line(673); local aiSeg = plan.aiSeg

    -- cleanup path if it has gotten too long
    if (predicate_analyzer(not race and plan[aiSeg].pathidx >= 10 and currentRoute.path[20], 428, 429)) then
      process_line(674); local newPath = {}
      process_line(675); local j, k = 0, plan[aiSeg].pathidx
      for_loop_range(k, #currentRoute.path, 430, 431)
      for i = k, #currentRoute.path do
        for_loop_range(i, #currentRoute.path, 430, 431)
        process_line(676); j = j + 1
        process_line(677); newPath[j] = currentRoute.path[i]
      end
      process_line(678); currentRoute.path = newPath

      process_line(679); k = k - 1
      for_loop_range(1, #plan, 432, 433)
      for i = 1, #plan do
        for_loop_range(i, #plan, 432, 433)
        process_line(680); plan[i].pathidx = plan[i].pathidx - k
      end
    end

    process_line(681); local targetSpeed = plan.targetSpeed
    process_line(682); trafficBlock.coef = plan.trafficBlockCoef or trafficBlock.coef

    if (predicate_analyzer(ai.upVec:dot(gravityVec) > 0, 434, 435)) then -- vehicle upside down
      enter_or_exit_function('exit', 'updateGFX')
      process_line(683); return
    end

    process_line(684); local lowTargetSpeedVal = 0.24
    if (predicate_analyzer(not plan[aiSeg+2] and ((targetSpeed < lowTargetSpeedVal and aiSpeed < 0.15) or (targetPos - aiPos):dot(aiDirVec) < 0), 436, 437)) then
      if (predicate_analyzer(M.mode == 'span', 438, 439)) then
        process_line(685); local path = currentRoute.path
        for_loop_range(1, #path - 1, 440, 441)
        for i = 1, #path - 1 do
          for_loop_range(i, #path - 1, 440, 441)
          process_line(686); local key = path[i]..'\0'..path[i+1]
          -- in case we have gone over an edge that is not in the edgeDict list
          process_line(687); edgeDict[key] = edgeDict[key] and (edgeDict[key] * 20)
        end
      end

      process_line(688); driveCar(0, 0, 0, 1)
      process_line(689); aistatus('route done', 'route')
      process_line(690); guihooks.message("Route done", 5, "AI debug")
      process_line(691); currentRoute = nil
      process_line(692); speedRecordings = {}
      enter_or_exit_function('exit', 'updateGFX')
      process_line(693); return
    end

    -- come off controls when close to intermediate node with zero speed (ex. intersection), arcade autobrake takes over
    if (predicate_analyzer((plan[aiSeg+1].speed == 0 and plan[aiSeg+2]) and aiSpeed < 0.15, 442, 443)) then
      process_line(694); driveCar(0, 0, 0, 0)
      enter_or_exit_function('exit', 'updateGFX')
      process_line(695); return
    end

      -- TODO: this still runs if there is no currentPlan, but raises error if there is no targetSpeed
    if (predicate_analyzer(not controller.isFrozen and aiSpeed < 0.1 and targetSpeed > 0.5 and (lastCommand.throttle ~= 0 or lastCommand.brake ~= 0), 444, 445)) then
      process_line(696); crash.time = crash.time + dt
      if (predicate_analyzer(crash.time > 1, 446, 447)) then
        process_line(697); crash.dir = vec3(aiDirVec)
        process_line(698); crash.manoeuvre = 1
      end
    else
      process_line(699); crash.time = 0
    end

    -- Throttle and Brake control
    process_line(700); local dif = targetSpeed - aiSpeed
    if (predicate_analyzer(dif <= 0, 448, 449)) then
      process_line(701); targetSpeedSmoother:set(dif)
    end
    process_line(702); local speedDif = targetSpeedSmoother:get(dif, dt)
    process_line(703); local lowSpeedDif = (speedDif - clamp((aiSpeed - 2) * 0.5, 0, 1)) * 0.5
    process_line(704); local lowTargSpeedConstBrake = lowTargetSpeedVal - targetSpeed -- apply constant brake below some targetSpeed
    process_line(705); local throttle = clamp(lowSpeedDif, 0, 1) * sign(math.max(0, -lowTargSpeedConstBrake)) -- throttle not enganged for targetSpeed < 0.26
    process_line(706); local brakeLimLow = sign(math.max(0, lowTargSpeedConstBrake)) * 0.5
    process_line(707); local brake = clamp(-speedDif, brakeLimLow, 1) * sign(math.max(0, electrics.values.smoothShiftLogicAV or 0 - 3)) -- arcade autobrake comes in at |smoothShiftLogicAV| < 5

    process_line(708); driveToTarget(targetPos, throttle, brake)
  end
  enter_or_exit_function('exit', 'updateGFX')
end

local function debugDraw(focusPos)
  local debugDrawer = obj.debugDrawProxy

  if M.mode == 'script' and scriptai ~= nil then
    scriptai.debugDraw()
  end

  if currentRoute then
    local plan = currentRoute.plan
    local targetPos = plan.targetPos
    local targetSpeed = plan.targetSpeed
    if targetPos then
      local tmpfl3 = float3(targetPos:xyz())
      debugDrawer:drawSphere(0.25, tmpfl3, color(255,0,0,255))

      local aiSeg = plan.aiSeg
      local shadowPos = currentRoute.plan[aiSeg].pos + plan.aiXnormOnSeg * (plan[aiSeg+1].pos - plan[aiSeg].pos)
      tmpfl3:set(shadowPos:xyz())
      local blue = color(0,0,255,255)
      debugDrawer:drawSphere(0.25, tmpfl3, blue)

      for plID, _ in pairs(mapmgr.objects) do
        if plID ~= objectId then
          local plPosFront = vec3(obj:getObjectFrontPosition(plID))
          tmpfl3:set(plPosFront:xyz())
          debugDrawer:drawSphere(0.25, tmpfl3, blue)
        end
      end

      if player then
        tmpfl3:set(player.pos:xyz())
        debugDrawer:drawSphere(0.3, tmpfl3, color(0,255,0,255))
      end
    end

    if M.debugMode == 'target' then
      if mapData and mapData.graph and currentRoute.path then
        local p = mapData.positions[currentRoute.path[#currentRoute.path]]:toFloat3()
        --debugDrawer:drawSphere(4, p, color(255,0,0,100))
        --debugDrawer:drawText(p + float3(0, 0, 4), color(0,0,0,255), 'Destination')
      end

    elseif M.debugMode == 'route' then
      if currentRoute.path then
        local p = mapData.positions[currentRoute.path[#currentRoute.path]]:toFloat3()
        debugDrawer:drawSphere(4, p, color(255,0,0,100))
        debugDrawer:drawText(p + float3(0, 0, 4), color(0,0,0,255), 'Destination')
      end

      local maxLen = 700
      local last = routeRec.last
      local len = math.min(#routeRec, maxLen)
      if len == 0 or (routeRec[last] - aiPos:toFloat3()):length() > 7 then
        last = 1 + last % maxLen
        routeRec[last] = aiPos:toFloat3()
        len = math.min(len+1, maxLen)
        routeRec.last = last
      end

      local fl3 = float3(0.7, ai.width, 0.7)
      local black = color(0,0,0,128)
      for i = 1, len-1 do
        debugDrawer:drawSquarePrism(routeRec[1+(last+i-1)%len], routeRec[1+(last+i)%len], fl3, fl3, black)
      end

      if currentRoute.plan[1].pathidx then
        local positions = mapData.positions
        local path = currentRoute.path
        fl3 = fl3 + float3(0, ai.width, 0)
        local transparentRed = color(255,0,0,120)
        local tmp1fl3 = float3(0, 0, 0)
        local tmp2fl3 = float3(0, 0, 0)
        for i = currentRoute.plan[1].pathidx, #path - 1 do
          tmp1fl3:set(positions[path[i]]:xyz())
          tmp2fl3:set(positions[path[i+1]]:xyz())
          debugDrawer:drawSquarePrism(tmp1fl3, tmp2fl3, fl3, fl3, transparentRed)
        end
      end

    elseif M.debugMode == 'speeds' then
      if plan[1] then
        local red = color(255,0,0,200) -- getContrastColor(objectId)
        local black = color(0, 0, 0, 255)
        local prevSpeed = -1
        local drawLen = 0
        local prevPoint = plan[1].pos:toFloat3()
        local tmpfl3 = float3(0, 0, 0)
        local p = float3(0, 0, 0)
        for i = 1, #plan do
          local n = plan[i]
          p:set(n.pos:xyz())
          local speed = (n.speed >= 0 and n.speed) or prevSpeed
          tmpfl3:set(0, 0, speed * 0.2)
          local p1 = p + tmpfl3
          debugDrawer:drawCylinder(p, p1, 0.03, red)
          debugDrawer:drawCylinder(prevPoint, p1, 0.05, red)
          debugDrawer:drawText(p1, black, string.format("%2.0f", speed*3.6).." kph")
          prevPoint = p1
          prevSpeed = speed
          --drawLen = drawLen + n.vec:length()
          -- if traffic and traffic[i] then
          --   for _, data in ipairs(traffic[i]) do
          --     local plPosOnPlan = linePointFromXnorm(n.pos, plan[i+1].pos, data[2])
          --     debugDrawer:drawSphere(0.25, plPosOnPlan:toFloat3(), color(0,255,0,100))
          --   end
          -- end
          -- if drawLen > 150 then break end
        end

        local aiPosFlt = aiPos:toFloat3()
        local speedRecLen = #speedRecordings

        if speedRecLen == 0 or (speedRecordings[speedRecLen][1]-aiPosFlt):length() > 0.25 then
          table.insert(speedRecordings, {aiPosFlt, aiSpeed, targetSpeed, lastCommand.brake, lastCommand.throttle})
          speedRecLen = speedRecLen + 1
        end

        local lastEntry
        local zOffSet = float3(0,0,0.5)
        local yellow, blue = color(255,255,0,200), color(0,0,255,200)
        local tmp2fl3 = float3(0, 0, 0)
        for i = 1, speedRecLen do
          local v = speedRecordings[i]
          if lastEntry then
            -- actuall speed
            tmpfl3:set(0, 0, lastEntry[2] * 0.2)
            tmp2fl3:set(0, 0, v[2] * 0.2)
            debugDrawer:drawCylinder(lastEntry[1] + tmpfl3, v[1] + tmp2fl3, 0.02, yellow)

            -- target speed
            tmpfl3:set(0, 0, lastEntry[3] * 0.2)
            tmp2fl3:set(0, 0, v[3] * 0.2)
            debugDrawer:drawCylinder(lastEntry[1] + tmpfl3, v[1] + tmp2fl3, 0.02, blue)
          end

          tmpfl3:set(0, 0, v[3] * 0.2)
          debugDrawer:drawCylinder(v[1], v[1] + tmpfl3, 0.01, blue)

          if (focusPos - v[1]):length() < labelRenderDistance then
            tmpfl3:set(0, 0, v[2] * 0.2)
            debugDrawer:drawText(v[1] + tmpfl3 + zOffSet, yellow, string.format("%2.0f", v[2]*3.6).." kph")

            tmpfl3:set(0, 0, v[3] * 0.2)
            debugDrawer:drawText(v[1] + tmpfl3 + zOffSet, blue, string.format("%2.0f", v[3]*3.6).." kph")
          end
          lastEntry = v
        end

        if speedRecLen > 175 then
          table.remove(speedRecordings, 1)
        end
      end

      -- Debug Throttle brake application
      local maxLen = 175
      local len = math.min(#trajecRec, maxLen)
      local last = trajecRec.last
      local aiPosFlt = aiPos:toFloat3()
      if len == 0 or (trajecRec[last][1]-aiPosFlt):length() > 0.25 then
        last = 1 + last % maxLen
        trajecRec[last] = {aiPosFlt, lastCommand.throttle, lastCommand.brake}
        len = math.min(len+1, maxLen)
        trajecRec.last = last
      end

      local fl3 = float3(0.7, ai.width, 0.7)
      for i = 1, len-1 do
        local n = trajecRec[1+(last+i)%len]
        debugDrawer:drawSquarePrism(trajecRec[1+(last+i-1)%len][1], n[1], fl3, fl3, color(255*math.sqrt(math.abs(n[3])), 255*math.sqrt(n[2]), 0, 100))
      end

      -- Player segment visual debug for chase / follow mode
      -- if chaseData.playerRoad then
      --   local col1, col2
      --   if internalState == 'tail' then
      --     col1 = color(0,0,0,200)
      --     col2 = color(0,0,0,200)
      --   else
      --     col1 = color(255,0,0,100)
      --     col2 = color(0,0,255,100)
      --   end
      --   local plwp1 = chaseData.playerRoad[1]
      --   debugDrawer:drawSphere(2, mapData.positions[plwp1]:toFloat3(), col1)
      --   local plwp2 = chaseData.playerRoad[2]
      --   debugDrawer:drawSphere(2, mapData.positions[plwp2]:toFloat3(), col2)
      -- end

    elseif M.debugMode == 'trajectory' then
      -- Debug Curvatures
      -- local plan = currentRoute.plan
      -- if plan ~= nil then
      --   local prevPoint = plan[1].pos:toFloat3()
      --   for i = 1, #plan do
      --     local p = plan[i].pos:toFloat3()
      --     local v = plan[i].curvature or 1e-10
      --     local scaledV = math.abs(1000 * v)
      --     debugDrawer:drawCylinder(p, p + float3(0, 0, scaledV), 0.06, color(math.abs(math.min(fsign(v),0))*255,math.max(fsign(v),0)*255,0,200))
      --     debugDrawer:drawText(p + float3(0, 0, scaledV), color(0,0,0,255), string.format("%5.4e", v))
      --     debugDrawer:drawCylinder(prevPoint, p + float3(0, 0, scaledV), 0.06, col)
      --     prevPoint = p + float3(0, 0, scaledV)
      --   end
      -- end

      -- Debug Planned Speeds
      if plan[1] then
        local col = getContrastColor(objectId)
        local prevPoint = plan[1].pos:toFloat3()
        local prevSpeed = -1
        local drawLen = 0
        for i = 1, #plan do
          local n = plan[i]
          local p = n.pos:toFloat3()
          local v = (n.speed >= 0 and n.speed) or prevSpeed
          local p1 = p + float3(0, 0, v*0.2)
          --debugDrawer:drawLine(p + float3(0, 0, v*0.2), (n.pos + n.turnDir):toFloat3() + float3(0, 0, v*0.2), col)
          debugDrawer:drawCylinder(p, p1, 0.03, col)
          debugDrawer:drawCylinder(prevPoint, p1, 0.05, col)
          debugDrawer:drawText(p1, color(0,0,0,255), string.format("%2.0f", v*3.6) .. " kph")
          prevPoint = p1
          prevSpeed = v
          drawLen = drawLen + n.vec:length()
          if drawLen > 80 then
            break
          end
        end
      end

      -- Debug Throttle brake application
      local maxLen = 175
      local len = math.min(#trajecRec, maxLen)
      local last = trajecRec.last
      if len == 0 or (trajecRec[last][1] - aiPos:toFloat3()):length() > 0.25 then
        last = 1 + last % maxLen
        trajecRec[last] = {aiPos:toFloat3(), lastCommand.throttle, lastCommand.brake}
        len = math.min(len+1, maxLen)
        trajecRec.last = last
      end

      local fl3 = float3(0.7, ai.width, 0.7)
      for i = 1, len-1 do
        local n = trajecRec[1+(last+i)%len]
        debugDrawer:drawSquarePrism(trajecRec[1+(last+i-1)%len][1], n[1], fl3, fl3, color(255*math.sqrt(math.abs(n[3])), 255*math.sqrt(n[2]), 0, 100))
      end
    end
  end
end

local function setAvoidCars(v)
  enter_or_exit_function('enter', 'setAvoidCars')
  process_line(709); M.avoidCarsMaster = v
  if (predicate_analyzer(M.avoidCarsMaster == 'off' or M.avoidCarsMaster == 'on', 450, 451)) then
    process_line(710); avoidCars = M.avoidCarsMaster
  else
    process_line(711); avoidCars = M.mode == 'manual' and 'off' or 'on'
  end
  process_line(712); stateChanged()
  enter_or_exit_function('exit', 'setAvoidCars')
end

local function driveInLane(v)
  enter_or_exit_function('enter', 'driveInLane')
  if (predicate_analyzer(v == 'on', 452, 453)) then
    process_line(713); M.driveInLaneFlag = 'on'
    process_line(714); driveInLaneFlag = true
  else
    process_line(715); M.driveInLaneFlag = 'off'
    process_line(716); driveInLaneFlag = false
  end
  process_line(717); stateChanged()
  enter_or_exit_function('exit', 'driveInLane')
end

local function setMode(mode)
  enter_or_exit_function('enter', 'setMode')
  if (predicate_analyzer(M.avoidCarsMaster == 'off' or M.avoidCarsMaster == 'on', 454, 455)) then
    process_line(718); avoidCars = M.avoidCarsMaster
  else
    process_line(719); avoidCars = (mode == 'manual' or (mode == nil and M.mode == 'manual')) and 'off' or 'on'
  end

  if (predicate_analyzer(mode ~= nil, 456, 457)) then
    process_line(720); M.mode = mode
  end

  if (predicate_analyzer(M.mode ~= 'script', 458, 459)) then
    if (predicate_analyzer(M.mode ~= 'disabled' and M.mode ~= 'stop', 460, 461)) then
      process_line(721); resetMapAndRoute()

      process_line(722); mapmgr.requestMap() -- a map request is also performed in the startFollowing function of scriptai
      process_line(723); M.updateGFX = updateGFX
      process_line(724); targetSpeedSmoother = newTemporalSmoothingNonLinear(math.huge, 0.2, vec3(obj:getVelocity()):length())

      if (predicate_analyzer(controller.mainController, 462, 463)) then
        process_line(725); controller.mainController.setGearboxMode("arcade")
      end
    end

    if (predicate_analyzer(M.mode == 'disabled', 464, 465)) then
      process_line(726); driveCar(0, 0, 0, 0)
      process_line(727); M.updateGFX = nop
      process_line(728); currentRoute = nil
    end

    if (predicate_analyzer(M.mode == 'traffic', 466, 467)) then
      process_line(729); setSpeedMode('legal')
      process_line(730); obj:setSelfCollisionMode(2)
      process_line(731); obj:setAerodynamicsMode(2)
    else
      process_line(732); obj:setSelfCollisionMode(1)
      process_line(733); obj:setAerodynamicsMode(1)
    end

    process_line(734); stateChanged()
  end

  process_line(735); speedRecordings = {}
  process_line(736); trajecRec = {last = 0}
  process_line(737); routeRec = {last = 0}
  enter_or_exit_function('exit', 'setMode')
end

local function reset() -- called when the user pressed I
  enter_or_exit_function('enter', 'reset')
  process_line(738); M.manualTargetName = nil
  process_line(739); trafficBlock = {timer = 0, coef = 0, limit = 6, horn = 0}

  process_line(740); trafficSide.timer = 0
  process_line(741); trafficSide.cTimer = 0
  process_line(742); trafficSide.side = 1

  process_line(743); trafficSignal = {hTimer = 0, hLimit = 1}
  process_line(744); intersection = {stopTime = 0, timer = 0, turn = 0}
  process_line(745); smoothTcs:set(1)

  if (predicate_analyzer(M.mode ~= 'disabled', 468, 469)) then
    process_line(746); driveCar(0, 0, 0, 0)
  end
  process_line(747); setMode() -- some scenarios don't work if this is changed to setMode('disabled')
  process_line(748); stateChanged()
  enter_or_exit_function('exit', 'reset')
end

local function resetLearning()
  enter_or_exit_function('enter', 'resetLearning')
  enter_or_exit_function('exit', 'resetLearning')
end

local function setVehicleDebugMode(newMode)
  tableMerge(M, newMode)
  if M.debugMode ~= 'trajectory' then
    trajecRec = {last = 0}
  end
  if M.debugMode ~= 'route' then
    routeRec = {last = 0}
  end
  if M.debugMode ~= 'speeds' then
    speedRecordings = {}
  end
  if M.debugMode ~= 'off' then
    M.debugDraw = debugDraw
  else
    M.debugDraw = nop
  end
end

local function setState(newState)
  tableMerge(M, newState)
  setAggressionExternal(M.extAggression)
  setMode()
  setVehicleDebugMode(M)
  setTargetObjectID(M.targetObjectID)
end

local function setTarget(wp)
  enter_or_exit_function('enter', 'setTarget')
  process_line(749); M.manualTargetName = wp
  process_line(750); validateInput = validateUserInput
  process_line(751); wpList = {wp}
  enter_or_exit_function('exit', 'setTarget')
end

local function setPath(path)
  manualPath = path
  validateInput = validateUserInput
end

local function driveUsingPath(arg)
  --[[ At least one argument of either path or wpTargetList must be specified. All other arguments are optional.

  * path: A sequence of waypoint names that form a path by themselves to be followed in the order provided.
  * wpTargetList: Type: A sequence of waypoint names to be used as succesive targets ex. wpTargetList = {'wp1', 'wp2'}.
                  Between any two consequitive waypoints a shortest path route will be followed.

  -- Optional Arguments --
  * wpSpeeds: Type: (key/value pairs, key: "node_name", value: speed, number in m/s)
              Define target speeds for individual waypoints. The ai will try to meet this speed when at the given waypoint.
  * noOfLaps: Type: number. Default value: nil
              The number of laps if the path is a loop. If not defined, the ai will just follow the succesion of waypoints once.
  * routeSpeed: A speed in m/s. To be used in tandem with "routeSpeedMode".
                Type: number
  * routeSpeedMode: Values: 'limit': the ai will not go above the 'routeSpeed' defined by routeSpeed.
                            'set': the ai will try to always go at the speed defined by "routeSpeed".
  * driveInLane: Values: 'on' (anything else is considered off/inactive)
                 When 'on' the ai will keep on the correct side of the road on two way roads.
                 This also affects pathFinding in that when this option is active ai paths will traverse roads in the legal direction if posibble.
                 Default: inactive
  * aggression: Value: 0.3 - 1. The aggression value with which the ai will drive the route.
                At 1 the ai will drive at the limit of traction. A value of 0.3 would be considered normal every day driving, going shopping etc.
                Default: 0.3
  * avoidCars: Values: 'on' / 'off'.  When 'on' the ai will be aware of (avoid crashing into) other vehicles on the map. Default is 'off'
  * examples:
  ai.driveUsingPath{ wpTargetList = {'wp1', 'wp10'}, driveInLane = 'on', avoidCars = 'on', routeSpeed = 35, routeSpeedMode = 'limit', wpSpeeds = {wp1 = 10, wp2 = 40}, aggression = 0.3}
  In the above example the speeds set for wp1 and wp2 will take precedence over "routeSpeed" for the specified nodes.
  --]]

  if (arg.wpTargetList == nil and arg.path == nil and arg.script == nil) or (type(arg.wpTargetList) ~= 'table' and type(arg.path) ~= 'table' and type(arg.script) ~= 'table') or (arg.wpSpeeds ~= nil and type(arg.wpSpeeds) ~= 'table') or (arg.noOfLaps ~= nil and type(arg.noOfLaps) ~= 'number') or (arg.routeSpeed ~= nil and type(arg.routeSpeed) ~= 'number') or (arg.routeSpeedMode ~= nil and type(arg.routeSpeedMode) ~= 'string') or (arg.driveInLane ~= nil and type(arg.driveInLane) ~= 'string') or (arg.aggression ~= nil and type(arg.aggression) ~= 'number') then
    return
  end

  if arg.resetLearning then
    resetLearning()
  end

  setState({mode = 'manual'})

  noOfLaps = arg.noOfLaps and math.max(arg.noOfLaps, 1) or 1
  wpList = arg.wpTargetList
  manualPath = arg.path
  validateInput = validateUserInput
  avoidCars = arg.avoidCars or 'off'

  if noOfLaps > 1 and wpList[2] and wpList[1] == wpList[#wpList] then
    race = true
  end

  speedList = arg.wpSpeeds or {}
  setSpeed(arg.routeSpeed)
  setSpeedMode(arg.routeSpeedMode)

  driveInLane(arg.driveInLane)

  setAggressionExternal(arg.aggression)
  stateChanged()
end

local function spanMap(cutOffDrivability)
  M.cutOffDrivability = cutOffDrivability or 0
  setState({mode = 'span'})
  stateChanged()
end

local function setCutOffDrivability(drivability)
  enter_or_exit_function('enter', 'setCutOffDrivability')
  process_line(752); M.cutOffDrivability = drivability or 0
  process_line(753); stateChanged()
  enter_or_exit_function('exit', 'setCutOffDrivability')
end

local function onDeserialized(v)
  setState(v)
  stateChanged()
end

local function dumpCurrentRoute()
  dump(currentRoute)
end

local function startRecording()
  M.mode = 'script'
  scriptai = require("scriptai")
  scriptai.startRecording()
  M.updateGFX = scriptai.updateGFX
end

local function stopRecording()
  M.mode = 'disabled'
  scriptai = require("scriptai")
  local script = scriptai.stopRecording()
  M.updateGFX = scriptai.updateGFX
  return script
end

local function startFollowing(...)
  M.mode = 'script'
  scriptai = require("scriptai")
  scriptai.startFollowing(...)
  M.updateGFX = scriptai.updateGFX
end

local function scriptStop(...)
  M.mode = 'disabled'
  scriptai = require("scriptai")
  scriptai.scriptStop(...)
  M.updateGFX = scriptai.updateGFX
end

local function scriptState()
  scriptai = require("scriptai")
  return scriptai.scriptState()
end

local function setScriptDebugMode(mode)
  scriptai = require("scriptai")
  if mode == nil or mode == 'off' then
    M.debugMode = 'all'
    M.debugDraw = nop
    return
  end

  M.debugDraw = debugDraw
  scriptai.debugMode = mode
end

local function isDriving()
  enter_or_exit_function('enter', 'isDriving')
  enter_or_exit_function('exit', 'isDriving')
  process_line(754); return M.updateGFX == updateGFX or (scriptai ~= nil and scriptai.isDriving())
end

-- additional public interface
M.getLineCoverageArray = getLineCoverageArray
M.getBranchCoverageArray = getBranchCoverageArray
M.resetCoverageArrays = resetCoverageArrays
M.startCoverage = startCoverage
M.getPathCoverageArray = getPathCoverageArray
-- public interface
M.driveInLane = driveInLane
M.stateChanged = stateChanged
M.reset = reset
M.setMode = setMode
M.setAvoidCars = setAvoidCars
M.setTarget = setTarget
M.setPath = setPath
M.setSpeed = setSpeed
M.setSpeedMode = setSpeedMode
M.setVehicleDebugMode = setVehicleDebugMode
M.setState = setState
M.getState = getState
M.debugDraw = nop
M.driveUsingPath = driveUsingPath
M.setAggressionMode = setAggressionMode
M.setAggression = setAggressionExternal
M.onDeserialized = onDeserialized
M.setTargetObjectID = setTargetObjectID
M.dumpCurrentRoute = dumpCurrentRoute
M.spanMap = spanMap
M.setCutOffDrivability = setCutOffDrivability
M.resetLearning = resetLearning
M.isDriving = isDriving

-- scriptai
M.startRecording = startRecording
M.stopRecording = stopRecording
M.startFollowing = startFollowing
M.stopFollowing = scriptStop
M.scriptStop = scriptStop
M.scriptState = scriptState
M.setScriptDebugMode = setScriptDebugMode
return M
