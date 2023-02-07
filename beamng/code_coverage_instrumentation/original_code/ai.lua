-- This Source Code Form is subject to the terms of the bCDDL, v. 1.1.
-- If a copy of the bCDDL was not distributed with this
-- file, You can obtain one at http://beamng.com/bCDDL-1.1.txt

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
  return M
end

local function stateChanged()
  if playerInfo.anyPlayerSeated then
    guihooks.trigger("AIStateChange", getState())
  end
end

local function setSpeed(speed)
  if type(speed) ~= 'number' then
    M.routeSpeed = nil
  else 
    M.routeSpeed = speed
  end
end

local function setSpeedMode(speedMode)
  if speedMode == 'set' or speedMode == 'limit' or speedMode == 'legal' or speedMode == 'off' then
    M.speedMode = speedMode
  else
    M.speedMode = nil
  end
end

local function resetSpeedModeAndValue()
  M.speedMode = nil -- maybe this should be 'off'
  M.routeSpeed = nil
end

local function setAggressionInternal(v)
  aggression = v and v or M.extAggression
end

local function setAggressionExternal(v)
  M.extAggression = v or M.extAggression
  setAggressionInternal()
  stateChanged()
end

local function setAggressionMode(aggrmode)
  if aggrmode == 'rubberBand' or aggrmode == 'manual' then
    aggressionMode = aggrmode
  else
    aggressionMode = nil
  end
end

local function resetAggression()
  setAggressionInternal()
end

local function setTargetObjectID(id)
  M.targetObjectID = M.targetObjectID ~= objectId and id or -1
  if M.targetObjectID ~= -1 then
    targetObjectSelectionMode = 'manual'
  end
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
  input.event("steering", steering, 1)
  input.event("throttle", throttle, 2)
  input.event("brake", brake, 2)
  input.event("parkingbrake", parkingbrake, 2)

  lastCommand.steering = steering
  lastCommand.throttle = throttle
  lastCommand.brake = brake
  lastCommand.parkingbrake = parkingbrake
end

local function driveToTarget(targetPos, throttle, brake, targetSpeed)
  if not targetPos then
    return
  end

  local brakeCoef = 1
  local throttleCoef = 1

  local targetVec = (targetPos - aiPos):normalized()

  local dirAngle = math.asin(ai.rightVec:dot(targetVec))
  local dirVel = aiVel:dot(aiDirVec)
  local absAiSpeed = math.abs(dirVel)
  local plan = currentRoute and currentRoute.plan
  targetSpeed = targetSpeed or plan and plan.targetSpeed
  if not targetSpeed then
    return
  end

  -- oversteer
  if aiSpeed > 1 then
    local rightVel = ai.rightVec:dot(aiVel)
    if rightVel * ai.rightVec:dot(targetPos - aiPos) > 0 then
      local rotVel = math.min(1, (ai.prevDirVec:projectToOriginPlane(ai.upVec):normalized()):distance(aiDirVec) * dt * 10000)
      throttleCoef = throttleCoef * math.max(0, 1 - math.abs(rightVel * aiSpeed * 0.05) * math.min(1, dirAngle * dirAngle * aiSpeed * 6) * rotVel)
    end
  end

  if plan and plan[3] and dirVel > 3 then
    local p1, p2 = plan[1].pos, plan[2].pos
    local p2p1 = p2 - p1
    local turnRight = p2p1:cross(ai.upVec):normalized()
    local tp2
    if plan.targetSeg and plan.targetSeg > 1 then
      tp2 = (targetPos - p2):normalized():dot(turnRight)
    else
      tp2 = (plan[3].pos - p2):normalized():dot(turnRight)
    end

    local outDeviation = aiDeviationSmoother:value() - aiDeviation * sign(tp2)
    outDeviation = sign(outDeviation) * math.min(1, math.abs(outDeviation))
    aiDeviationSmoother:set(outDeviation)
    aiDeviationSmoother:getUncapped(0, dt)

    if outDeviation > 0 and absAiSpeed > 3 then
      local steerCoef = outDeviation * absAiSpeed * absAiSpeed * math.min(1, dirAngle * dirAngle * 4)
      local understeerCoef = math.max(0, steerCoef) * math.min(1, math.abs(aiVel:dot(p2p1:normalized()) * 3))
      local noUndersteerCoef = math.max(0, 1 - understeerCoef)
      throttleCoef = throttleCoef * noUndersteerCoef
      brakeCoef = math.min(brakeCoef, math.max(0, 1 - understeerCoef * understeerCoef))
    end
  else
    aiDeviationSmoother:set(0)
  end

  -- wheel speed
  if absAiSpeed > 0.05 then
    if sensors.gz <= 0 then
      local totalSlip = 0
      local totalDownForce = 0
      local propSlip = 0
      local propDownForce = 0
      local lwheels = wheels.wheels
      for i = 0, tableSizeC(lwheels) - 1 do
        local wd = lwheels[i]
        if not wd.isBroken then
          local lastSlip = wd.lastSlip
          totalSlip = totalSlip + lastSlip
          if wd.isPropulsed then
            propSlip = math.max(propSlip, lastSlip)
          end
        end
      end

      -- math.abs
      brakeCoef = brakeCoef * square(math.max(0, absAiSpeed - totalSlip) / absAiSpeed)

      -- tcs
      local tcsCoef = math.max(0, absAiSpeed - propSlip * propSlip) / absAiSpeed
      throttleCoef = throttleCoef * math.min(tcsCoef, smoothTcs:get(tcsCoef, dt))
    else
      brakeCoef = 0
      throttleCoef = 0
    end
  end

  local dirTarget = aiDirVec:dot(targetVec)
  local dirDiff = math.asin(aiDirVec:cross(ai.upVec):normalized():dot(targetVec))

  if crash.manoeuvre == 1 and dirTarget < aiDirVec:dot(crash.dir) then
    driveCar(-fsign(dirDiff), brake * brakeCoef, throttle * throttleCoef, 0)
    return
  else
    crash.manoeuvre = 0
  end

  aiForceGoFrontTime = math.max(0, aiForceGoFrontTime - dt)
  if threewayturn.state == 1 and aiCannotMoveTime > 1 and aiForceGoFrontTime == 0 then
    threewayturn.state = 0
    aiCannotMoveTime = 0
    aiForceGoFrontTime = 2
  end

  if aiForceGoFrontTime > 0 and dirTarget < 0 then
    dirTarget = -dirTarget
    dirDiff = -dirDiff
  end

  if (dirTarget < 0 or (dirTarget < 0.15 and threewayturn.state == 1)) and currentRoute and not damageFlag then
    local n1, n2, n3 = plan[1], plan[2], plan[3]
    local edgeDist = math.min((n2 or n1).radiusOrig, n1.radiusOrig) - aiPos:z0():distanceToLine((n3 or n2).posOrig:z0(), n2.posOrig:z0())
    if edgeDist > ai.width and threewayturn.state == 0 then
      driveCar(fsign(dirDiff), 0.5 * throttleCoef, 0, math.min(math.max(aiSpeed - 3, 0), 1))
    else
      if threewayturn.state == 0 then
        threewayturn.state = 1
        threewayturn.speedDifInt = 0
      end
      local angleModulation = math.min(math.max(0, -(dirTarget-0.15)), 1)
      local speedDif = (10 * aggression * angleModulation) - aiSpeed
      threewayturn.speedDifInt = threewayturn.speedDifInt + speedDif * dt
      local pbrake = clamp(sign2(aiDirVec:dot(gravityDir) - 0.17), 0, 1) -- apply parking brake if reversing on an incline >~ 10 deg
      driveCar(-sign2(dirDiff), 0, clamp(0.05 * speedDif + 0.01 * threewayturn.speedDifInt, 0, 1), pbrake)
    end
  else
    threewayturn.state = 0
    local pbrake
    if aiVel:dot(aiDirVec) * math.max(aiSpeed*aiSpeed - 1e-2, 0) < 0 then
      if aiSpeed < 0.15 and targetSpeed <= 1e-5 then
        pbrake = 1
      else
        pbrake = 0
      end
      driveCar(dirDiff, 0.5 * throttleCoef, 0, pbrake)
    else
      if (aiSpeed > 4 and aiSpeed < 30 and math.abs(dirDiff) > 0.97 and brake == 0) or (aiSpeed < 0.15 and targetSpeed <= 1e-5) then
        pbrake = 1
      else
        pbrake = 0
      end
      driveCar(dirDiff, throttle * throttleCoef, brake * brakeCoef, pbrake)
    end
  end
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
  local aiSeg = 1
  local aiXnormOnSeg = 0
  for i = 1, planCount-1 do
    local p0Pos, p1Pos = plan[i].pos, plan[i+1].pos
    local xnorm = aiPos:xnormOnLine(p0Pos, p1Pos)
    if xnorm < 1 then
      if i < planCount - 2 then
        local nextXnorm = aiPos:xnormOnLine(p1Pos, plan[i+2].pos)
        if nextXnorm >= 0 then
          local p1Radius = plan[i+1].radius
          if aiPos:squaredDistance(linePointFromXnorm(p1Pos, plan[i+2].pos, nextXnorm)) < square(ai.width + lerp(p1Radius, plan[i+2].radius, math.min(1, nextXnorm))) then
            aiXnormOnSeg = nextXnorm
            aiSeg = i + 1
            break
          end
        end
      end
      aiXnormOnSeg = xnorm
      aiSeg = i
      break
    end
  end

  for _ = 1, aiSeg-1 do
    table.remove(plan, 1)
    planCount = planCount-1
  end

  return aiXnormOnSeg, planCount
end

local function calculateTarget(plan, planCount)
  planCount = planCount or #plan
  local aiXnormOnSeg
  aiXnormOnSeg, planCount = aiPosOnPlan(plan, planCount)
  local targetLength = math.max(aiSpeed * 0.65, 4.5)

  if planCount >= 3 then
    local xnorm = clamp(aiXnormOnSeg, 0, 1)
    local p2Pos = plan[2].pos
    targetLength = math.max(targetLength, plan[1].pos:distance(p2Pos) * (1-xnorm), p2Pos:distance(plan[3].pos) * xnorm)
  end

  local remainder = targetLength

  local targetPos = vec3(plan[planCount].pos)
  local targetSeg = math.max(1, planCount-1)
  local prevPos = linePointFromXnorm(plan[1].pos, plan[2].pos, aiXnormOnSeg) -- aiPos

  local segVec = vec3()
  for i = 2, planCount do
    local pos = plan[i].pos
    segVec:set(pos)
    segVec:setSub(prevPos)
    local segLen = segVec:length()

    if remainder <= segLen then
      targetSeg = i - 1
      targetPos:set(segVec)
      targetPos:setScaled(remainder / (segLen + 1e-25))
      targetPos:setAdd(prevPos)

      -- smooth target
      local xnorm = clamp(targetPos:xnormOnLine(prevPos, pos), 0, 1)
      local lp_n1n2 = linePointFromXnorm(prevPos, pos, xnorm * 0.5 + 0.25)
      if xnorm <= 0.5 then
        if i >= 3 then
          targetPos = linePointFromXnorm(linePointFromXnorm(plan[i-2].pos, prevPos, xnorm * 0.5 + 0.75), lp_n1n2, xnorm + 0.5)
        end
      else
        if i <= planCount - 2 then
          targetPos = linePointFromXnorm(lp_n1n2, linePointFromXnorm(pos, plan[i+1].pos, xnorm * 0.5 - 0.25), xnorm - 0.5)
        end
      end
      break
    end
    prevPos = pos
    remainder = remainder - segLen
  end

  return targetPos, targetSeg, aiXnormOnSeg, planCount
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
  -- given current speed, distance required to come to a stop if I can decelerate at 0.2g
  limLow = limLow or 150
  speed = speed or aiSpeed
  accelg = math.max(0.2, accelg or 0.2)
  return math.min(550, math.max(limLow, 0.5 * speed * speed / (accelg * g)))
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
  if newPath == nil then
    return
  end
  local pathCount = #path
  if path[pathCount] ~= newPath[1] then
    return
  end
  pathCount = pathCount - 1
  for i = 2, #newPath do
    path[pathCount+i] = newPath[i]
  end
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
  local vec1Sqlen, vec2Sqlen = vec1:squaredLength(), vec2:squaredLength()
  local dot12 = vec1:dot(vec2)
  local cos8sq = math.min(1, dot12 * dot12 / math.max(1e-30, vec1Sqlen * vec2Sqlen))

  if dot12 < 0 then
    local minDsq = math.min(vec1Sqlen, vec2Sqlen)
    local maxDsq = minDsq / math.max(1e-30, cos8sq)
    if math.max(vec1Sqlen, vec2Sqlen) > (minDsq + maxDsq) * 0.5 then
      if vec1Sqlen > vec2Sqlen then
        vec1, vec2 = vec2, vec1
        vec1Sqlen, vec2Sqlen = vec2Sqlen, vec1Sqlen
      end
      vec2 = math.sqrt(0.5 * (minDsq + maxDsq) / math.max(1e-30, vec2Sqlen)) * vec2
    end
  end

  return 2 * math.sqrt((1 - cos8sq) / math.max(1e-30, (vec1 + vec2):squaredLength()))
end

local function getPathLen(path, startIdx, stopIdx)
  if not path then
    return
  end
  startIdx = startIdx or 1
  stopIdx = stopIdx or #path
  local positions = mapData.positions
  local pathLen = 0
  for i = startIdx+1, stopIdx do
    pathLen = pathLen + positions[path[i-1]]:distance(positions[path[i]])
  end

  return pathLen
end

local function waypointInPath(path, waypoint, startIdx, stopIdx)
  if not path or not waypoint then
    return
  end
  startIdx = startIdx or 1
  stopIdx = stopIdx or #path
  for i = startIdx, stopIdx do
    if path[i] == waypoint then
      return i
    end
  end
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
  local planLen, planCount = 0, #plan
  for i = 2, planCount do
    planLen = planLen + plan[i].pos:distance(plan[i-1].pos)
  end
  return planLen, planCount
end

local function buildNextRoute(plan, planCount, path)
  local nextPathIdx = (plan[planCount].pathidx or 0) + 1 -- if the first plan node is the aiPos it does not have a pathIdx value yet

  if race == true and noOfLaps and noOfLaps > 1 and not path[nextPathIdx] then -- in case the path loops
    local loopPathId
    local pathCount = #path
    local lastWayPoint = path[pathCount]
    for i = 1, pathCount do
      if lastWayPoint == path[i] then
        loopPathId = i
        break
      end
    end
    nextPathIdx = 1 + loopPathId -- nextPathIdx % #path
    noOfLaps = noOfLaps - 1
  end

  local nextNodeName = path[nextPathIdx]
  if not nextNodeName then
    return
  end

  local n1Pos, oneWay
  local n2 = mapData.graph[nextNodeName]
  if not n2 then
    return
  end
  local n2Pos = mapData.positions[nextNodeName]
  local n2Radius = mapData.radius[nextNodeName]
  local validOneWayLink = true -- test for if oneWay road does not merge into a not oneWay road

  if path[nextPathIdx-1] then
    n1Pos = mapData.positions[path[nextPathIdx-1]]
    local link = mapData.graph[path[nextPathIdx-1]][path[nextPathIdx]]
    local legalSpeed
    if link then
      oneWay = link[3]
      if oneWay then
        for k, v in pairs(mapData.graph[path[nextPathIdx]]) do
          if k ~= path[nextPathIdx - 1] and not v[3] and ((n2Pos - n1Pos):normalized()):dot((mapData.positions[k] - n2Pos):normalized()) > 0.2 then
            validOneWayLink = false
          end
        end
      end

      if nextPathIdx > 2 then
        legalSpeed = math.min(link[4], mapData.graph[path[nextPathIdx-1]][path[nextPathIdx-2]][4])
      else
        legalSpeed = link[4]
      end
    end
    plan[planCount].legalSpeed = legalSpeed
  elseif path[nextPathIdx+1] then
    n1Pos = vec3(aiPos)
    local link = mapData.graph[path[nextPathIdx]][path[nextPathIdx+1]]
    oneWay = link and link[3] -- why do we need the link check here?
  end

  if driveInLaneFlag then
    local lane = 1
    plan.lane = plan.lane or 0
    if oneWay and validOneWayLink then
      if plan.lane ~= 0 then
        lane = plan.lane
      else
        if path[2] then
          local curPathIdx = (plan[1] and plan[1].pathidx) and math.max(2, plan[1].pathidx) or 2
          local p1Pos = mapData.positions[path[curPathIdx-1]]
          lane = sign((mapData.positions[path[curPathIdx]] - p1Pos):z0():cross(gravityDir):dot(p1Pos - aiPos))
          plan.lane = lane
        end
      end
    else
      plan.lane = 0
    end

    local nVec1
    if path[nextPathIdx-1] then
      nVec1 = (n1Pos - n2Pos):z0():cross(gravityDir):normalized()
    else
      nVec1 = vec3()
    end

    local nVec2
    if path[nextPathIdx+1] then
      nVec2 = (n2Pos - mapData.positions[path[nextPathIdx+1]]):z0():cross(gravityDir):normalized()
    else
      nVec2 = vec3()
    end

    local width = math.max(n2Radius * 0.5, ai.width * 0.7)
    local displacement = math.max(0, n2Radius - width) -- provide a bit more space in narrow roads so other vehicles can overtake
    n2Pos = n2Pos + displacement * lane * (1 - nVec1:dot(nVec2) * 0.5) * (nVec1 + nVec2)
    n2Radius = width
  end

  local lastPlanPos = plan[planCount] and plan[planCount].pos or aiPos
  local vec = (lastPlanPos - n2Pos):z0()
  local manSpeed = speedList and speedList[nextNodeName]

  return {pos = vec3(n2Pos), posOrig = vec3(n2Pos), radius = n2Radius, radiusOrig = n2Radius,  posz0 = n2Pos:z0(), vec = vec, dirVec = vec:normalized(), turnDir = vec3(0,0,0), manSpeed = manSpeed, pathidx = nextPathIdx}
end

local function mergePathPrefix(source, dest, srcStart)
  srcStart = srcStart or 1
  local sourceCount = #source
  local dict = table.new(0, sourceCount-(srcStart-1))
  for i = srcStart, sourceCount do
    dict[source[i]] = i
  end

  local destCount = #dest
  for i = destCount, 1, -1 do
    local srci = dict[dest[i]]
    if srci ~= nil then
      local res = table.new(destCount, 0)
      local resi = 1
      for i1 = srcStart, srci - 1 do
        res[resi] = source[i1]
        resi = resi + 1
      end
      for i1 = i, destCount do
        res[resi] = dest[i1]
        resi = resi + 1
      end

      return res, srci
    end
  end

  return dest, 0
end

local function planAhead(route, baseRoute)
  if route == nil then
    return
  end
  if route.path == nil then
    route.path = {}
    for i = 1, #route do
      route.path[i] = route[i]
      route[i] = nil
    end
    route.plan = {}
  end

  local plan = route.plan

  if baseRoute and not plan[1] then
    -- merge from base plan
    local bsrPlan = baseRoute.plan
    if bsrPlan[2] then
      local commonPathEnd
      route.path, commonPathEnd = mergePathPrefix(baseRoute.path, route.path, bsrPlan[2].pathidx)
      if commonPathEnd >= 1 then
        local refpathidx = bsrPlan[2].pathidx - 1
        for i = 1, #bsrPlan do
          local n = bsrPlan[i]
          if n.pathidx > commonPathEnd then
            break
          end
          plan[i] = {pos = vec3(n.pos), posOrig = vec3(n.posOrig), posz0 = vec3(n.posz0), vec = vec3(n.vec), dirVec = vec3(n.dirVec), turnDir = vec3(n.turnDir),  radius = n.radius, radiusOrig = n.radiusOrig, pathidx = math.max(1, n.pathidx-refpathidx), legalSpeed = n.legalSpeed}
        end
        if plan[bsrPlan.targetSeg+1] then
          plan.targetSeg = bsrPlan.targetSeg
          plan.targetPos = vec3(bsrPlan.targetPos)
          plan.aiSeg = bsrPlan.aiSeg
        end
      end
    end
  end

  if not plan[1] then
    plan[1] = {posOrig = vec3(aiPos), pos = vec3(aiPos), posz0 = aiPos:z0(), vec = (-8) * aiDirVec,  dirVec = -aiDirVec, turnDir = vec3(0,0,0), radiusOrig = 2, radius = 2}
  end

  local planLen, planCount = getPlanLen(plan)
  local minPlanLen = getMinPlanLen()
  while not plan[minPlanCount] or planLen < minPlanLen do
    local n = buildNextRoute(plan, planCount, route.path)
    if not n then
      break
    end
    planCount = planCount + 1
    plan[planCount] = n
    planLen = planLen + n.pos:distance(plan[planCount-1].pos)
  end

  if not plan[2] then
    return
  end
  if not plan[1].pathidx then
    plan[1].pathidx = plan[2].pathidx
  end

  do
    local segmentSplitDelay = plan.segmentSplitDelay or 0
    local distOnPlan = 0
    for i = 1, planCount-1 do
      local curDist = plan[i].posOrig:squaredDistance(plan[i+1].posOrig)
      local xSq = square(distOnPlan)
      if curDist > square(math.min(220, (25e-8 * xSq + 1e-5) * xSq + 6)) and distOnPlan < 550 then
        if segmentSplitDelay == 0 then
          local n1, n2 = plan[i], plan[i+1]
          local pos = (n1.pos + n2.pos) * 0.5
          local vec = (n1.pos - pos):z0()
          n2.vec = (pos - n2.pos):z0()
          n2.dirVec = n2.vec:normalized()
          local legalSpeed
          if n2.pathidx > 1 then
            legalSpeed = mapData.graph[route.path[n2.pathidx]][route.path[n2.pathidx-1]][4]
          else
            legalSpeed = n2.legalSpeed
          end
          table.insert(plan, i+1, {posOrig = (n1.posOrig + n2.posOrig) * 0.5, pos = pos, posz0 = pos:z0(),  vec = vec, dirVec = vec:normalized(), turnDir = vec3(0, 0, 0), radiusOrig = (n1.radiusOrig + n2.radiusOrig) * 0.5, radius = (n1.radius + n2.radius) * 0.5, pathidx = n2.pathidx, legalSpeed = legalSpeed})
          planCount = planCount + 1
          segmentSplitDelay = math.min(5, math.floor(90/aiSpeed))
        else
          segmentSplitDelay = segmentSplitDelay - 1
        end
        break
      end
      distOnPlan = distOnPlan + math.sqrt(curDist)
    end
    plan.segmentSplitDelay = segmentSplitDelay
  end

  if plan.targetSeg == nil then
    local aiXnormOnSeg
    plan.targetPos, plan.targetSeg, aiXnormOnSeg, planCount = calculateTarget(plan, planCount)
    plan.aiSeg = 1
  end

  for i = 0, planCount do
    if forces[i] then
      forces[i]:set(0,0,0)
    else
      forces[i] = vec3(0,0,0)
    end
  end

  -- calculate spring forces
  for i = 1, planCount-1 do
    local n1 = plan[i]
    local n2 = plan[i+1]
    local v1 = n1.dirVec
    local v2 = -n2.dirVec
    local turnDir = (v1 + v2):normalized()

    local nforce = (1-threewayturn.state) * math.max(1 + v1:dot(v2), 0) * 2 * turnDir
    forces[i+1]:setSub(nforce)
    forces[i-1]:setSub(nforce)
    nforce:setScaled(2)
    forces[i]:setAdd(nforce)

    n1.turnDir:set(turnDir)
    n1.speed = 0
  end

  if M.mode == 'traffic' then
    doTrafficActions(route.path, plan)
  end

  -- other vehicle awareness
  plan.trafficMinProjSpeed = math.huge
  plan.trafficBlockCoef = nil
  if avoidCars ~= 'off' then
    table.clear(trafficTable)
    local trafficTableLen = 0
    for plID, v in pairs(mapmgr.objects) do
      if plID ~= objectId and (M.mode ~= 'chase' or plID ~= player.id or chaseData.suspectState == 'stopped') then
        v.length = obj:getObjectInitialLength(plID) + 0.3
        v.width = obj:getObjectInitialWidth(plID)
        v.targetType = (player and plID == player.id) and M.mode
        if v.targetType == 'follow' then
          v.width = v.width * 4
        end
        local posFront = obj:getObjectFrontPosition(plID)
        local dirVec = v.dirVec
        v.posFront = dirVec * 0.3 + posFront
        v.posRear = dirVec * (-v.length) + posFront
        v.lightbar = ((v.states and v.states.lightbar) and v.states.lightbar > 0) and true or false
        table.insert(trafficTable, v)
        trafficTableLen = trafficTableLen + 1
      end
    end

    local trafficMinSpeedSq = math.huge
    local distanceT = 0
    local aiPathVel = aiVel:dot((plan[2].pos-plan[1].pos):normalized())
    local aiPathVelInv = 1 / math.abs(aiPathVel + 1e-30)
    local minTrafficDir = 1

    for i = 2, planCount-1 do
      local n1, n2 = plan[i], plan[i+1]
      local n1pos, n2pos = n1.pos, n2.pos
      local n1n2 = n2pos - n1pos
      local n1n2len = n1n2:length()
      local nDir = n1n2 / (n1n2len + 1e-30)
      n1.trafficSqVel = math.huge
      local arrivalT = distanceT * aiPathVelInv

      if damageFlag or (intersection.planStop and intersection.planStop < i) then
        n1.trafficSqVel = 0
      else
        for j = trafficTableLen, 1, -1 do
          local v = trafficTable[j]
          local plPosFront, plPosRear, plWidth = v.posFront, v.posRear, v.width
          local ai2PlVec = plPosFront - aiPos
          local ai2PlDir = ai2PlVec:dot(aiDirVec)
          local ai2PlSqDist = ai2PlVec:squaredLength()
          if ai2PlDir > 0 then
            local velDisp = arrivalT * v.vel
            plPosFront = plPosFront + velDisp
            plPosRear = plPosRear + velDisp
          end
          local extVec = nDir * (math.max(ai.width, plWidth) * 0.5)
          local n1ext, n2ext = n1pos - extVec, n2pos + extVec
          local rnorm, vnorm = closestLinePoints(n1ext, n2ext, plPosFront, plPosRear)

          if M.mode == 'traffic' and v.lightbar and ai2PlSqDist <= 10000 then -- lightbar awareness
            local tmpVec = ai.rightVec * 2
            forces[i]:setAdd(tmpVec)
            forces[i + 1]:setAdd(tmpVec)
            trafficSide.cTimer = math.max(5, trafficSide.cTimer)
            n1.trafficSqVel = clamp(math.sqrt(ai2PlSqDist) * 2 - 25, 0, 200)
          end

          local minSqDist = math.huge
          if rnorm > 0 and rnorm < 1 and vnorm > 0 and vnorm < 1 then
            minSqDist = 0
          else
            local rlen = n1n2len + plWidth
            local xnorm = plPosFront:xnormOnLine(n1ext, n2ext) * rlen
            if xnorm > 0 and xnorm < rlen then
              minSqDist = math.min(minSqDist, (n1ext + nDir * xnorm):squaredDistance(plPosFront))
            end

            xnorm = plPosRear:xnormOnLine(n1ext, n2ext) * rlen
            if xnorm > 0 and xnorm < rlen then
              minSqDist = math.min(minSqDist, (n1ext + nDir * xnorm):squaredDistance(plPosRear))
            end

            rlen = v.length + ai.width
            local v1 = vec3(n1ext)
            v1:setSub(plPosRear)
            local v1dot = v1:dot(v.dirVec)
            if v1dot > 0 and v1dot < rlen then
              minSqDist = math.min(minSqDist, v1:squaredDistance(v1dot * v.dirVec))
            end

            v1:set(n2ext)
            v1:setSub(plPosRear)
            v1dot = v1:dot(v.dirVec)
            if v1dot > 0 and v1dot < rlen then
              minSqDist = math.min(minSqDist, v1:squaredDistance(v1dot * v.dirVec))
            end
          end

          if minSqDist < square((ai.width + plWidth) * 0.8) then
            local velProjOnSeg = math.max(0, v.vel:dot(nDir))
            local middlePos = (plPosFront + plPosRear) * 0.5
            local forceCoef = trafficSide.side * 0.5 * math.max(0, math.max(aiSpeed - velProjOnSeg, sign(-(nDir:dot(v.dirVec))) * trafficSide.cTimer)) / ((1 + minSqDist) * (1 + distanceT * math.min(0.1, 1 / (2 * math.max(0, aiPathVel - v.vel:dot(nDir)) + 1e-30))))

            if intersection.planStop or v.targetType ~= 'follow' then
              forces[i]:setSub((sign(n1.turnDir:dot(middlePos - n1.posOrig)) * forceCoef) * n1.turnDir)
              forces[i+1]:setSub((sign(n2.turnDir:dot(middlePos - n2.posOrig)) * forceCoef) * n2.turnDir)
            end

            if avoidCars == 'on' and M.mode ~= 'flee' and M.mode ~= 'random' then
              if minSqDist < square((ai.width + plWidth) * 0.51)  then
                -- obj.debugDrawProxy:drawSphere(0.25, v.posFront:toFloat3(), color(0,0,255,255))
                -- obj.debugDrawProxy:drawSphere(0.25, plPosFront:toFloat3(), color(0,0,255,255))
                table.remove(trafficTable, j)
                trafficTableLen = trafficTableLen - 1
                plan.trafficMinProjSpeed = math.min(plan.trafficMinProjSpeed, velProjOnSeg)

                n1.trafficSqVel = math.min(n1.trafficSqVel, velProjOnSeg * velProjOnSeg)
                trafficMinSpeedSq = math.min(trafficMinSpeedSq, v.vel:squaredLength())
                minTrafficDir = math.min(minTrafficDir, v.dirVec:dot(nDir))
              end

              if i == 2 and minSqDist < square((ai.width + plWidth) * 0.6) and ai2PlDir > 0 and v.vel:dot(ai.rightVec) * ai2PlVec:dot(ai.rightVec) < 0 then
                n1.trafficSqVel = math.max(0, n1.trafficSqVel - math.abs(1 - v.vel:dot(aiDirVec)) * (v.vel:length()))
              end
            end
          end
        end
      end
      distanceT = distanceT + n1n2len
    end

    if math.max(trafficMinSpeedSq, aiSpeed*aiSpeed) < 0.25 then
      plan.trafficBlockCoef = clamp((1 - minTrafficDir) * 0.5, 0, 1)
    else
      plan.trafficBlockCoef = 0
    end
    plan[1].trafficSqVel = plan[2].trafficSqVel
  end

  -- spring force integrator
  for i = 2, planCount do
    local n = plan[i]
    local k = n.turnDir:dot(forces[i])
    local nodeDisplVec = n.pos + fsign(k) * math.min(math.abs(k), 0.5) * n.turnDir - n.posOrig
    local nodeDisplLen = nodeDisplVec:length()
    local maxDispl = math.max(0, n.radiusOrig - ai.width * (0.35 + 0.3 / (1 + trafficSide.cTimer * 0.1))) * math.min(1, aggression * (1 + trafficSide.cTimer * 0.3))
    local nodeDisplLenLim = clamp(nodeDisplLen, 0, maxDispl)
    --n.radius = math.max(n.radiusOrig - fsign(nodeDisplVec:dot(n.turnDir)) * nodeDisplLenLim - distFromEdge, distFromEdge)
    n.radius = math.max(0, n.radiusOrig - nodeDisplLenLim)
    n.pos = n.posOrig + (nodeDisplLenLim / (nodeDisplLen + 1e-30)) * nodeDisplVec
    n.posz0:set(n.pos)
    n.posz0.z = 0
    n.vec = plan[i-1].posz0 - n.posz0
    n.dirVec:set(n.vec)
    n.dirVec:normalize()
  end

  local targetSeg = plan.targetSeg
  -- smoothly distribute error from planline onto the front segments
  if targetSeg ~= nil and planCount > targetSeg and plan.targetPos and threewayturn.state == 0 then
    local dTotal = 0
    local sumLen = {}
    for i = 2, targetSeg - 1  do
      sumLen[i] = dTotal
      dTotal = dTotal + plan[i+1].pos:distance(plan[i].pos)
    end
    dTotal = dTotal + plan.targetPos:distance(plan[targetSeg].pos)

    dTotal = math.max(1, dTotal)
    local p1, p2 = plan[1].pos, plan[2].pos
    local dispVec = (aiPos - linePointFromXnorm(p1, p2, aiPos:xnormOnLine(p1, p2)))
    dispVec:setScaled(0.5 * dt)
    aiDeviation = dispVec:dot((p2-p1):cross(ai.upVec):normalized())

    local dispVecRatio = dispVec / dTotal
    for i = targetSeg - 1, 3, -1 do
      plan[i].pos:setAdd((dTotal - sumLen[i]) * dispVecRatio)
      plan[i].posz0 = plan[i].pos:z0()
      plan[i+1].vec = plan[i].posz0 - plan[i+1].posz0
      plan[i+1].dirVec = plan[i+1].vec:normalized()
    end

    plan[1].pos:setAdd(dispVec)
    plan[1].posz0 = plan[1].pos:z0()
    plan[2].pos:setAdd(dispVec)
    plan[2].posz0 = plan[2].pos:z0()
    if plan[3] then
      plan[3].vec = plan[2].posz0 - plan[3].posz0
      plan[3].dirVec = plan[3].vec:normalized()
    end
  end

  plan.targetPos, plan.targetSeg, plan.aiXnormOnSeg, planCount = calculateTarget(plan)
  plan.aiSeg = 1
  plan.planCount = planCount

  -- plan speeds
  local totalAccel = math.min(aggression, staticFrictionCoef) * g

  local rLast = plan[planCount]
  if route.path[rLast.pathidx+1] or (race and noOfLaps and noOfLaps > 1) then
    rLast.speed = rLast.manSpeed or math.sqrt(2 * 550 * totalAccel) -- shouldn't this be calculated based on the path length remaining?
  else
    rLast.speed = rLast.manSpeed or 0
  end

  -- speed planning
  local tmpEdgeVec = vec3()
  for i = planCount-1, 1, -1 do
    local n1 = plan[i]
    local n2 = plan[i+1]

    -- inclination calculation
    tmpEdgeVec:set(n2.pos) -- = n2.pos - n1.pos
    tmpEdgeVec:setSub(n1.pos)
    local dist = tmpEdgeVec:length() + 1e-30
    tmpEdgeVec:setScaled(1 / dist)

    local Gf = gravityVec:dot(tmpEdgeVec) -- acceleration due to gravity parallel to road segment, positive when downhill
    local Gt = gravityVec:distance(tmpEdgeVec * Gf) / g -- gravity vec normal to road segment

    local n2SpeedSq = square(n2.speed)

    local n0vec = plan[math.max(1, i-2)].posz0 - n1.posz0
    local n3vec = n1.posz0 - plan[math.min(planCount, i + 2)].posz0
    local curvature = math.min(math.min(inCurvature(n1.vec, n2.vec), inCurvature(n0vec, n3vec)), math.min(inCurvature(n0vec, n2.vec), inCurvature(n1.vec, n3vec))) + 1.6e-7

    local turnSpeedSq = totalAccel * Gt / curvature -- available centripetal acceleration * radius

    -- https://physics.stackexchange.com/questions/312569/non-uniform-circular-motion-velocity-optimization
    --local deltaPhi = 2 * math.asin(0.5 * n2.vec:length() * curvature) -- phi = phi2 - phi1 = 2 * math.asin(halfcord / radius)
    local n1SpeedSq = turnSpeedSq * math.sin(math.min(math.asin(math.min(1, n2SpeedSq / turnSpeedSq)) + 2*curvature*dist, math.pi*0.5))

    n1SpeedSq = math.min(n1SpeedSq, n1.trafficSqVel or math.huge)
    n1.trafficSqVel = math.huge

    n1.speed = n1.manSpeed or (M.speedMode == 'legal' and n1.legalSpeed and math.min(n1.legalSpeed, math.sqrt(n1SpeedSq))) or (M.speedMode == 'limit' and M.routeSpeed and math.min(M.routeSpeed, math.sqrt(n1SpeedSq))) or (M.speedMode == 'set' and M.routeSpeed) or math.sqrt(n1SpeedSq)
  end

  plan.targetSpeed = plan[1].speed + math.max(0, plan.aiXnormOnSeg) * (plan[2].speed - plan[1].speed)

  return plan
end

local function resetMapAndRoute()
  mapData = nil
  signalsData = nil
  currentRoute = nil
  race = nil
  noOfLaps = nil
  damageFlag = false
  internalState = 'onroad'
  changePlanTimer = 0
  resetAggression()
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
  local positions = mapData.positions
  local radii = mapData.radius

  local wp1, wp2, dist1 = mapmgr.findClosestRoad(aiPos)
  if wp1 == nil or wp2 == nil then
    internalState = 'offroad'
    return
  end

  if aiDirVec:dot(positions[wp1] - positions[wp2]) > 0 then
    wp1, wp2 = wp2, wp1
  end

  local plwp1, plwp2, dist2 = mapmgr.findClosestRoad(player.pos)
  if plwp1 == nil or plwp2 == nil then
    internalState = 'offroad'
    return
  end

  if dist1 > math.max(radii[wp1], radii[wp2]) * 2 and dist2 > math.max(radii[plwp1], radii[plwp2]) * 2 then
    internalState = 'offroad'
    return
  end

  local playerNode = plwp2
  local playerSpeed = player.vel:length()
  local plDriveVel = playerSpeed > 1 and player.vel or player.dirVec
  if plDriveVel:dot(positions[plwp1] - positions[plwp2]) > 0 then
    plwp1, plwp2 = plwp2, plwp1
  end

  --chaseData.playerRoad = {plwp1, plwp2}

  local aiPlDist = aiPos:distance(player.pos) -- should this be a signed distance?
  local aggrValue = aggressionMode == 'manual' and M.extAggression or 0.9
  if M.mode == 'follow' then
    setAggressionInternal(math.min(aggrValue, 0.3 + 0.002 * aiPlDist))
  else
    setAggressionInternal(aggrValue) -- constant value for better chase experience?
  end

  -- consider calculating the aggression value but then passing it through a smoother so that transitions between chase mode and follow mode are smooth

  if playerSpeed < 1.5 then
    chaseData.suspectStoppedTimer = chaseData.suspectStoppedTimer + dt
  else
    chaseData.suspectStoppedTimer = 0
  end

  if chaseData.suspectStoppedTimer > 5 then
    chaseData.suspectState = 'stopped'
    if aiPlDist < 20 and aiSpeed < 0.3 then
      -- do not plan new route if stopped near player
      currentRoute = nil
      internalState = 'onroad'
      return
    end
  else
    chaseData.suspectState = nil
  end

  if M.mode == 'follow' and aiSpeed < 0.3 and (wp1 == playerNode or wp2 == playerNode) then
    currentRoute = nil
  end

  if currentRoute then
    local curPath = currentRoute.path
    local curPlan = currentRoute.plan
    local playerNodeInPath = waypointInPath(curPath, playerNode, curPlan[2].pathidx)
    local playerOtherWay = (player.pos - aiPos):dot(positions[wp2] - aiPos) < 0 and plDriveVel:dot(positions[wp2] - aiPos) < 0 -- player is moving other way from next ai wp
    local pathLen = getPathLen(curPath, playerNodeInPath or math.huge) -- curPlan[2].pathidx
    local playerMinPlanLen = getMinPlanLen(0, playerSpeed) -- math.min(curPlan[#curPlan].speed, curPlan[#curPlan-1].speed, aiSpeed)

    if not playerNodeInPath or playerOtherWay or (M.mode == 'chase' and pathLen < playerMinPlanLen) then
      local newRoute = mapData:getPath(wp2, playerNode, driveInLaneFlag and 1e3 or 1) -- maybe wp2 should be curPath[curPlan[2].pathidx]
      --pathLen = getPathLen(newRoute)

      if M.mode == 'chase' then -- and pathLen < playerMinPlanLen
        --local playerSpeed = player.vel:length() -- * 1.1
        if playerSpeed > 1 then -- is this needed?
          --local playerMinPlanLen = getMinPlanLen(0, playerSpeed, staticFrictionCoef * 0.5)
          pathExtend(newRoute, mapData:getFleePath(playerNode, plDriveVel, player.pos, playerMinPlanLen, 0, 0)) -- math.max(minPlanLen-pathLen, playerMinPlanLen) -- mapNodes[plwp1].pos - mapNodes[plwp2].pos
        end
      end

      local tempPlan = planAhead(newRoute, currentRoute)
      if tempPlan and tempPlan.targetSpeed >= math.min(aiSpeed, curPlan.targetSpeed) and (tempPlan.targetPos-curPlan.targetPos):dot(aiDirVec) >= 0 then
        currentRoute = newRoute
      else
        planAhead(currentRoute)
      end
    else
      planAhead(currentRoute)
    end

    --chaseData.playerSeg, chaseData.playerXnormOnSeg = posOnPlan(player.pos, currentRoute.plan)

    if M.mode == 'chase' and plPrevVel and (plwp2 == curPath[curPlan[2].pathidx] or plwp2 == curPath[curPlan[2].pathidx + 1]) then
    --aiPlDist < math.max(20, aiSpeed * 2.5)
      local playerNodePos1 = positions[plwp2]
      local segDir = (playerNodePos1 - positions[plwp1])
      local targetLineDir = vec3(-segDir.y, segDir.x, 0):normalized()
      local xnorm = closestLinePoints(playerNodePos1, playerNodePos1 + targetLineDir, player.pos, player.pos + player.dirVec)
      local tarPos = playerNodePos1 + targetLineDir * clamp(xnorm, -radii[plwp2], radii[plwp2])

      local p2Target = (tarPos - player.pos):normalized()
      local plVel2Target = playerSpeed > 0.1 and player.vel:dot(p2Target) or 0
      --local plAccel = (plVel2Target - plPrevVel:dot(p2Target)) / dt
      --plAccel = plAccel + sign2(plAccel) * 1e-5
      --local plTimeToTarget = (math.sqrt(math.max(plVel2Target * plVel2Target + 2 * plAccel * (tarPos - player.pos):length(), 0)) - plVel2Target) / plAccel
      local plTimeToTarget = tarPos:distance(player.pos) / (plVel2Target + 1e-30) -- accel maybe not needed; this gives smooth results

      local aiVel2Target = aiSpeed > 0.1 and aiVel:dot((tarPos - aiPos):normalized()) or 0
      local aiTimeToTarget = tarPos:distance(aiPos) / (aiVel2Target + 1e-30)

      if aiTimeToTarget < plTimeToTarget then
        internalState = 'tail'
        -- return
      else
        internalState = 'onroad'
      end
    else
      internalState = 'onroad'
    end
  else
    if M.mode == 'follow' and aiPlDist < 20 then
      -- do not plan new route if opponent is stopped and ai has reached opponent
      internalState = 'onroad'
      return
    end

    local newRoute = mapData:getPath(wp2, playerNode, driveInLaneFlag and 1e3 or 1)

    local tempPlan = planAhead(newRoute)
    if tempPlan then
      currentRoute = newRoute
    end
  end
end

local function warningAIDisabled(message)
  guihooks.message(message, 5, "AI debug")
  M.mode = 'disabled'
  M.updateGFX = nop
  resetMapAndRoute()
  stateChanged()
end

local function offRoadFollowControl()
  if not player or not player.pos or not aiPos or not aiSpeed then
    return 0, 0, 0
  end

  local ai2PlVec = player.pos - aiPos
  local ai2PlDist = ai2PlVec:length()
  local ai2PlDirVec = ai2PlVec / (ai2PlDist + 1e-30)
  local plSpeedFromAI = player.vel:dot(ai2PlDirVec)
  ai2PlDist = math.max(0, ai2PlDist - 12)
  local targetSpeed = math.sqrt(math.max(0, plSpeedFromAI*plSpeedFromAI*plSpeedFromAI / (math.abs(plSpeedFromAI) + 1e-30) + 2 * 9.81 * math.min(1, staticFrictionCoef) * ai2PlDist))
  local speedDif = targetSpeed - aiSpeed
  local throttle = clamp(speedDif, 0, 1)
  local brake = clamp(-speedDif, 0, 1)

  return throttle, brake, targetSpeed
end

M.updateGFX = nop
local function updateGFX(dtGFX)
  dt = dtGFX

  if mapData ~= mapmgr.mapData then
    currentRoute = nil
  end

  mapData = mapmgr.mapData
  signalsData = mapmgr.signalsData

  if mapData == nil then
    return
  end

  -- local cgPos = obj:calcCenterOfGravity()
  -- aiPos:set(cgPos)
  -- aiPos.z = obj:getSurfaceHeightBelow(cgPos)
  local tmpPos = obj:getFrontPosition()
  aiPos:set(tmpPos)
  aiPos.z = math.max(aiPos.z - 1, obj:getSurfaceHeightBelow(tmpPos))
  ai.prevDirVec:set(aiDirVec)
  aiDirVec:set(obj:getDirectionVector())
  ai.upVec:set(obj:getDirectionVectorUp())
  aiVel:set(obj:getVelocity())
  aiSpeed = aiVel:length()
  ai.width = ai.width or obj:getInitialWidth()
  ai.length = ai.length or obj:getInitialLength()
  ai.rightVec = aiDirVec:cross(ai.upVec):normalized()
  staticFrictionCoef = 0.95 * obj:getStaticFrictionCoef()

  if trafficBlock.coef > 0 then
    trafficBlock.timer = trafficBlock.timer + dt * trafficBlock.coef
  else
    trafficBlock.timer = trafficBlock.timer * 0.8
  end

  if math.max(lastCommand.throttle, lastCommand.throttle) > 0.5 and aiSpeed < 1 then
    aiCannotMoveTime = aiCannotMoveTime + dt
  else
    aiCannotMoveTime = 0
  end

  if aiSpeed < 3 then
    trafficSide.cTimer = trafficSide.cTimer + dt
    trafficSide.timer = (trafficSide.timer + dt) % (2 * trafficSide.timerRange)
    trafficSide.side = sign2(trafficSide.timerRange - trafficSide.timer)
  else
    trafficSide.cTimer = math.max(0, trafficSide.cTimer - dt)
    trafficSide.timer = 0
    trafficSide.side = 1
  end

  changePlanTimer = math.max(0, changePlanTimer - dt)

  ------------------ RANDOM MODE ----------------
  if M.mode == 'random' then
    local newRoute
    if currentRoute == nil or currentRoute.plan[2].pathidx > #currentRoute.path * 0.5 then
      local wp1, wp2 = mapmgr.findClosestRoad(aiPos)
      if wp1 == nil or wp2 == nil then
        warningAIDisabled("Could not find a road network, or closest road is too far")
        return
      end

      if internalState == 'offroad' then
        local vec1 = mapData.positions[wp1] - aiPos
        local vec2 = mapData.positions[wp2] - aiPos
        if aiDirVec:dot(vec1) > 0 and aiDirVec:dot(vec2) > 0 then
          if vec1:squaredLength() > vec2:squaredLength() then
            wp1, wp2 = wp2, wp1
          end
        elseif aiDirVec:dot(mapData.positions[wp2] - mapData.positions[wp1]) > 0 then
          wp1, wp2 = wp2, wp1
        end
      elseif aiDirVec:dot(mapData.positions[wp2] - mapData.positions[wp1]) > 0 then
        wp1, wp2 = wp2, wp1
      end

      newRoute = mapData:getRandomPath(wp1, wp2, driveInLaneFlag and 1e4 or 1)

      if newRoute and newRoute[1] then
        local tempPlan = planAhead(newRoute, currentRoute)
        if tempPlan then
          if not currentRoute then
            currentRoute = newRoute
          else
            local curPlanIdx = currentRoute.plan[2].pathidx
            local curPathCount = #currentRoute.path
            if curPlanIdx >= curPathCount * 0.9 or ((tempPlan.targetPos-aiPos):dot(aiDirVec) >= 0 and (curPlanIdx >= curPathCount*0.8 or tempPlan.targetSpeed >= aiSpeed)) then
              currentRoute = newRoute
            end
          end
        end
      end
    end

    if currentRoute ~= newRoute then
      planAhead(currentRoute)
    end

  ------------------ TRAFFIC MODE ----------------
  elseif M.mode == 'traffic' then
    local newRoute
    if currentRoute == nil or aiPos:squaredDistance(mapData.positions[currentRoute.path[#currentRoute.path]]) < square(getMinPlanLen()) or trafficBlock.timer > trafficBlock.limit then -- getPathLen(currentRoute.path, currentRoute.plan[2].pathidx)
      if currentRoute and currentRoute.path[3] and trafficBlock.timer <= trafficBlock.limit and internalState ~= 'offroad' then
        local path = currentRoute.path
        local pathCount = #path
        local cr0, cr1, cr2 = path[pathCount-2], path[pathCount-1], path[pathCount]
        local cr2Pos = mapData.positions[cr2]
        local dir1 = (cr2Pos - mapData.positions[cr1]):normalized()
        local vec = cr2Pos - mapData.positions[cr0]
        local mirrorOfVecAboutdir1 = (2 * vec:dot(dir1) * dir1 - vec):normalized()
        pathExtend(path, mapData:getPathT(cr2, cr2Pos, getMinPlanLen(), 1e4, mirrorOfVecAboutdir1))
      else
        local wp1, wp2 = mapmgr.findClosestRoad(aiPos)

        if wp1 == nil or wp2 == nil then
          guihooks.message("Could not find a road network, or closest road is too far", 5, "AI debug")
          currentRoute = nil
          internalState = 'offroad'
          changePlanTimer = 0
          driveCar(0, 0, 0, 1)
          return
        end

        local dirVec
        if trafficBlock.timer > trafficBlock.limit and not mapData.graph[wp1][wp2][3] and (mapData.radius[wp1] + mapData.radius[wp2]) * 0.5 > 2 then
          dirVec = -aiDirVec
        else
          dirVec = aiDirVec
        end

        wp1 = pickAiWp(wp1, wp2, dirVec)

        -- local newRoute = mapData:getRandomPathG(wp1, aiDirVec, getMinPlanLen(), 0.4, 1 / (aiSpeed + 1e-30))
        --newRoute = mapData:getRandomPathG(wp1, dirVec, getMinPlanLen(), 0.4, math.huge)
        newRoute = mapData:getPathT(wp1, aiPos, getMinPlanLen(), 1e4, aiDirVec)

        if newRoute and newRoute[1] then
          local tempPlan = planAhead(newRoute, currentRoute)

          if tempPlan then
            trafficBlock.limit = math.random() * 10 + 5
            trafficBlock.horn = math.random() >= 0.4 and trafficBlock.limit - (math.random() * 2.5) or math.huge -- horn time start
            trafficSignal.hLimit = math.random() * 2
            intersection.turn = 0

            if not currentRoute or trafficBlock.timer > trafficBlock.limit then
              trafficBlock.timer = 0
              currentRoute = newRoute
            elseif tempPlan.targetSpeed >= aiSpeed and targetsCompatible(currentRoute, newRoute) then
              currentRoute = newRoute
            end
          end
        end
      end
    end

    if currentRoute ~= newRoute then
      planAhead(currentRoute)
    end

  ------------------ MANUAL MODE ----------------
  elseif M.mode == 'manual' then
    if validateInput(wpList or manualPath) then
      newManualPath()
    end

    if aggressionMode == 'rubberBand' then
      updatePlayerData()
      if player ~= nil then
        if (aiPos - player.pos):dot(aiDirVec) > 0 then
          setAggressionInternal(math.max(math.min(0.1 + math.max((150 - player.pos:distance(aiPos))/150, 0), 1), 0.5))
        else
          setAggressionInternal()
        end
      end
    end

    planAhead(currentRoute)

  ------------------ SPAN MODE ------------------
  elseif M.mode == 'span' then
    if currentRoute == nil then
      local positions = mapData.positions
      local wpAft, wpFore = mapmgr.findClosestRoad(aiPos)
      if not (wpAft and wpFore) then
        warningAIDisabled("Could not find a road network, or closest road is too far")
        return
      end
      if aiDirVec:dot(positions[wpFore] - positions[wpAft]) < 0 then
        wpAft, wpFore = wpFore, wpAft
      end

      local target, targetLink

      if not edgeDict then
        -- creates the edgeDict and returns a random edge
        target, targetLink = getMapEdges(M.cutOffDrivability or 0, wpFore)
        if not target then
          warningAIDisabled("No available target with selected characteristics")
          return
        end
      end

      local newRoute = {}

      while true do
        if not target then
          local maxDist = -math.huge
          local lim = 1
          repeat
            -- get most distant non walked edge
            for k, v in pairs(edgeDict) do
              if v <= lim then
                if lim > 1 then
                  edgeDict[k] = 1
                end
                local i = string.find(k, '\0')
                local n1id = string.sub(k, 1, i-1)
                local sqDist = positions[n1id]:squaredDistance(aiPos)
                if sqDist > maxDist then
                  maxDist = sqDist
                  target = n1id
                  targetLink = string.sub(k, i+1, #k)
                end
              end
            end
            lim = math.huge -- if the first iteration does not produce a target
          until target
        end

        local nodeDegree = 1
        for lid, edgeData in pairs(mapData.graph[target]) do
          -- we're looking for neighboring nodes other than the targetLink
          if lid ~= targetLink then
            nodeDegree = nodeDegree + 1
          end
        end
        if nodeDegree == 1 then
          local key = target..'\0'..targetLink
          edgeDict[key] = edgeDict[key] + 1
        end

        newRoute = mapData:spanMap(wpFore, wpAft, target, edgeDict, driveInLaneFlag and 10e7 or 1)

        if not newRoute[2] and wpFore ~= target then
          -- remove edge from edgeDict list and get a new target (while loop will iterate again)
          edgeDict[target..'\0'..targetLink] = nil
          edgeDict[targetLink..'\0'..target] = nil
          target = nil
          if next(edgeDict) == nil then
            warningAIDisabled("Could not find a path to any of the possible targets")
            return
          end
        elseif not newRoute[1] then
          warningAIDisabled("No Route Found")
          return
        else
          -- insert the second edge node in newRoute if it is not already contained
          local newRouteLen = #newRoute
          if newRoute[newRouteLen-1] ~= targetLink then
            newRoute[newRouteLen+1] = targetLink
          end
          break
        end
      end

      if planAhead(newRoute) == nil then
        return
      end
      currentRoute = newRoute
    else
      planAhead(currentRoute)
    end

  ------------------ FLEE MODE ------------------
  elseif M.mode == 'flee' then
    updatePlayerData()
    if player then
      if validateInput() then
        targetWPName = wpList[1]
        wpList = nil
      end

      local aggrValue = aggressionMode == 'manual' and M.extAggression or 1
      setAggressionInternal(math.max(0.3, aggrValue - 0.002 * player.pos:distance(aiPos)))

      fleePlan()

      if internalState == 'offroad' then
        local targetPos = aiPos + (aiPos - player.pos) * 100
        local targetSpeed = math.huge
        driveToTarget(targetPos, 1, 0, targetSpeed)
        return
      end
    else
      --guihooks.message("No vehicle to Flee from", 5, "AI debug") -- TODO: this freezes the up because it runs on the gfx step
      return
    end

  ------------------ CHASE MODE ------------------
  elseif M.mode == 'chase' or M.mode == 'follow' then
    updatePlayerData()
    if player then
      chasePlan()

      if plPrevVel then
        plPrevVel:set(player.vel)
      else
        plPrevVel = vec3(player.vel)
      end

      if internalState == 'tail' then
        --internalState = 'onroad'
        --currentRoute = nil
        local plai = player.pos - aiPos
        local relvel = aiVel:dot(plai) - player.vel:dot(plai)
        if chaseData.suspectState == 'stopped' then
          driveToTarget(player.pos, 0, 0.5, 0) -- could be better to apply throttle, brake, and targetSpeed values here
        elseif relvel > 0 then
          driveToTarget(player.pos + (plai:length() / (relvel + 1e-30)) * player.vel, 1, 0, math.huge)
        else
          driveToTarget(player.pos, 1, 0, math.huge)
        end
        return
      elseif internalState == 'offroad' then
        if M.mode == 'follow' then
          local throttle, brake, targetSpeed = offRoadFollowControl()
          driveToTarget(player.pos, throttle, brake, targetSpeed)
        else
          driveToTarget(player.pos, 1, 0, math.huge)
        end
        return
      elseif currentRoute == nil then
        driveCar(0, 0, 0, 1)
        return
      end

    else
      --guihooks.message("No vehicle to Chase", 5, "AI debug")
      return
    end

  ------------------ STOP MODE ------------------
  elseif M.mode == 'stop' then
    if currentRoute then
      planAhead(currentRoute)
      local targetSpeed = math.max(0, aiSpeed - math.sqrt(math.max(0, square(staticFrictionCoef * g) - square(sensors.gx2))) * dt)
      currentRoute.plan.targetSpeed = math.min(currentRoute.plan.targetSpeed, targetSpeed)
    elseif aiVel:dot(aiDirVec) > 0 then
      driveCar(0, 0, 0.5, 0)
    else
      driveCar(0, 1, 0, 0)
    end
    if aiSpeed < 0.08 then --  or aiVel:dot(aiDirVec) < 0
      driveCar(0, 0, 0, 1) -- only parkingbrake
      M.mode = 'disabled'
      M.manualTargetName = nil
      M.updateGFX = nop
      resetMapAndRoute()
      stateChanged()
      return
    end
  end
  -----------------------------------------------

  if currentRoute then
    local plan = currentRoute.plan
    local targetPos = plan.targetPos
    local aiSeg = plan.aiSeg

    -- cleanup path if it has gotten too long
    if not race and plan[aiSeg].pathidx >= 10 and currentRoute.path[20] then
      local newPath = {}
      local j, k = 0, plan[aiSeg].pathidx
      for i = k, #currentRoute.path do
        j = j + 1
        newPath[j] = currentRoute.path[i]
      end
      currentRoute.path = newPath

      k = k - 1
      for i = 1, #plan do
        plan[i].pathidx = plan[i].pathidx - k
      end
    end

    local targetSpeed = plan.targetSpeed
    trafficBlock.coef = plan.trafficBlockCoef or trafficBlock.coef

    if ai.upVec:dot(gravityVec) > 0 then -- vehicle upside down
      return
    end

    local lowTargetSpeedVal = 0.24
    if not plan[aiSeg+2] and ((targetSpeed < lowTargetSpeedVal and aiSpeed < 0.15) or (targetPos - aiPos):dot(aiDirVec) < 0) then
      if M.mode == 'span' then
        local path = currentRoute.path
        for i = 1, #path - 1 do
          local key = path[i]..'\0'..path[i+1]
          -- in case we have gone over an edge that is not in the edgeDict list
          edgeDict[key] = edgeDict[key] and (edgeDict[key] * 20)
        end
      end

      driveCar(0, 0, 0, 1)
      aistatus('route done', 'route')
      guihooks.message("Route done", 5, "AI debug")
      currentRoute = nil
      speedRecordings = {}
      return
    end

    -- come off controls when close to intermediate node with zero speed (ex. intersection), arcade autobrake takes over
    if (plan[aiSeg+1].speed == 0 and plan[aiSeg+2]) and aiSpeed < 0.15 then
      driveCar(0, 0, 0, 0)
      return
    end

      -- TODO: this still runs if there is no currentPlan, but raises error if there is no targetSpeed
    if not controller.isFrozen and aiSpeed < 0.1 and targetSpeed > 0.5 and (lastCommand.throttle ~= 0 or lastCommand.brake ~= 0) then
      crash.time = crash.time + dt
      if crash.time > 1 then
        crash.dir = vec3(aiDirVec)
        crash.manoeuvre = 1
      end
    else
      crash.time = 0
    end

    -- Throttle and Brake control
    local dif = targetSpeed - aiSpeed
    if dif <= 0 then
      targetSpeedSmoother:set(dif)
    end
    local speedDif = targetSpeedSmoother:get(dif, dt)
    local lowSpeedDif = (speedDif - clamp((aiSpeed - 2) * 0.5, 0, 1)) * 0.5
    local lowTargSpeedConstBrake = lowTargetSpeedVal - targetSpeed -- apply constant brake below some targetSpeed
    local throttle = clamp(lowSpeedDif, 0, 1) * sign(math.max(0, -lowTargSpeedConstBrake)) -- throttle not enganged for targetSpeed < 0.26
    local brakeLimLow = sign(math.max(0, lowTargSpeedConstBrake)) * 0.5
    local brake = clamp(-speedDif, brakeLimLow, 1) * sign(math.max(0, electrics.values.smoothShiftLogicAV or 0 - 3)) -- arcade autobrake comes in at |smoothShiftLogicAV| < 5

    driveToTarget(targetPos, throttle, brake)
  end
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
  M.avoidCarsMaster = v
  if M.avoidCarsMaster == 'off' or M.avoidCarsMaster == 'on' then
    avoidCars = M.avoidCarsMaster
  else
    avoidCars = M.mode == 'manual' and 'off' or 'on'
  end
  stateChanged()
end

local function driveInLane(v)
  if v == 'on' then
    M.driveInLaneFlag = 'on'
    driveInLaneFlag = true
  else
    M.driveInLaneFlag = 'off'
    driveInLaneFlag = false
  end
  stateChanged()
end

local function setMode(mode)
  if M.avoidCarsMaster == 'off' or M.avoidCarsMaster == 'on' then
    avoidCars = M.avoidCarsMaster
  else
    avoidCars = (mode == 'manual' or (mode == nil and M.mode == 'manual')) and 'off' or 'on'
  end

  if mode ~= nil then
    M.mode = mode
  end

  if M.mode ~= 'script' then
    if M.mode ~= 'disabled' and M.mode ~= 'stop' then
      resetMapAndRoute()

      mapmgr.requestMap() -- a map request is also performed in the startFollowing function of scriptai
      M.updateGFX = updateGFX
      targetSpeedSmoother = newTemporalSmoothingNonLinear(math.huge, 0.2, vec3(obj:getVelocity()):length())

      if controller.mainController then
        controller.mainController.setGearboxMode("arcade")
      end
    end

    if M.mode == 'disabled' then
      driveCar(0, 0, 0, 0)
      M.updateGFX = nop
      currentRoute = nil
    end

    if M.mode == 'traffic' then
      setSpeedMode('legal')
      obj:setSelfCollisionMode(2)
      obj:setAerodynamicsMode(2)
    else
      obj:setSelfCollisionMode(1)
      obj:setAerodynamicsMode(1)
    end

    stateChanged()
  end

  speedRecordings = {}
  trajecRec = {last = 0}
  routeRec = {last = 0}
end

local function reset() -- called when the user pressed I
  M.manualTargetName = nil
  trafficBlock = {timer = 0, coef = 0, limit = 6, horn = 0}

  trafficSide.timer = 0
  trafficSide.cTimer = 0
  trafficSide.side = 1

  trafficSignal = {hTimer = 0, hLimit = 1}
  intersection = {stopTime = 0, timer = 0, turn = 0}
  smoothTcs:set(1)

  if M.mode ~= 'disabled' then
    driveCar(0, 0, 0, 0)
  end
  setMode() -- some scenarios don't work if this is changed to setMode('disabled')
  stateChanged()
end

local function resetLearning()
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
  M.manualTargetName = wp
  validateInput = validateUserInput
  wpList = {wp}
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
  M.cutOffDrivability = drivability or 0
  stateChanged()
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
  return M.updateGFX == updateGFX or (scriptai ~= nil and scriptai.isDriving())
end

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
