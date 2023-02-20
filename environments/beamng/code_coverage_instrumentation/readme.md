# Adding Code Coverage to BeamNGPy


## Adding instrumented Lua Code

We have provided you with instrumented `ai.lua` code. This new `ai.lua` is an instrumented version which updates an array each time a line is covered. We use that array to compute line, branch, and path coverage. At the bottom of this document we have provided you with how to use our tool to instrument any Lua code.

Replace the original `ai.lua` file with our newly instrumented file inside the `instrumented_code` folder. You can do that using:
```bash
cp instrumented_code\ai.lua C:<path to beamng>\BeamNG.tech.v0.21.3.0\lua\vehicle\ai.lua 
```

## Adding Coverage functions to BeamNGpy

Next you need to link beamngpy and Luo with the new functions to do that, add the following functions to the end of the `researchVE.lua` file. The file can be located at `C:<path to beamngpy>\BeamNGpy\src\beamngpy\lua\researchVE.lua`. 

**Note:** Make sure to add these before the `return M` line

```lua
-- Added this to handle line and branch coverage
M.handleGetLineCoverageArray = function(skt, msg)
  local response = {type = 'GetLineCoverageArray', data = response}
  local coverage = ai.getLineCoverageArray()
  response['lines_covered'] = coverage
  log('I', 'GetLineCoverageArray sending back response ' .. tostring(response))
  rcom.sendMessage(skt, response)
end

M.handleGetBranchCoverageArray = function(skt, msg)
  local response = {type = 'GetBranchCoverageArray', data = response}
  local coverage = ai.getBranchCoverageArray()
  response['branches_covered'] = coverage
  log('I', 'GetBranchCoverageArray sending back response ' .. tostring(response))
  rcom.sendMessage(skt, response)
end

M.handleGetPathCoverageArray = function(skt, msg)
  local response = {type = 'GetPathCoverageArray', data = response}
  local path = ai.getPathCoverageArray()
  response['path_taken'] = path
  log('I', 'GetPathCoverageArray sending back response ' .. tostring(response))
  rcom.sendMessage(skt, response)
end

M.handleResetCoverageArrays = function(skt, msg)
  local response = {type = 'ResetCoverageArrays', data = response}
  ai.resetCoverageArrays()
  response['result'] = true
  log('I', 'ResetCoverageArrays sending back response ' .. tostring(response))
  rcom.sendMessage(skt, response)
end

M.handleStartCoverage = function(skt, msg)
  local response = {type = 'StartCoverage', data = response}
  ai.startCoverage()
  response['result'] = true
  log('I', 'StartCoverage sending back response ' .. tostring(response))
  rcom.sendMessage(skt, response)
end
```

Then you need to add a way to get the coverage for each of the vehicles in BeamNG. To do that edit the `vehicle.py` file. You can find that file in `C:<path to beamngpy>\BeamNGpy\src\beamngpy\vehicle.py`. Add the following:

```python
    # Requires you import numpy as np
    # returns list of lines covered
    def get_line_coverage(self):
        # Request data
        print("Requesting lua function: GetLineCoverageArray")
        data = dict(type='GetLineCoverageArray')
        self.send(data)
        # Wait to get a response
        resp = self.recv()
        print("Received response from GetLineCoverageArray")
        assert (resp['type'] == 'GetLineCoverageArray')
        # Retrun the data
        result = np.fromstring(resp['lines_covered'], dtype=np.uint16, sep=' ')
        return result

    # returns list of lines covered
    def get_branch_coverage(self):
        # Request data
        print("Requesting lua function: GetBranchCoverageArray")
        data = dict(type='GetBranchCoverageArray')
        self.send(data)
        # Wait to get a response
        resp = self.recv()
        print("Received response from GetBranchCoverageArray")
        assert (resp['type'] == 'GetBranchCoverageArray')
        # Retrun the data
        result = np.fromstring(resp['branches_covered'], dtype=np.uint16, sep=' ')
        return result

    # returns the path taken
    def get_path_taken(self):
        # Request data
        print("Requesting lua function: GetPathCoverageArray")
        data = dict(type='GetPathCoverageArray')
        self.send(data)
        # Wait to get a response
        resp = self.recv()
        print("Received response from GetPathCoverageArray")
        assert (resp['type'] == 'GetPathCoverageArray')
        # Retrun the data
        result = resp['path_taken']
        return result

    # Resets the coverage arrays
    def reset_coverage_arrays(self):
        # Request data
        print("Requesting lua function: ResetCoverageArrays")
        data = dict(type='ResetCoverageArrays')
        self.send(data)
        # Wait to get a response
        resp = self.recv()
        print("Received response from ResetCoverageArrays")
        assert (resp['type'] == 'ResetCoverageArrays')
        return True

    # Start recording coverage
    def start_coverage(self):
        # Request data
        print("Requesting lua function: StartCoverage")
        data = dict(type='StartCoverage')
        self.send(data)
        # Wait to get a response
        resp = self.recv()
        print("Received response from StartCoverage")
        assert (resp['type'] == 'StartCoverage')
        return True
```

**Note:** remember to add the `import numpy as np` to the top of that file.

**YOU ARE NOW DONE AND CAN RETURN TO RUNNING OUR STUDY**

---
## Additional Changes to the Lua.ai

These are the changes I made to the original code.

I had to unwrap any statements written on a single line. Therefor

```lua
if not plan then return end
```

Becomes:
```lua
if not plan then
  return
end
```

I also needed to replace the following variables:
```lua
-- [[ STORE FREQUENTLY USED FUNCTIONS IN UPVALUES ]] --
local max = math.max
local min = math.min
local sin = math.sin
local asin = math.asin
local pi = math.pi
local abs = math.abs
local sqrt = math.sqrt
local floor = math.floor
local tableInsert = table.insert
local tableRemove = table.remove
local strFormat = string.format
```

This removes a number of up variables and thus allows us to instrument the function UpdateGFX in the code.


## Insturmenting any Lua Code

Note: This step was already done, and we have provided the instrumented Lua code in `instrumented_code` folder. However to instrument any code do the following

```bash
cp C:<path to file requiring instrumentation>\file.lua C:<Path to This Repo>\PhysicalCoverageBeamNG\InstrumentCode\original_code\file.lua
```

Next you want to run the python script to instrument the code
```bash
python .\process.py
```

That will produce an instrumented file inside the `instrumented_code` folder. 