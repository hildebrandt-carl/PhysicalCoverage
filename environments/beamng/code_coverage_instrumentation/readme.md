First you need to place `ai.lua` inside the original code folder.

```shell
cp C:\Users\hilde\Documents\Beamng\BeamNG.tech.v0.23.5.1\lua\vehicle\ai.lua C:\Users\hilde\Documents\Beamng\PhysicalCoverageBeamNG\InstrumentCode\original_code\ai.lua
```

Next you want to run the python script to instrument the code
```shell
clear; python .\process.py
```

That will produce an instrumented file inside the `instrumented_code` folder. Copy that back to your `BeamNG` install.
```shell
cp instrumented_code\ai.lua C:\Users\hilde\Documents\Beamng\BeamNG.tech.v0.23.5.1\lua\vehicle\ai.lua 
```

Next you need to link beamngpy and Luo with the new functions to do that, add the following functions to the end of the `researchVE.lua` file. The file can be located at `C:\Users\hilde\Documents\Beamng\BeamNGpy\src\beamngpy\lua\researchVE.lua`. 

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

Then you need to add a way to get the coverage for each of the vehicles in BeamNG. To do that edit the `vehicle.py` file. You can find that file in `C:\Users\hilde\Documents\Beamng\BeamNGpy\src\beamngpy\vehicle.py`. Add the following:
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


-----

These are the changes I made to the original code. They do not need to changed in your code, unless you are starting with a fresh `ai.lua`. If you are not using `ai.lua` then you do not need to worry.

You can now run the `run_random_tests.py` script to generate data.

Note: For this to work you need to manually unwrap any single line if statements. They break they code:(
```
if not plan then return end
```
Becomes
```
if not plan then
  return
end
```

Additionally any if statements written over multiple lines need to be grouped together


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