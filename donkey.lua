require 'image'
paths.dofile('DataLoader.lua')

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')

local function loadImage(path)
   local input = image.load(path)
   if input:dim() == 2 then -- 1-channel image loaded as 2D tensor
      input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
   elseif input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
      input = input:repeatTensor(3,1,1)
   elseif input:dim() == 3 and input:size(1) == 3 then -- 3-channel image
   elseif input:dim() == 3 and input:size(1) == 4 then -- image with alpha
      input = input[{{1,3},{},{}}]
   else
      print(#input)
      error('not 2-channel or 3-channel image')
   end
   input = image.scale(input, 256, 256)
   -- input:cuda()
   return input
end

-- VGG preprocessing
local bgr_means = {103.939,116.779,123.68}
local function vggPreprocess(img)
  local im2 = img:clone()
  im2[{1,{},{}}] = img[{3,{},{}}]
  im2[{3,{},{}}] = img[{1,{},{}}]

  im2:mul(255)
  for i=1,3 do
    im2[i]:add(-bgr_means[i])
  end
  return im2
end

local function centerCrop(input)
   local oH = 224
   local oW = 224
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
   return out
end

-- function to load the image
local loadHook = function(path)
   collectgarbage()
   local worked, im = pcall(loadImage, path)
   if worked then
    local input = loadImage(path)
    local vggPreprocessed = vggPreprocess(input)
    local out = centerCrop(vggPreprocessed)
    return out, worked
   else
    return 0, worked
   end
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   loader = torch.load(trainCache)
   loader.loadHook = loadHook
else
   print('Creating test metadata')
   loader = DataLoader{data = opt.data, validation_data = opt.validation_data}
   torch.save(trainCache, loader)
   loader.loadHook = loadHook
end
collectgarbage()