require 'lfs'
require 'image'
local cjson = require 'cjson'
local ffi = require 'ffi'
torch.setdefaulttensortype('torch.FloatTensor')

local DataLoader = torch.class('DataLoader')
DataLoader.__index = DataLoader

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToTensor(self, query_image_table, positive_image_table, negative_image_table, quantity)
   local query_image_data, positive_image_data, negative_image_data
   assert(query_image_table[1]:dim() == 3)
   assert(positive_image_table[1]:dim() == 3)
   assert(negative_image_table[1]:dim() == 3)
   
   -- init image tensors
   query_image_data = torch.Tensor(quantity,
           3, 224, 224)
   positive_image_data = torch.Tensor(quantity,
           3, 224, 224)
   negative_image_data = torch.Tensor(quantity,
           3, 224, 224)
   -- fill all the tensors
   for i=1,quantity do
      query_image_data[i]:copy(query_image_table[i])
      positive_image_data[i]:copy(positive_image_table[i])
      negative_image_data[i]:copy(negative_image_table[i])
   end
   return query_image_data, positive_image_data, negative_image_data
end

-- get a random example index that != positive example index
function getRandomContrastiveIndex(self, positive_idx)
  local idx = math.ceil(torch.uniform() * self.imagePaths:size(1))
  while idx == positive_idx do
    idx = math.ceil(torch.uniform() * self.imagePaths:size(1))
  end
  return idx
end

-- Given the positive index, return a contrastive negative image
-- (from a random other listing)
local function getNegativeImage(self, positive_idx)
  local index = getRandomContrastiveIndex(self, positive_idx)
  local image_paths_string = ffi.string(torch.data(self.imagePaths[index]))
  local image_paths = image_paths_string:split(",")
  local shuffled = torch.randperm(#image_paths)
  local negative_image_path = image_paths[shuffled[1]]
  local negative_image, worked = self.loadHook(negative_image_path)
  return negative_image, worked
end

-- Get a pair of random training listing's images
local function getImagePair(self, path)
  local index = math.ceil(torch.uniform() * self.imagePaths:size(1))
  local image_paths_string = ffi.string(torch.data(self.imagePaths[index]))
  local image_paths = image_paths_string:split(",")
  local shuffled = torch.randperm(#image_paths)

  local query_image_path = image_paths[shuffled[1]]
  local positive_image_path = image_paths[shuffled[2]]

  local query_image, worked_q = self.loadHook(query_image_path)
  local positive_image, worked_p = self.loadHook(positive_image_path)
  
  return query_image, positive_image, index, worked_p, worked_q
end

function DataLoader:__init(opt)
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut' 
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Load training data and cache
   self.imagePaths = torch.CharTensor()
   self.validationImagePaths = torch.CharTensor()
   --==========================================================================

   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. opt.data .. "' |"
                                           .. cut .. " -f1 -d' '"))

   local imPathsMaxLength = tonumber(sys.fexecute(cut .. " " .. opt.data ..
                                               " -f2 | " .. wc .. " -L | " .. cut 
                                               .. " -f1") + 1) + 1

   local validationLength = tonumber(sys.fexecute(wc .. " -l '"
                                           .. opt.validation_data .. "' |"
                                           .. cut .. " -f1 -d' '"))

   local validationImPathsMaxLength = tonumber(sys.fexecute(cut .. " " .. opt.validation_data ..
                                               " -f2 | " .. wc .. " -L | " .. cut 
                                               .. " -f1") + 1) + 1

   assert(length > 0, "Could not find any image file in the given input paths")
   assert(imPathsMaxLength > 0, "paths are length 0?")

   assert(validationLength > 0, "Could not find any validation image file in the given input paths")
   assert(validationImPathsMaxLength > 0, "validation paths are length 0?")

   self.imagePaths:resize(length, imPathsMaxLength):fill(0)
   self.validationImagePaths:resize(validationLength, validationImPathsMaxLength):fill(0)

   local im_path_data = self.imagePaths:data()
   local val_im_path_data = self.validationImagePaths:data()

   print("=> Loading image paths")
   local count = 0
   for line in io.lines(opt.data) do
      local id_paths = line:split("\t")
      ffi.copy(im_path_data, id_paths[2])

      im_path_data = im_path_data + imPathsMaxLength

      if count % 1000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end
   print("=> Loaded " .. count .. " image paths")

   print("=> Loading validation image paths")
   local count = 0
   for line in io.lines(opt.validation_data) do
      local id_paths = line:split("\t")
      ffi.copy(val_im_path_data, id_paths[2])

      val_im_path_data = val_im_path_data + validationImPathsMaxLength

      if count % 100 == 0 then
         xlua.progress(count, validationLength)
      end;
      count = count + 1
   end
   print("=> Loaded " .. count .. "validation image paths")

end

--[[
  Returns a quantity size batch of triplet instances: (query image, positive image, contrastive image)
  The data is iterated linearly in order
--]]
function DataLoader:getBatch(quantity)
   assert(quantity)

   local query_image_table = {}
   local positive_image_table = {}
   local negative_image_table = {}

   local cnt = 0
   for i=1,quantity do
      -- Get one matching (query_image, positive_image) pair
      local query_image, positive_image, positive_idx, worked_q, worked_p = getImagePair(self)
      local negative_image, worked_n = getNegativeImage(self, positive_idx)
      if worked_q and worked_p and worked_n then
        table.insert(query_image_table, query_image)
        table.insert(positive_image_table, positive_image)
        table.insert(negative_image_table, negative_image)
        cnt = cnt + 1
      end
   end

   -- Convert image tables to batch tensors
   local query_images, positive_images, negative_images = tableToTensor(self, query_image_table, positive_image_table, negative_image_table, cnt)

   local data = {}
   data.query_images = query_images
   data.positive_images = positive_images
   data.negative_images = negative_images

   return data
end

function DataLoader:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      -- load the sample
   local query_image_table = {}
   local positive_image_table = {}
   local negative_image_table = {}

   local cnt = 0
   for i=1,quantity do
      -- Get one matching (query_image, positive_image) pair
      local query_image, positive_image, positive_idx, worked_q, worked_p = getImagePair(self)
      local negative_image, worked_n = getNegativeImage(self, positive_idx)
      if worked_q and worked_p and worked_n then
        table.insert(query_image_table, query_image)
        table.insert(positive_image_table, positive_image)
        table.insert(negative_image_table, negative_image)
        cnt = cnt + 1
      end
   end

   -- Convert image tables to batch tensors
   local query_images, positive_images, negative_images = tableToTensor(self, query_image_table, positive_image_table, negative_image_table, cnt)

   local val_data = {}
   val_data.query_images = query_images
   val_data.positive_images = positive_images
   val_data.negative_images = negative_images

   return val_data
end
end
