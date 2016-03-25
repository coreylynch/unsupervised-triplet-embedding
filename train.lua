require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
-- require 'DataLoader'
local TripletEmbedder = require 'TripletEmbedder'
local cjson = require 'cjson'
require 'optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a triplet metric embedder')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-data', '', 'file containing (id, image path, sentence)')
cmd:option('-validation_data', '', 'file containing (id, image path, sentence)')
cmd:option('-cache', 'cache', 'cache folder')
cmd:option('-nThreads', 0, 'number of loader threads')
cmd:option('-manualSeed', 28, 'seed')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')

-- Optimization: for the CNN
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

cmd:option('-model','vgg_pretrained','model to use for the embedder; vgg_pretrained|vgg|vgg19')
cmd:option('-freeze_pretrained', 1,'whether or not to freeze every layer except last during fine-tuning')
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
print(opt)

print('Saving everything to: ' .. opt.cache)
os.execute('mkdir -p ' .. opt.cache)

-------------------------------------------------------------------------------
-- Create the Data Loader
-------------------------------------------------------------------------------
assert(opt.data ~= '', "path to data file not specified?")
assert(opt.vocab_json_file ~= '', "path to vocab json not specified?")
-- Spins up the multiple data loader threads
paths.dofile("data.lua")

-------------------------------------------------------------------------------
-- Initialize the model
-------------------------------------------------------------------------------
local model = paths.dofile(opt.model .. '.lua')
model = model:cuda()

local triplet_embedder = TripletEmbedder.getNetwork(model)
triplet_embedder = triplet_embedder:cuda()

local criterion = nn.MarginRankingCriterion(0.5)
criterion = criterion:cuda()
print(criterion)

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = triplet_embedder:getParameters()
print('total number of parameters in CNN: ', params:nElement())
assert(params:nElement() == grad_params:nElement())
collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  
  triplet_embedder:evaluate()

  
  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun(inputs)
  triplet_embedder:training()
  grad_params:zero()

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- forward the ConvNet on images (most work happens here)
  local ranks = triplet_embedder:forward(inputs)
  
  -- forward the margin ranking criterion
  local loss = criterion:forward(ranks, 1.0)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dcriterion = criterion:backward(ranks, 1.0)
  
  -- backprop the triplet embedder
  triplet_embedder:backward(inputs, dcriterion)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    grad_params:add(opt.cnn_weight_decay, params)
  end
  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { total_loss = loss:mean() }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

-- GPU inputs (preallocate)
local query_image_inputs = torch.CudaTensor()
local positive_image_inputs = torch.CudaTensor()
local negative_image_inputs = torch.CudaTensor()

local function write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

local function trainBatch(triplets)
  cutorch.synchronize()
  collectgarbage()

  -- copy the cpu inputs into the gpu tensors
  query_image_inputs:resize(triplets.query_images:size()):copy(triplets.query_images)
  positive_image_inputs:resize(triplets.positive_images:size()):copy(triplets.positive_images)
  negative_image_inputs:resize(triplets.negative_images:size()):copy(triplets.negative_images)
  
  local inputs = {query_image_inputs, positive_image_inputs, negative_image_inputs}

   -- eval loss/gradient
   local losses = lossFun(inputs)
   
   if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
   print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    -- evaluate the validation performance
    -- local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    -- print('validation loss: ', val_loss)
    -- -- print(lang_stats)
    -- val_loss_history[iter] = val_loss
    -- if lang_stats then
    --   val_lang_stats_history[iter] = lang_stats
    -- end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    -- checkpoint.val_loss_history = val_loss_history
    -- checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    -- checkpoint.val_lang_stats_history = val_lang_stats_history

    write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
  end

  -- decay the learning rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end
  
  if opt.cnn_optim == 'sgd' then
    sgd(params, grad_params, cnn_learning_rate)
  elseif opt.cnn_optim == 'sgdm' then
    sgdm(params, grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
  elseif opt.cnn_optim == 'adam' then
    adam(params, grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
  else
    error('bad option for opt.cnn_optim')
  end

end

while true do
   cutorch.synchronize()
   -- queue jobs to data-workers
   donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         local triplets = loader:getBatch(opt.batch_size)
         return triplets
      end,
      -- the end callback (runs in the main thread)
      trainBatch
   )
   donkeys:synchronize()
   cutorch.synchronize()

  iter = iter + 1

  -- stopping conditions
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think

end