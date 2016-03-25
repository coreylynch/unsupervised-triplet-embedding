require 'cudnn'
require 'cunn'

-- TODO: this can go in a utils
function getPretrainedModel(backend)
  if not paths.dirp('model_weights') then
    print('=> Downloading VGG 19 model weights')
    os.execute('mkdir model_weights')
    local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
    local proto_url = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'
    os.execute('wget --output-document model_weights/VGG_ILSVRC_19_layers.caffemodel ' .. caffemodel_url)
    os.execute('wget --output-document model_weights/VGG_ILSVRC_19_layers_deploy.prototxt ' .. proto_url)
  end
  
  local proto = 'model_weights/VGG_ILSVRC_19_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_19_layers.caffemodel'

  if backend == 'cudnn' then
      require 'cudnn'
  elseif backend == 'cunn' then
      print('using cunn backend')
  else
      error('unrecognized backend: ' .. backend)
  end

  local original = loadcaffe.load(proto, caffemodel, backend)

  local keep_layers = 43 -- original has 46, we remove softmax, class scores, and dropout layer

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, keep_layers do
    local layer = original:get(i)

    if opt.freeze_pretrained == 1 then
      -- Freeze all layers except the last one
      if i ~= keep_layers then
        print ("Freezing layer " .. i)
        local function f()
        end
        layer.accGradParameters = f
      end
    end

    cnn_part:add(layer)
  end

  cnn_part:add(nn.Normalize(2)) -- L2 normalize embeddings so that dot prod reduces to cosine sim

  return cnn_part
end

local model = getPretrainedModel(opt.backend)

print(model)

return model