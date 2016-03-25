require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'loadcaffe'

--[[
    Creates a siamese triplet image embedder:
    - takes (query image, positive image, negative image) as input
    - passes each through a pretrained VGG net, taking the final hidden state
      as the image embedding
    - produces (inner product between query and pos embedding, inner prod between query and neg embedding)
]]--

local TripletEmbedder = {}

function TripletEmbedder.getNetwork(image_model)
	-- Set up symbolic inputs
	local query_image = nn.Identity()()
	local positive_image = nn.Identity()()
	local negative_image = nn.Identity()()
	
	-- Set up 3 separate branches of the triplet net and tie the weights
	local query_image_embedder = image_model
	local positive_image_embedder = query_image_embedder:clone('weight','bias','gradWeight','gradBias','running_mean','running_std') -- clone running mean/std in case we use batchnorm
	local negative_image_embedder = query_image_embedder:clone('weight','bias','gradWeight','gradBias','running_mean','running_std') -- clone running mean/std in case we use batchnorm

	-- Get embeddings for query, positive, and negative images
	local query_embedding = query_image_embedder(query_image)
	local positive_embedding = positive_image_embedder(positive_image)
	local negative_embedding = negative_image_embedder(negative_image)

	-- Get inner products in embedded space between (query, pos) and (query, neg)
	local query_pos_dot_prod = nn.DotProduct()({query_embedding, positive_embedding})
	local query_neg_dot_prod = nn.DotProduct()({query_embedding, negative_embedding})

	local triplet_embedder = nn.gModule({query_image, positive_image, negative_image}, {query_pos_dot_prod, query_neg_dot_prod})
	collectgarbage()
	
	print("=> Finished creating triplet embedder")

	return triplet_embedder
end

return TripletEmbedder
