import json
with open('model_id.json') as data_file:
	data = json.load(data_file)
for idx in sorted([int(i) for i  in list(data['loss_history'])]):
  print(str(idx)+":     "+str(data['loss_history'][str(idx)]))