TR_CONST = {}
TR_CONST["dim_labels"] = 8
TR_CONST["height_image"] = 252
TR_CONST["width_image"] = 258
TR_CONST["clips_per_song"] = 3
TR_CONST["tf_type"] = 'cqt' # can be overriden
TR_CONST["isClass"] = False # can be overriden
TR_CONST["isRegre"] = True

TR_CONST["num_epoch"] = 30
# TR_CONST["num_songs"] = 300
TR_CONST["model_type"] = 'vgg_sequential' # or vgg_graph, etc.

TR_CONST["loss_function"] = 'rmse' # rmse, mse, mae, binary_crossentropy
TR_CONST["optimiser"] = 'rmsprop'

TR_CONST["num_layers"] = 5 # can be overriden
TR_CONST["num_feat_maps"] = [48]*TR_CONST["num_layers"]
TR_CONST["activations"] = ['relu']*TR_CONST["num_layers"]
TR_CONST["dropouts"] = [0.0]*TR_CONST["num_layers"]
TR_CONST["regulariser"] = [('l2', 0.0002)]*TR_CONST["num_layers"] # use [None] not to use.

TR_CONST["num_fc_layers"] = 2
TR_CONST["dropouts_fc_layers"] = [0.0]*TR_CONST["num_fc_layers"]
TR_CONST["nums_units_fc_layers"] = [512]*TR_CONST["num_fc_layers"]
TR_CONST["activations_fc_layers"] = ['relu']*TR_CONST["num_fc_layers"]
TR_CONST["regulariser_fc_layers"] = [('l2', 0.0002)]*TR_CONST["num_fc_layers"]

# TODO: change this to a Clas