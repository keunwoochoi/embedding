TR_CONST = {}
TR_CONST["dim_labels"] = 5
TR_CONST["height_image"] = 252
TR_CONST["width_image"] = 258
TR_CONST["clips_per_song"] = 3
TR_CONST["tf_type"] = 'cqt' # can be overriden
TR_CONST["isClass"] = False # can be overriden
TR_CONST["isRegre"] = True

TR_CONST["num_epoch"] = 3
# TR_CONST["num_songs"] = 300
TR_CONST["model_type"] = 'vgg_simple' # or vgg_graph, etc.

TR_CONST["loss_function"] = 'binary_crossentropy' # rmse, mse, mae, binary_crossentropy
TR_CONST["optimiser"] = 'adagrad'
TR_CONST['learning_rate'] = 3e-7

TR_CONST["num_layers"] = 5 # can be overriden
TR_CONST["num_feat_maps"] = [48]*TR_CONST["num_layers"]
TR_CONST["activations"] = ['prelu']*TR_CONST["num_layers"]
TR_CONST["dropouts"] = [0.0]*TR_CONST["num_layers"]
TR_CONST["regulariser"] = [('l2', 5e-4)]*TR_CONST["num_layers"] # use [None] not to use.
TR_CONST["BN"] = True
TR_CONST["strides"] = True
TR_CONST["debug"] = False

TR_CONST["num_fc_layers"] = 1
TR_CONST["dropouts_fc_layers"] = [0.0]*TR_CONST["num_fc_layers"]
TR_CONST["nums_units_fc_layers"] = [512]*TR_CONST["num_fc_layers"]
TR_CONST["activations_fc_layers"] = ['prelu']*TR_CONST["num_fc_layers"]
# TR_CONST["regulariser_fc_layers"] = [('l2', 0.0007)]*TR_CONST["num_fc_layers"]
TR_CONST["regulariser_fc_layers"] = [('l2', 5e-4), ('l2', 5e-4)]
TR_CONST["BN_fc_layers"] = True 
TR_CONST["output_activation"] = 'sigmoid'
TR_CONST["!memo"] = ''
TR_CONST["is_test"] = False

TR_CONST["conv_mode"] = '2d'