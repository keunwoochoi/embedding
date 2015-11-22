
SR = 22050
N_FFT = 1024
WIN_LEN = 1024
HOP_LEN = 512 # 11 sec --> 512 frames


FILE_DICT = {}
FILE_DICT["id_path"] = "id_path_dict_w_audio.cP" #dict
FILE_DICT["moodnames"] = "moodnames.cP" #list
FILE_DICT["track_ids"] = "track_ids_w_audio.cP" #list
FILE_DICT["mood_tags_matrix"] = "mood_tags_matrix_w_audio.npy" # matrix, 9320-by-100
FILE_DICT["mood_latent_matrix"] = "mood_latent_matrix_w_audio_%d.npy"
FILE_DICT["mood_embeddings"] = "moodname_embeddings.cP" # dictionary