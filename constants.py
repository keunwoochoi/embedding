
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

FILE_DICT["mood_tags_tfidf_matrix"]   = "mood_tags_matrix_tfidf_w_audio.npy" # matrix, 9320-by-100
FILE_DICT["mood_latent_tfidf_matrix"] = "mood_latent_tfidf_matrix_w_audio_%d.npy" # THIS IS NORMALISED ONE
# FILE_DICT["mood_latent_tfidf_matrix_nor"] = "mood_latent_tfidf_matrix_w_audio_%d_normalised.npy" # same as above but normalised to be a unit length.
FILE_DICT["mood_topics_strings"] = "mood_%d_topics.cP" # list of list of topic words.

FILE_DICT["mood_embeddings"] = "moodname_embeddings.cP" # dictionary
FILE_DICT["sentiment_big_dict"] = "sentiment_dictionary.cP" # big one
FILE_DICT["mood_sentiment"] = "sentiment_mood_dictionary.cP" # Mood_Sentiment class, only has my mood lists <100

CQT_CONST = {}
CQT_CONST["hop_len"] = 512
CQT_CONST["num_octaves"] = 7
CQT_CONST["bins_per_note"] = 3
CQT_CONST["bins_per_octave"] = CQT_CONST["bins_per_note"]*12
CQT_CONST["sr"] = SR
CQT_CONST["n_bins"] = CQT_CONST["bins_per_octave"]*CQT_CONST["num_octaves"]

