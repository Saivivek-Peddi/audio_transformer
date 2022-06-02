from preprocess import Preprocess

pre = Preprocess()
# pre.concurrent_prepare_audio_for_training()
pre.create_training_data()
# pre.get_age()