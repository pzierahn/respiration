import keras


class HeartBeat(keras.callbacks.Callback):
    def __init__(self, train_gen, test_gen, args, cv_split, save_dir):
        super(HeartBeat, self).__init__()
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.args = args
        self.cv_split = cv_split
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        print('PROGRESS: 0.00%')
