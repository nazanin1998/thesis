import tensorflow as tf
from tensorflow_addons.layers import CRF

from lib.utils.constants import PHEME_LABEL_COL_NAME
from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bilstm.bilstm_configurations import BI_LSTM_BATCH_SIZE, BI_LSTM_EPOCH, BI_LSTM_OPTIMIZER_NAME, \
    BI_LSTM_LEARNING_RATE, BI_LSTM_DROPOUT_RATE, BI_LSTM_BUFFER_SIZE
from lib.utils.log.logger import log_start_phase, log_phase_desc, log_end_phase, log_line


class BiLstmModelImpl:
    def __init__(self,
                 train_tensor_dataset,
                 val_tensor_dataset,
                 test_tensor_dataset,
                 # num_classes,
                 train_len,
                 validation_len,
                 test_len,
                 preprocess_model,
                 train_df,
                 val_df,
                 test_df,
                 ):
        self.__label_feature_name = PHEME_LABEL_COL_NAME
        self.__training_feature_names = ['text', ]

        self.__train_tensor_dataset = train_tensor_dataset
        self.__val_tensor_dataset = val_tensor_dataset
        self.__test_tensor_dataset = test_tensor_dataset
        self.__train_len = train_len
        self.__validation_len = validation_len
        self.__test_len = test_len
        self.__train_df = train_df
        self.__test_df = test_df
        self.__val_df = val_df
        self.__preprocess_model = preprocess_model

        self.__loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.__metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

        self.__steps_per_epoch = self.__train_len // BI_LSTM_BATCH_SIZE
        self.__validation_steps = self.__validation_len // BI_LSTM_BATCH_SIZE
        self.__num_train_steps = self.__steps_per_epoch * BI_LSTM_EPOCH
        self.__num_warmup_steps = self.__num_train_steps // 10

    @staticmethod
    def get_optimizer():
        if BI_LSTM_OPTIMIZER_NAME == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=BI_LSTM_LEARNING_RATE)
        elif BI_LSTM_OPTIMIZER_NAME == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=BI_LSTM_LEARNING_RATE)
        elif BI_LSTM_OPTIMIZER_NAME == "adamax":
            return tf.keras.optimizers.Adamax(learning_rate=BI_LSTM_LEARNING_RATE)
        elif BI_LSTM_OPTIMIZER_NAME == "adadelta":
            return tf.keras.optimizers.Adadelta(learning_rate=BI_LSTM_LEARNING_RATE)
        elif BI_LSTM_OPTIMIZER_NAME == "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=BI_LSTM_LEARNING_RATE)

    def __fit_model(self, model, x_train, y_train, x_val, y_val):
        # history = model.fit(
        #     x=self.__train_tensor_dataset,
        #     batch_size=BI_LSTM_BATCH_SIZE,
        #     validation_data=self.__val_tensor_dataset,
        #     steps_per_epoch=self.__steps_per_epoch,
        #     epochs=BI_LSTM_EPOCH,
        #     validation_steps=self.__validation_steps,
        # )

        # print(x_train.shape)
        # # x_train.reshape(-1, 2, 77)
        # numpy.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1] ))
        # # tf.keras.layers.Reshape((128, 1,))(model)
        # print(x_train.shape)
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            batch_size=BI_LSTM_BATCH_SIZE, epochs=BI_LSTM_EPOCH
        )
        return history

    def __get_x_y_from_df(self, df):
        y = df[self.__label_feature_name]
        x = df[self.__training_feature_names]
        return x, y

    def start(self):
        log_start_phase(3, 'BI_LSTM MODEL STARTED')
        # log_phase_desc(f'Preprocess sequence len  : {PREPROCESS_SEQ_LEN}')
        # log_phase_desc(f'Preprocess batch size    : {PREPROCESS_BATCH_SIZE}')
        log_phase_desc(f'Preprocess buffer size   : {BI_LSTM_BUFFER_SIZE}')
        # log_phase_desc(f'Do shuffle on splitting  : {PREPROCESS_DO_SHUFFLING}')
        log_phase_desc(f'BI_LSTM batch size          : {BI_LSTM_BATCH_SIZE}')
        log_phase_desc(f'BI_LSTM epochs              : {BI_LSTM_EPOCH}')
        log_phase_desc(f'BI_LSTM dropout rate        : {BI_LSTM_DROPOUT_RATE}')
        log_phase_desc(f'BI_LSTM learning rate       : {BI_LSTM_LEARNING_RATE}')
        log_phase_desc(f'BI_LSTM optimizer           : {BI_LSTM_OPTIMIZER_NAME}')
        x_train, y_train = self.__get_x_y_from_df(self.__train_df)
        x_test, y_test = self.__get_x_y_from_df(self.__test_df)
        x_val, y_val = self.__get_x_y_from_df(self.__val_df)
        # x_train = tf.convert_to_tensor(x_train)
        # y_train = tf.cast(y_train, dtype=tf.float32)
        print('type(y_train)')
        print(type(y_train))
        # print((y_train.value_counts()))
        # classifier_model = self.build_classifier_model(num_classes=2, allow_cudnn_kernel=True)
        # print(self.__train_tensor_dataset)
        # preprocess_layers = []
        # print("x_train.items()")
        # print(x_train.items())
        # for item in x_train.items():
        #     text_input = tf.keras.Input(shape=(), dtype=tf.string)
        #     encoder = tf.keras.layers.TextVectorization()
        #     encoder.adapt(x_train[item])(text_input)
        #     # preprocessed_item = encoder(text_input)
        #     # preprocessed_item = tf.cast(preprocessed_item, dtype=tf.float32, name=f"vectorization_{item}")
        #     preprocess_layers.append(preprocessed_item)
        text_input = tf.keras.Input(shape=(None, ), )

        word_embedding_size = 128
        #
        model = tf.keras.layers.Embedding(input_dim=100000, output_dim=word_embedding_size,
                                          input_length=200)(text_input)
        # model = tf.cast(preprocessed_item, dtype=tf.float32)(model)
        # model = tf.keras.layers.Reshape((128, 1,))(model)
        model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=word_embedding_size,
                                                                   return_sequences=True,
                                                                   dropout=0.5,
                                                                   recurrent_dropout=0.5,
                                                                   # kernel_initializer=k.initializers.he_normal()
                                                                   ))(model)
        model = tf.keras.layers.LSTM(units=word_embedding_size * 2,
                                     return_sequences=True,
                                     dropout=0.5,
                                     recurrent_dropout=0.5, )(model)
        model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation="relu"))(
            model)  # previously softmax output layer

        crf = CRF(2)  # CRF layer
        out = crf(model)  # output
        model = tf.keras.Model(text_input, out)

        adam = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')

        # return;
        # classifier_model = tf.keras.Sequential()
        #
        # classifier_model.add(Reshape((120, 2048), input_shape=(2048,)))
        #
        # classifier_model.add(
        #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(BI_LSTM_BATCH_SIZE, return_sequences=True),
        #                                   input_shape=(None, BI_LSTM_BATCH_SIZE))
        # )
        # # classifier_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(BI_LSTM_BATCH_SIZE // 2)))
        # classifier_model.add(tf.keras.layers.Dense(10))
        #
        # #
        # #
        #
        # classifier_model.compile(
        #     optimizer=self.get_optimizer(),
        #     loss=self.__loss, metrics=[self.__metrics])
        model.summary()

        history = self.__fit_model(model, x_train, y_train, x_val, y_val)

        # classifier_model.summary()

        self.save_model(classifier_model=model)

        analyser = BertModelAnalysis(model=model, history=history)
        analyser.plot_bert_model()
        acc, val_acc, loss, val_loss, first_loss, accuracy = analyser.evaluation(
            test_tensor_dataset=self.__test_tensor_dataset)

        analyser.plot_bert_evaluation_metrics(
            acc=acc,
            val_acc=val_acc,
            loss=loss,
            val_loss=val_loss)

        log_end_phase(3, 'BERT ON TWEET TEXT')
        log_line()

    @staticmethod
    def get_save_model_path():
        return os.path.join(BERT_SAVE_MODEL_DIR, BERT_SAVE_MODEL_NAME)

    def save_model(self, classifier_model):
        saved_model_path = self.get_save_model_path()
        # preprocess_inputs = self.__bert_preprocess_model.inputs
        # bert_encoder_inputs = self.__bert_preprocess_model(preprocess_inputs)
        # bert_outputs = classifier_model(bert_encoder_inputs)

        # model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)
        classifier_model.save(saved_model_path, include_optimizer=True)
        # model_for_export.save(saved_model_path, include_optimizer=True)
        print(f'SAVE MODEL (PATH): {saved_model_path}')

    def build_classifier_model(self_out, num_classes, allow_cudnn_kernel=True):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes):
                super(Classifier, self).__init__(name="prediction")
                # input_dim = ()
                # units = 64
                # self.bi_lstm = tf.keras.layers.Bidirectional(
                #     tf.keras.layers.LSTM(units, return_sequences=True, input_shape=(BI_LSTM_BATCH_SIZE, None, units)),
                #     input_shape=(BI_LSTM_BATCH_SIZE, None, units))
                # # self.encoder = hub.KerasLayer(self_out.bert_model_name, trainable=True)
                # self.dropout = tf.keras.layers.Dropout(BI_LSTM_DROPOUT_RATE)
                # self.dense = tf.keras.layers.Dense(num_classes)

            def call(self, preprocessed_text):
                print("called")
                print(preprocessed_text)
                bi_lstm_outputs = self.bi_lstm(preprocessed_text)
                print(bi_lstm_outputs)
                # encoder_outputs = self.encoder(preprocessed_text)
                pooled_output = bi_lstm_outputs["pooled_output"]
                x = self.dropout(pooled_output)
                x = self.dense(x)

                # model = tf.keras.models.Sequential(
                #     [
                #         self.bi_lstm,
                #         keras.layers.BatchNormalization(),
                #         keras.layers.Dense(nu),
                #     ]
                # )
                # if allow_cudnn_kernel:
                #     #                 # The LSTM layer with default options uses CuDNN.
                #     lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
                # else:
                #                 # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
                #                 lstm_layer = keras.layers.RNN(
                #                     keras.layers.LSTMCell(units), input_shape=(None, input_dim)
                #                 )
                #             model = keras.models.Sequential(
                #                 [
                #                     lstm_layer,
                #                     keras.layers.BatchNormalization(),
                #                     keras.layers.Dense(output_size),
                #                 ]
                #             )
                #             return model

                # model.add(
                #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True),
                #                                   input_shape=input_dim)
                # )
                # model.add(layers.Bidirectional(layers.LSTM(32)))
                # model.add(layers.Dense(10))

                # model.summary()
                return x

        model = Classifier(num_classes)
        # batch_size = 64
        # # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
        # # Each input sequence will be of size (28, 28) (height is treated like time).
        # input_dim = 28
        #
        # units = 64
        # output_size = 10  # labels are from 0 to 9
        #
        # # Build the RNN model
        # def build_model(allow_cudnn_kernel=True):
        #     # CuDNN is only available at the layer level, and not at the cell level.
        #     # This means `LSTM(units)` will use the CuDNN kernel,
        #     # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.

        return model
