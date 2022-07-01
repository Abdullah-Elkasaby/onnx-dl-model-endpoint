import tensorflow as tf
import numpy as np

class Model:
    def __init__(self , dense , imgSize , learnigRate ,modelPath) -> None:
        self.dense = dense 
        self.imgSize = imgSize 
        self.learningRate = learnigRate  
        self.model = self.InceptionResNetV2_model()
        self.model.load_weights(modelPath)

    def build_test_tfrecord(self , img ,test_tfrecord):  # Generate TFRecord of test set
        with tf.io.TFRecordWriter(test_tfrecord)as writer:
                image = img

                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    def _parse_example(self , example_string):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }

        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=3)
        feature_dict['image'] = tf.image.resize(feature_dict['image'], [224, 224]) / 255.0
        return feature_dict['image']

    def get_test_dataset(self , test_tfrecord):
        raw_test_dataset = tf.data.TFRecordDataset(test_tfrecord)
        test_dataset = raw_test_dataset.map(self._parse_example)

        return test_dataset


    def data_Preprocessing(self , test_dataset):

        test_dataset = test_dataset.batch(32)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return test_dataset

    
    def InceptionResNetV2_model(self):
        incp_res_v2 = tf.keras.applications.InceptionResNetV2(weights='imagenet',include_top=False, input_shape=[self.imgSize , self.imgSize ,3])
        incp_res_v2.trainable= True
        model = tf.keras.Sequential([
            incp_res_v2,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.dense, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = self.learningRate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )

        return model


    def test(self , test_dataset):
        predIdxs = self.model.predict(test_dataset)
        result = []
        for i in range (3):
            frist = np.argmax(predIdxs, axis=1)
            result.append(Result(frist , predIdxs[0][frist]))
            predIdxs[0][frist] = 0
        return result

    def simulation(self , img  ):
        test_tfrecord = 'test.tfrecords'
        self.build_test_tfrecord(img, test_tfrecord)
        test_dataset = self.get_test_dataset(test_tfrecord)
        test_dataset = self.data_Preprocessing(test_dataset) 
        return self.test(test_dataset ) 


        
class Result :
    def __init__(self, index , prop) :
        self.index = int(index[0])
        self.prop = float(prop[0])

    def __repr__(self):
        return f'{self.index} {self. prop}'

