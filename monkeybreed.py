from keras.applications import MobileNet
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

img_rows, img_cols = 224,224

mobilenet = MobileNet(
    weights="imagenet",
    include_top= False,
    input_shape= (img_rows, img_cols,3)
)

# Freezing the low level layers of the cnn.
for layer in mobilenet.layers:
    layer.trainable = False

# for (i, layer) in enumerate(mobilenet.layers):
#     print(str(i)+" "+ layer.__class__.__name__, layer.trainable)


# Creating the new top model.
def newTopModel(bottomModel, num_classes):

    top_model = bottomModel.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation="relu")(top_model)
    top_model = Dense(1024, activation="relu")(top_model)
    top_model = Dense(512, activation="relu")(top_model)
    top_model = Dense(num_classes, activation="softmax")(top_model)
    return top_model

num_classes = 10

FC_head = newTopModel(mobilenet, num_classes)
model = Model(inputs = mobilenet.input, outputs = FC_head)

# print(model.summary())

train_dir = "./monkey_breed/train"
test_dir = "./monkey_breed/validation"

train_datagen = ImageDataGenerator(
    rescale=1. /255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1. /255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (img_rows, img_cols),
    batch_size= batch_size,
    class_mode= "categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size= (img_rows, img_cols),
    batch_size= batch_size,
    class_mode= "categorical"
)

modelcheckpoint = ModelCheckpoint(
    "monkey.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

earlystopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

callbacks = [modelcheckpoint, earlystopping]

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics = ["accuracy"])

nb_train = 1097 
nb_test = 272

batch_size = 16
epochs = 5

model.fit(
    train_generator,
    steps_per_epoch= nb_train // batch_size,
    epochs=epochs,
    callbacks= callbacks,
    validation_data= test_generator,
    validation_steps= nb_test // batch_size
)