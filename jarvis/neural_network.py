from text_analysis import load_data, feature_similarity
import tensorflow as tf
from tensorflow import keras

# load in user
user1_data, user2_data = load_data("j2lam@ucsd.edu", "jerometlam47@gmail.com")

# grab the combination we are going to use to train the neural network as well as how similar the answers are
user1_features = user1_data[4::]
user2_features = user2_data[4::]

combination, similarity = feature_similarity(user1_features, user2_features)
#print(len(combination)) #484
#print(len(similarity))  #484

# if greater than one hundred points train model otherwise always match

def build_model(similarities):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1,similarities)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

model = build_model(len(similarity))
model.compile(optimizer = "adam")
model.summary()

#match = Model.Predict(similarity)
print("Hey This Works!")
