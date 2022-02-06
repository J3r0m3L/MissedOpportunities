import numpy as np
import spacy
import csv

def load_data(user1, user2):
    with open("user_profiles.csv", encoding="utf8") as csv_files:
        csv_reader = csv.reader(csv_files)
        for line in csv_reader:
            if (user1 == line[0]):
                user1_data = (line)
            if (user2 == line[0]):
                user2_data = (line)

    return user1_data, user2_data

def feature_similarity(user1_features, user2_features):
    nlp = spacy.load("en_core_web_lg")

    similarities = np.zeros(484)
    #user1_notNull = np.zeros(10)
    #user2_notNull = np.zeros(10)

    user1_notNull = np.zeros(22)
    user2_notNull = np.zeros(22)

    counter = 0
    for index, item in enumerate(user1_features):
        if (item != 'Null'): # Need to Fix This
            #user1_notNull[counter] = index
            user1_notNull[index] = index
            counter += 1
        else:
            user1_notNull[index] = -1
            
    counter = 0
    for index, item in enumerate(user2_features):
        if (item != 'Null'): # Need to Fix This 
            #user2_notNull[counter] = index
            user2_notNull[index] = index
            counter += 1
        else:
            user2_notNull[index] = -1
    
    combinations = np.array(np.meshgrid([user1_notNull], 
                                        [user2_notNull])).T.reshape(-1, 2)

    for index, item in enumerate(combinations):
        if int(item[0]) == -1 or int(item[1]) == -1:
            similarities[index] = 0
        else:
            feature1 = nlp(user1_features[int(item[0])])
            feature2 = nlp(user2_features[int(item[1])])
            similarities[index] = feature1.similarity(feature2)

    return combinations, similarities