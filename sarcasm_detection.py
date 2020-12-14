import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras import layers, models, optimizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost
from scipy.sparse import hstack

# Function to load the test or training data into a pandas dataframe object
def get_pd(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = pd.read_json(F'{dir_path}/data/{name}.jsonl',lines = True)
    return f

# Here is the preprocessing and cleaning done to the dataset
def preprocess_text(df, cleaning=False):
    # concatenate all strings in the context list
    df["context"] = df["context"].apply(lambda x: " ".join(x))

    # optional data cleaning step(actually worsen the performance)
    if cleaning:
        punctuation_signs = list("?:!.,;")
        lemmatizer = WordNetLemmatizer()
        stop_words = list(stopwords.words('english'))
        columns = ['response', 'context']
        for col in columns:
            new_name = col
            df[new_name] = df[col].str.replace("\r", " ")
            df[new_name] = df[new_name].str.replace("\n", " ")
            df[new_name] = df[new_name].str.replace("    ", " ")
            df[new_name] = df[new_name].str.replace('"', '')
            df[new_name] = df[new_name].str.lower()

            #signs
            for punct_sign in punctuation_signs:
                df[new_name] = df[new_name].str.replace(punct_sign, '')
            df[new_name] = df[new_name].str.replace("'s", "")

            result_list = []
            # lemmatization and stemming
            for row in range(0, len(df)):
                lemmatized_list = []
                text = df.loc[row][new_name]
                text_words = text.split(" ")
                for word in text_words:
                    lemmatized_list.append(lemmatizer.lemmatize(word, pos="v"))
                lemmatized_text = " ".join(lemmatized_list)
                result_list.append(lemmatized_text)
            #stop words
            df[new_name] = result_list
            for stop_word in stop_words:
                regex_sw = r"\b" + stop_word + r"\b"
                df[new_name] = df[new_name].str.replace(regex_sw, '')
    return df

def encode_label(df):
    # label encoding, not necessary
    classes = {'NOT_SARCASM': 0, 'SARCASM': 1}
    return df.replace({'label': classes})

def get_tfidf_features():
    # returns an sklearn tfidf feature tokenizer object
    ngram_range = (1,2)
    return TfidfVectorizer(ngram_range=ngram_range)

def get_data_label(name):
    #helper function for label processing
    df = get_pd(name)
    preprocess_text(df)
    df = encode_label(df)

    tfidf = get_tfidf_features()
    X = df['parsed_response']
    y = df['binary_label']
    features = tfidf.fit_transform(X).toarray()
    return features, y

# funcition for running SVM classification on the test set by training first using training data
def run_svm():
    #prepare training data
    df_train = get_pd('train')
    preprocess_text(df_train)
    df_train = encode_label(df_train)

    # get tfidf vector
    tfidf = get_tfidf_features()
    X_train = df_train['response']
    labels_train = df_train['label']
    features_train = tfidf.fit_transform(X_train).toarray()

    #prepare test data
    df_test = get_pd('test')
    preprocess_text(df_test)
    X_test = df_test['response']
    features_test = tfidf.transform(X_test).toarray()

    # train svm classifier
    svc = svm.SVC(random_state=8, kernel='linear', C=0.1, probability=True)
    print(svc)
    svc.fit(features_train, labels_train)

    #predict test result
    svc_pred = svc.predict(features_test)
    print("The training accuracy is: ")
    print(accuracy_score(labels_train, svc.predict(features_train)))

    #write predicted output to Answer.txt
    output = ["{},{}".format(df_test['id'][i], 'SARCASM' if svc_pred[i] == 1 else 'NOT_SARCASM') for i in range(len(svc_pred))]
    with open('answer.txt', 'w') as f:
        print('\n'.join(output), file=f)
    f.close()

# funcition for running MLP classification on the test set by training first using training data
def run_MLP():
    # prepare training data
    df_train = get_pd('train')
    preprocess_text(df_train)
    df_train = encode_label(df_train)

    # get tfidf vector
    tfidf = get_tfidf_features()
    X_train = df_train['response']
    labels_train = df_train['label']
    features_train = tfidf.fit_transform(X_train).toarray()

    # prepare test data
    df_test = get_pd('test')
    preprocess_text(df_test)
    X_test = df_test['response']
    features_test = tfidf.transform(X_test).toarray()

    #define network input size
    classifier = create_mlp_model(500)

    classifier.fit(features_train, labels_train)

    # predict the labels on validation dataset
    predictions = classifier.predict(features_test)
    # predictions = predictions.argmax(axis=-1)

    output = ["{},{}".format(df_test['id'][i], 'SARCASM' if predictions[i] >= 0.5 else 'NOT_SARCASM') for i in range(len(predictions))]
    with open('answer.txt', 'w') as f:
        print('\n'.join(output), file=f)
    f.close()


def create_mlp_model(input_size):
    # create input layer
    input_layer = layers.Input((input_size,), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(512, activation="relu")(input_layer)
    hidden_layer = layers.Dense(256, activation="relu")(hidden_layer)
    hidden_layer = layers.Dense(64, activation="relu")(hidden_layer)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier

# XG boost training and predicting function similar to above functions
def run_boost():
    df_train = get_pd('train')
    preprocess_text(df_train)
    # df_train = encode_label(df_train)

    tfidf = get_tfidf_features()
    X_train = df_train['response']
    labels_train = df_train['label']
    features_train = tfidf.fit_transform(X_train).tocsc()

    df_test = get_pd('test')
    preprocess_text(df_test)
    X_test = df_test['response']
    features_test = tfidf.transform(X_test).tocsc()

    model = xgboost.XGBClassifier()

    model.fit(features_train, labels_train)
    rf_pred = model.predict_proba(features_test)
    print("The training accuracy is: ")
    print(accuracy_score(labels_train, model.predict(features_train)))
    print(rf_pred)

    threshhold = 0.5
    output = ["{},{}".format(df_test['id'][i], 'SARCASM' if rf_pred[i][1] >= threshhold else 'NOT_SARCASM') for i in range(len(rf_pred))]
    with open('answer.txt', 'w') as f:
        print('\n'.join(output), file=f)
    f.close()


# main model that allows training of different models on the dataset
# model: a classifier model object to train and test on
def run_model(model):
    #prepare training data
    df_train = get_pd('train')
    preprocess_text(df_train, cleaning=False)
    # df_train = encode_label(df_train)

    #get tf-idf vector
    labels_train = df_train['label']
    tfidf_response = get_tfidf_features()
    tfidf_context = get_tfidf_features()

    #concatenate both response and context tfidf vectors
    response_train = tfidf_response.fit_transform(df_train['response'])
    context_train = tfidf_context.fit_transform(df_train['context'])
    features_train = hstack([response_train, context_train])

    #prepare test data
    df_test = get_pd('test')
    preprocess_text(df_test, cleaning=False)

    response_test = tfidf_response.transform(df_test['response'])
    context_test = tfidf_context.transform(df_test['context'])
    features_test = hstack([response_test, context_test])

    #train the model and predict test set result
    model.fit(features_train, labels_train)
    rf_pred = model.predict_proba(features_test)
    print("The training accuracy is: ")
    print(accuracy_score(labels_train, model.predict(features_train)))
    print(rf_pred)

    # apply decision threshold on the test data set and write to output test file
    threshold = 0.465
    print(threshold)
    output = ["{},{}".format(df_test['id'][i], 'SARCASM' if rf_pred[i][1] > threshold else 'NOT_SARCASM') for i in range(len(rf_pred))]
    with open('answer.txt', 'w') as f:
        print('\n'.join(output), file=f)
    f.close()

# define svm model
svm = svm.SVC(random_state=8, kernel='linear', C=0.1, probability=True)

# define random forest model
rf_model = RandomForestClassifier(n_estimators=1000, random_state=17, warm_start=True, verbose=1)

# specify which model to use
model = rf_model

# run svm model
# run_svm()

# run mlp model
# run_MLP()

# run xgboost boosting model
# run_boost()

# run random forest or svm model here
run_model(model)

