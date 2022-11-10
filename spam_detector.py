import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
#for svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
#for multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
#for logistic regression
from sklearn.linear_model import LogisticRegression


spam = pd.read_csv("D:\Chan's Stuff\Coding Stuff\Python\AI+ML\spam.csv")

#z = spam['EmailText'] assigns the column EmailText from spam to z
#y = spam["Label"] assigns the column Label from spam to y
z = spam['v2']
y = spam["v1"]

#The function divides columns z and y into z_train for training inputs, 
# y_train for training labels, z_test for testing inputs, and y_test for testing labels.
#test_size=0.2 sets the testing set to 20 percent of z and y
z_train, z_test, y_train, y_test = train_test_split(z,y,test_size = 0.2, random_state=0) 

print("SVM Model training...")

#CountVectorizer() and the features function randomly assigns a number to each word
#then it counts the number of occurrences of words and saves it to cv
cv = CountVectorizer()
features = cv.fit_transform(z_train)

SVMmodel = svm.SVC()
SVMmodel.fit(features,y_train)

features_test = cv.transform(z_test)



print("Accuracy of SVM Model:", format(SVMmodel.score(features_test,y_test)))
print("SVM training complete.")


#PART 2 : Multinomial Naive Bayes

print("\nMultinomial Naive Bayes Model training...\n")

z_train = cv.transform(z_train)
z_test = cv.transform(z_test)

MNBmodel = MultinomialNB()
MNBmodel.fit(z_train, y_train)

predictionMNB = MNBmodel.predict(z_train)
#print(classification_report(y_train ,predictionMNB ))
print('Multinomial Naive Bayes Confusion Matrix: \n',confusion_matrix(y_train, predictionMNB))
print('Accuracy of Multinomial Naive Bayes Model: ', accuracy_score(y_train,predictionMNB))
print("Multinomial Naive Bayes training complete.")

#PART 3 : Logistic Regression

print("\nLogistic Regression Model training...\n")

z_train, z_test, y_train, y_test = train_test_split(z,y,test_size = 0.2, random_state=1)

training_data = cv.fit_transform(z_train).toarray()
testing_data = cv.transform(z_test).toarray()

LRmodel = LogisticRegression(random_state = 0).fit(training_data, y_train)

predictionLR = LRmodel.predict(testing_data)

print('\nLogistic Regression Confusion Matrix :\n', confusion_matrix(y_test, predictionLR))
print('Accuracy score of Logistic Regression Model: ', format(accuracy_score(y_test, predictionLR)))
print("Logistic Regression training complete.")
