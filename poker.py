from sklearn import svm
import csv

"""
Assumption:
    I don't know anything about the game. Hence everything I can do is just looking *carefully* at the cards, extracting features from them.
Features:
    X = [[# of cards of shape x], [# of cards of number x], [# of cards of shape,number (x1,x2)]]
    (...first two seems redundant but they gives me fairly good score)
"""

def LoadData(filename):
    rdr = csv.DictReader(open(filename, 'r'))
    ret = []
    for row in rdr:
        ret.append(row)
    return ret

def FitModel(X, y):
    model = svm.SVC()
    model.fit(X, y)
    return model

def PredictModel(model, X):
    return model.predict(X)

def ParseCards(row):
    ret = []
    for i in xrange(1,6):
        shape = "S" + str(i)
        value = "C" + str(i)
        sh = int(row[shape]) - 1
        va = int(row[value]) - 1
        ret.append((sh,va))
    return ret

def ExtractFeautre(data):
    X = []
    for row in data:
        by_number = [0] * 13
        by_shape = [0] * 4
        by_card = [0] * 52
        for card in ParseCards(row):
            by_shape[card[0]] += 1
            by_number[card[1]] += 1
            by_card[card[0]*13+card[1]] += 1
        X.append(by_number + by_shape + by_card)
    return X

def ExtractLabel(data):
    y = []
    for row in data:
        y.append(row['hand'])
    return y

model = FitModel(ExtractFeautre(LoadData('./train.csv')), ExtractLabel(LoadData('./train.csv')))
answer = PredictModel(model, ExtractFeautre(LoadData('./test.csv')))

print 'id,hand'
for i in xrange(len(answer)):
    print str(i+1) + "," + str(answer[i])
