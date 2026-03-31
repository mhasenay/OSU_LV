import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a) prikazi podatke u ravnini
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], cmap = 'bwr', c='red')
plt.scatter(X_test[:,0], X_test[:,1], cmap = 'bwr', c='blue', marker = 'x')
plt.show()

#b) napravi leg.reg
LogRegression_model = LogisticRegression()
LogRegression_model.fit( X_train , y_train )

#c)
param1 = LogRegression_model.intercept_[0]
w1, w2 = LogRegression_model.coef_.T
c = -param1/w2
m = -w1/w2
xmin, xmax = -5, 5
ymin, ymax = -5, 5
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='magenta', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='pink', alpha=0.2)
plt.show()

#d)
y_test_p = LogRegression_model.predict( X_test )

disp = ConfusionMatrixDisplay( confusion_matrix(y_test , y_test_p ))
disp.plot(cmap = 'PuRd')
plt.show()
print(classification_report(y_test , y_test_p))

#e)
y1 = (y_test == y_test_p)
y2 = (y_test != y_test_p)

X_false = []

for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        X_false.append([X_test[i,0], X_test[i,1]])

X_false = np.array(X_false)

plt.scatter(X_train[:,0], X_train[:,1], c='green')
plt.scatter(X_false[:,0], X_false[:,1], c='black', )
plt.show()