#%%
import numpy as np

np.random.seed(42)

train_set = np.load('../data/train_set.npy', allow_pickle=True)
test_set = np.load('../data/test_set.npy', allow_pickle=True)

#%%
train_set_len = len(train_set)
data_frame_len = len(train_set[0][0])
data_channel_len = len(train_set[0][0][0])
x_train = np.empty((train_set_len, data_frame_len, data_channel_len))
y_train = np.empty((train_set_len))
for idx in range(train_set_len):
    x_train[idx] = train_set[idx, 0]
    y_train[idx] = train_set[idx, 1]

xn_0 = x_train

test_set_len = len(test_set)
x_test = np.empty((test_set_len, data_frame_len, data_channel_len))
y_test = np.empty((test_set_len))
for idx in range(test_set_len):
    x_test[idx] = test_set[idx, 0]
    y_test[idx] = test_set[idx, 1]

#%%
from sklearn.base import TransformerMixin
import numpy as np
import scipy.stats as stats

# roor mean square
def rms(x):
  x = np.array(x)
  return np.sqrt(np.mean(np.square(x)))
# square root amplitude
def sra(x):
  x = np.array(x)
  return np.mean(np.sqrt(np.absolute(x)))**2
# peak to peak value
def ppv(x):
  x = np.array(x)
  return np.max(x)-np.min(x)
# crest factor
def cf(x):
  x = np.array(x)
  return np.max(np.absolute(x))/rms(x)
# impact factor
def ifa(x):
  x = np.array(x)
  return np.max(np.absolute(x))/np.mean(np.absolute(x))
# margin factor
def mf(x):
  x = np.array(x)
  return np.max(np.absolute(x))/sra(x)
# shape factor
def sf(x):
  x = np.array(x)
  return rms(x)/np.mean(np.absolute(x))
# kurtosis factor
def kf(x):
  x = np.array(x)
  return stats.kurtosis(x)/(np.mean(x**2)**2)

class StatisticalTime(TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    if X.shape[2] == 1:
      return np.array([[rms(x), sra(x), stats.kurtosis(x), stats.skew(x), ppv(x), cf(x), ifa(x), mf(x), sf(x), kf(x)] for x in X[:,:,0]])
    de = np.array([[rms(x), sra(x), stats.kurtosis(x), stats.skew(x), ppv(x), cf(x), ifa(x), mf(x), sf(x), kf(x)] for x in X[:,:,0]])
    fe = np.array([[rms(x), sra(x), stats.kurtosis(x), stats.skew(x), ppv(x), cf(x), ifa(x), mf(x), sf(x), kf(x)] for x in X[:,:,1]])
    return np.concatenate((de,fe),axis=1)
  
class StatisticalFrequency(TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    if X.shape[2] == 1:
      sig = []
      for x in X[:,:,0]:
        fx = np.absolute(np.fft.fft(x))
        fc = np.mean(fx)
        sig.append([fc, rms(fx), rms(fx-fc)])
      return np.array(sig)
    de = []
    for x in X[:,:,0]:
      fx = np.absolute(np.fft.fft(x))
      fc = np.mean(fx)
      de.append([fc, rms(fx), rms(fx-fc)])
    de = np.array(de)
    fe = []
    for x in X[:,:,1]:
      fx = np.absolute(np.fft.fft(x))
      fc = np.mean(fx)
      fe.append([fc, rms(fx), rms(fx-fc)])
    fe = np.array(fe)
    return np.concatenate((de,fe),axis=1)

class Statistical(TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    st = StatisticalTime()
    stfeats = st.transform(X)
    sf = StatisticalFrequency()
    sffeats = sf.transform(X)
    return np.concatenate((stfeats,sffeats),axis=1)
   
import pywt
class WaveletPackage(TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    def Energy(coeffs, k):
      return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])
    def getEnergy(wp):
      coefs = np.asarray([n.data for n in wp.get_leaf_nodes(True)])
      return np.asarray([Energy(coefs,i) for i in range(2**wp.maxlevel)])
    if X.shape[2] == 1:
      return np.array([getEnergy(pywt.WaveletPacket(data=x, wavelet='db4', mode='symmetric', maxlevel=4)) for x in X[:,:,0]])
    de = np.array([getEnergy(pywt.WaveletPacket(data=x, wavelet='db4', mode='symmetric', maxlevel=4)) for x in X[:,:,0]])
    fe = np.array([getEnergy(pywt.WaveletPacket(data=x, wavelet='db4', mode='symmetric', maxlevel=4)) for x in X[:,:,1]])
    return np.concatenate((de,fe),axis=1)

class Heterogeneous(TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    st = StatisticalTime()
    stfeats = st.transform(X)
    sf = StatisticalFrequency()
    sffeats = sf.transform(X)
    wp = WaveletPackage()
    wpfeats = wp.transform(X)
    return np.concatenate((stfeats,sffeats,wpfeats),axis=1)

from keras import backend as K
def f1_score_macro(y_true,y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#%%
from keras import layers
from keras import Input
from keras.models import Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.utils import to_categorical

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
class ANNConv1D(BaseEstimator, ClassifierMixin):
  def __init__(self, filters=16, kernel_size=22, shape=xn_0.shape):
    self.shape = shape
    self.filters = filters
    self.kernel_size = kernel_size

  def fit(self, X, y=None):
    y_cat = to_categorical(y)
    signal_input = Input(shape=(self.shape[1],self.shape[-1]), dtype='float32', name='signal')
    x = layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', name='conv1d_1')(signal_input)
    x = layers.MaxPooling1D(self.kernel_size, name='max_pooling1d_1')(x)
    x = layers.Flatten(name='flatten')(x)
    condition_output = layers.Dense(7,activation='softmax',name='condition')(x)
    self.model = Model(signal_input, condition_output) 
    self.model.compile(optimizer='rmsprop',
                       loss='mean_squared_error', 
                       metrics=['accuracy',f1_score_macro])
      
    self.history = self.model.fit(X ,y_cat, epochs=1, 
                                  validation_split=0.2,
                                  callbacks=[EarlyStopping(patience=3),
                                             ReduceLROnPlateau()],
                                  verbose=1)
    return self

  def predict(self, X, y=None):
    return np.argmax(self.model.predict(X), axis=1)

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

param_grid = [
        {'filters': [32, 64, 128], 'kernel_size': [5, 7, 10, 37]},
    ]

conv1d = ANNConv1D(shape=xn_0.shape)
grid_search = GridSearchCV(conv1d, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(x_train, y_train)


#%%
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

knn = Pipeline([('FeatureExtraction', Heterogeneous()),
                ('scaler', StandardScaler()),
                ('KNN', KNeighborsClassifier())])

param_dist = {"n_estimators": [10, 20],
              "max_features": [4, 8, None]}
rf = Pipeline([('FeatureExtraction', Heterogeneous()),
               ('scaler', StandardScaler()),
               ('RF', GridSearchCV(RandomForestClassifier(),
                                   param_grid=param_dist))])

conv1d = ANNConv1D(shape=xn_0.shape, **grid_search.best_params_)
clfs = [("K-NNeighbors", knn),
        ("RandomForest", rf),
        ("ANN-Conv1d", conv1d)]

#%%
from sklearn.metrics import f1_score, accuracy_score

class_names = ['Idle', 'Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5','Signal 6']
figprop = np.linspace(1, 0.5, 25)
tam = len(class_names) * figprop[len(class_names)]


results = {}
models = {}
genconfmat = True
results['st'] = {}
models['st'] = {}

for clfname, model in clfs:
    print(clfname, end=":\t")
    if not clfname in results['st']:
        results['st'][clfname] = []
    history = model.fit(x_train ,y_train)
    y_pred = model.predict(x_test)
    results['st'][clfname].append([accuracy_score(y_test,y_pred),f1_score(y_test,y_pred,average='macro')])
    print(results['st'][clfname][-1])
    if genconfmat:
        cnf_matrix = confusion_matrix(y_test, y_pred)
        print(cnf_matrix)
        plt.figure(figsize=(tam,tam))
        plot_confusion_matrix(cnf_matrix, classes=class_names, title=clfname+' - ST ', normalize=False)
        #plt.savefig('cnfmatrix_st'+clfname+str(fold)+'round'+str(j)+'.png')
        plt.show()

#%%
for evaluation in results.keys():
  print("\n"+30*"#"+"\n"+evaluation+"\n"+30*"#")
  for clfname,model in clfs:
    print("\n\t"+clfname+" Results\nAccuracy\tF1-Score")
    for i,r in enumerate(results[evaluation][clfname]):
      print("{}\t".format(i+1),end="")
      print(r)
#%%
