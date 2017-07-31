from six.moves import cPickle

# added, save a model
def saveModel(policy, name):
    f = open(name, 'wb')
    cPickle.dump(policy, f, protocol=2)
    f.close()

# added, load a model
def loadModel(name):
    f = open(name, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj