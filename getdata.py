from sklearn.model_selection import train_test_split
import pickle


def load(test_size=0.2):
    file = open('images.pkl', 'rb')
    f = pickle.load(file=file)
    file.close()

    x = f['x']
    y = f['y']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
    return x_train, x_test, y_train, y_test