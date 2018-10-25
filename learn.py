import formula
import rand
import sat
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
import numpy as np


VARIABLES = ['a', 'b', 'c', formula.bot.a]
CONNECTIVES = [formula.conj, formula.disj, formula.impl]
MAX_DEPTH = 5
TRAIN_SIZE = 5000
TEST_SIZE = 1000
CHARACTERS = 'abc⊥∧∨→() '  # 'a⊥→() '
FEATURES = len(CHARACTERS)


def random_formula():
    return rand.rand_lt_depth(MAX_DEPTH, VARIABLES, CONNECTIVES)


def vectorise(f, max_length):
    s = str(f)
    for x in s:
        if x not in CHARACTERS:
            raise NotImplementedError(f'The character "{x}" is not supported')
    v = [CHARACTERS.index(x) for x in s]
    return np.array(v + (max_length - len(v)) * [0])


def create_model(max_length):
    model = Sequential()
    model.add(Embedding(FEATURES, output_dim=256, input_length=max_length, mask_zero=True))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def generate_training_samples():
    formulae = [random_formula() for _ in range(TRAIN_SIZE)]
    max_length = max(map(len, map(str, formulae)))
    return formulae, max_length


def generate_testing_samples(max_length, k=1):
    results = []
    for i in range(k):
        f = random_formula()
        while len(str(f)) > max_length:
            f = random_formula()
        results.append(f)
    return results


def train_model(model, samples, max_length):
    x_train = np.array([vectorise(f, max_length) for f in samples])
    y_train = np.array([np.array([1.0 if sat.tautological(f) else 0.0]) for f in samples])
    print(x_train, y_train)
    checkpointer = ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=8, epochs=15, callbacks=[checkpointer])


def evaluate_model(model, max_length):
    formulae = generate_testing_samples(max_length, TEST_SIZE)
    #formulae = list(itertools.islice((random_formula() for i in itertools.count()), TEST_SIZE))
    x_test = np.array([vectorise(f, max_length) for f in formulae])
    y_test = np.array([np.array([1.0 if sat.tautological(f) else 0.0]) for f in formulae])
    print(x_test, y_test)
    score = model.evaluate(x_test, y_test, batch_size=16)
    return score


if __name__ == '__main__':
    samples, max_length = generate_training_samples()
    m = create_model(max_length)
    train_model(m, samples, max_length)
    print('Score:', evaluate_model(m, max_length))

    for f in generate_testing_samples(max_length, 10):
        [[prediction]] = m.predict(np.array([vectorise(f, max_length)]))
        if prediction > 0.9:
            result = 'strongly true'
        elif prediction > 0.55:
            result = 'true'
        elif prediction > 0.45:
            result = 'unsure'
        elif prediction > 0.1:
            result = 'false'
        else:
            result = 'strongly false'
        print(sat.tautological(f), prediction, result, f)
