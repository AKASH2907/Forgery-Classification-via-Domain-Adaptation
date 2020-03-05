# import the necessary packages
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Flatten,
    Input,
    BatchNormalization,
)
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal
from models import alexnet, vgg_16
import numpy as np
import tensorflow as tf
import time
import argparse
import matplotlib

matplotlib.use("Agg")

init = RandomNormal(mean=0.0, stddev=0.02)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def build_models(model_name):
    """Creates three different models, one used for source only training,
    two used for domain adaptation
    """
    inputs = Input(shape=(224, 224, 3))
    if model_name == "alexnet":
        x4 = alexnet(inputs)

    elif model_name == "vgg16":
        x4 = vgg_16(inputs)

    x4 = Flatten()(x4)

    x4 = Dense(32, activation="relu")(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("elu")(x4)

    source_classifier = Dense(2, activation="softmax", name="mo")(x4)

    # Domain Classification
    domain_classifier = Dense(16, activation="relu", name="do4")(x4)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("elu", name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)

    domain_classifier = Dense(2, activation="softmax", name="do")(domain_classifier)

    # Combined model
    comb_model = Model(inputs=inputs,
        outputs=[source_classifier, domain_classifier]
    )

    comb_model.compile(
        optimizer="Adam",
        loss={"mo": "categorical_crossentropy", "do": "categorical_crossentropy"},
        loss_weights={"mo": 1, "do": 2},
        metrics=["accuracy"],
    )

    source_classification_model = Model(inputs=inputs,
        outputs=[source_classifier],
    )

    source_classification_model.compile(
        optimizer="Adam",
        loss={"mo": "categorical_crossentropy"},
        metrics=["accuracy"],
    )

    domain_classification_model = Model(inputs=inputs,
        outputs=[domain_classifier]
    )
    
    domain_classification_model.compile(
        optimizer="Adam", loss={"do": "categorical_crossentropy"}, metrics=["accuracy"]
    )

    embeddings_model = Model(inputs=inputs, outputs=[x4])
    
    embeddings_model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return (
        comb_model,
        source_classification_model,
        domain_classification_model,
        embeddings_model,
    )


def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns
    batches of the same size
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(
            all_examples_indices, size=batch_size, replace=False
        )
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def train(Xs, ys, Xt, yt, enable_dann=True, n_iterations=200):
    """Domain Adaptation from source to target data.

    Xs: Source data
    ys: Source Labels
    Xt: Target data
    yt: Target Labels
    enable_dann: Choose domain adaptation or not
    n_iteration: Number of iterations over the whole dataset
    After every five epochs, f1 score is calculated and if it's better than the
    previous then weights are saved. Note: No target labels were used at the
    time of training. It's used only for accuracy evaluation.
    """
    f1_scores = 0

    batch_size = 32
    (
        model,
        source_classification_model,
        domain_classification_model,
        embeddings_model,
    ) = build_models("alexnet")

    log_path = "./graph"
    callback = TensorBoard(log_path)
    callback.set_model(model)
    train_names = ["t1", "t2", "t3", "t4", "t5"]

    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))

    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_adversarial = np.ones((batch_size * 2,))

    S_batches = batch_generator([Xs, to_categorical(ys)], batch_size)
    T_batches = batch_generator([Xt, np.zeros(shape=(len(Xt), 2))], batch_size)

    for i in range(n_iterations):
        y_adversarial_2 = to_categorical(
            np.array(([0] * batch_size + [1] * batch_size))
        )

        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)

        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])

        adv_weights = []
        for layer in model.layers:
            if layer.name.startswith("do"):
                adv_weights.append(layer.get_weights())

        if enable_dann:
            # note - even though we save and append weights, the batchnorms 
            # moving means and variances
            # are not saved throught this mechanism
            stats = model.train_on_batch(
                X_adv,
                [y_class, y_adversarial_1],
                sample_weight=[sample_weights_class, sample_weights_adversarial],
            )

            write_log(callback, train_names, stats, i)

            k = 0
            for layer in model.layers:
                if layer.name.startswith("do"):
                    layer.set_weights(adv_weights[k])
                    k += 1

            class_weights = []

            for layer in model.layers:
                if not layer.name.startswith("do"):
                    class_weights.append(layer.get_weights())

            stats2 = domain_classification_model.train_on_batch(
                X_adv, [y_adversarial_2]
            )

            k = 0
            for layer in model.layers:
                if not layer.name.startswith("do"):
                    layer.set_weights(class_weights[k])
                    k += 1

        else:
            stats = source_classification_model.train_on_batch(X0, y0)

        if (i + 1) % 5 == 0:
            print(i, stats)
            y_test_hat_t = source_classification_model.predict(Xt).argmax(1)
            y_test_hat_s = source_classification_model.predict(Xs).argmax(1)
            print(
                "Iteration %d, source accuracy =  %.5f, target accuracy = %.5f"
                % (
                    i,
                    accuracy_score(ys, y_test_hat_s),
                    accuracy_score(yt, y_test_hat_t),
                )
            )
            print(
                "Iteration %d, source f1_score =  %.5f, target f1_score = %.5f"
                % (i, f1_score(ys, y_test_hat_s), f1_score(yt, y_test_hat_t))
            )
            if f1_score(yt, y_test_hat_t) > f1_scores and i > 20:
                f1_scores = f1_score(yt, y_test_hat_t)
                source_classification_model.save_weights("alex_wts_cmf25.h5")
                print("saving wts....")
    return embeddings_model


def test(X_test, Y_test):
    """Generate predictions over target dataset.

    X_test: Target data
    Y_test: Target Labels
    Load the model weights and calclate the accuracy, precision, recall
    and f-1 score. Save the predictions as well.
    """
    (
        model,
        source_classification_model,
        domain_classification_model,
        embeddings_model,
    ) = build_models("alexnet")

    source_classification_model.load_weights(
        "./casia_wts/cmf+inpnt_wts/alex_wts_15k_cmf+inpaint.h5"
    )
    print("Model wts loaded...")

    start = time.time()

    pred = source_classification_model.predict(X_test).argmax(1)

    end = time.time()
    print("Time for execution:", (end - start))

    tests_f1 = f1_score(Y_test, pred)
    tests_acc = accuracy_score(Y_test, pred)
    precise = precision_score(Y_test, pred)
    recalls = recall_score(Y_test, pred)
    print("Target Test Accuracy:", tests_acc)
    print("Target Precision score:", precise)
    print("Target Recall score:", recalls)
    print("Target Test F1:", tests_f1)

    np.save("y_test.npy", Y_test)
    return tests_f1


def main():

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-t",
        "--train_flag",
        required=True,
        type=int,
        default=0,
        help="Do you want to train the model?? If yes, then set flag to 0",
    )
    ap.add_argument("-e", "--epochs", type=int, default=100)
    args = ap.parse_args()

    train_flag = args.train_flag

    if train_flag == 0:
        print("Model is training...")
        print("Number of iterations:", args.epochs)

        Xs = np.load("x_cmf25.npy")
        ys = np.load("y_cmf25.npy")

        Xt = np.load("x_casia_tr.npy")
        yt = np.load("y_casia_tr.npy")

        y_s = []

        for i in range(ys.shape[0]):
            # for i in range(10):

            if ys[i] == "authentic":
                y_s += [0]
            else:
                y_s += [1]
        y_s = np.asarray(y_s)

        y_t = []
        for i in range(yt.shape[0]):
            if yt[i] == "authentic":
                y_t += [0]
            else:
                y_t += [1]
        y_t = np.asarray(y_t)

        train_embs = train(Xs, y_s, Xt, y_t,
            enable_dann=True,
            n_iterations=args.epochs)

    else:
        print("Model testing...")

        Xtt = np.load("x_casia_test.npy")
        ytt = np.load("y_casia_test.npy")
        y_tt = []

        for i in range(ytt.shape[0]):
            if ytt[i] == "authentic":
                y_tt += [0]
            else:
                y_tt += [1]

        y_tt = np.asarray(y_tt)

        test_acc = test(Xtt, y_tt)


if __name__ == "__main__":
    main()
