from FashionMNISTClassifier import *


def train(classifier):
    (train_images, train_labels), (test_images,
                                   test_labels) = classifier.get_dataset()

    model = classifier.get_model()
    callbacks = classifier.get_callbacks()

    classifier.train(train_images, train_labels, model,
                     epochs=150, callbacks=callbacks)

    classifier.evaluate(test_images, test_labels, model)


if __name__ == "__main__":
    classifier = FashionMNISTClassifier()
    train(classifier)
