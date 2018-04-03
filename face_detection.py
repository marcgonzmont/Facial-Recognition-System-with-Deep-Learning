import argparse
from myPackage import tools as tl
from myPackage import model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    args = vars(ap.parse_args())

    # Configuration
    batch_size = 32
    n_epoch = 1
    # Prepare data for training
    X, y, n_categories = tl.prepareTrainData(args["path"])
    loaded_model = model.loadModel(n_categories)
    trained_model = model.trainModel(loaded_model, batch_size, n_epoch, X, y)

    X, y = tl.prepateTestData(args["path"])
    model.testModel(trained_model, batch_size, X, y)
    # print()
