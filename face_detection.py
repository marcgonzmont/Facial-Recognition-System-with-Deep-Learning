import argparse
from myPackage import tools as tl
from myPackage import model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    args = vars(ap.parse_args())

    # Prepare data for training
    X, y, n_categories = tl.prepareTrainData(args["path"])
    loaded_model = model.loadModel(n_categories)
    trained_model = model.trainModel(loaded_model, X, y)

    X, y = tl.prepateTestData(args["path"])
    model.testModel(trained_model, X)
