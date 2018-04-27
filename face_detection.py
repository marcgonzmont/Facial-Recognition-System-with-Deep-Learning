import argparse
from myPackage import tools as tl
from myPackage import model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    args = vars(ap.parse_args())

    # Configuration
    batch_size = 32*2
    n_epoch = 5
    val_split = None
    # Prepare data for training
    X_tr, y_tr, n_categories = tl.prepareTrainData(args["path"])
    loaded_model = model.loadModel(n_categories)
    trained_model = model.trainModel(loaded_model, batch_size, n_epoch, X_tr, y_tr, val_split)

    # X_te, y_te = tl.prepateTestData(args["path"])
    # model.testModel(trained_model, batch_size, X, y)
    # print()
