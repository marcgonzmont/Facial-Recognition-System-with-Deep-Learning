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
    generator = True
    # si quiero hacer train y val después de la preparación del conjunto quedarme con el % que corresponda (tr[0,np.floor(len(vect)*val_split]...)
    # Prepare data for training
    X_tr, y_tr, n_categories = tl.prepareTrainData(args["path"])
    loaded_model = model.loadModel(n_categories)
    # Train model
    if generator:
        # generated = model.generate(X_tr)
        trained_model = model.trainGenModel(loaded_model, batch_size, n_epoch, X_tr, y_tr, val_split)

    else:
        trained_model = model.trainModel(loaded_model, batch_size, n_epoch, X_tr, y_tr, val_split)

    # X_te, y_te = tl.prepateTestData(args["path"])
    # model.testModel(trained_model, batch_size, X, y)
    # print()
