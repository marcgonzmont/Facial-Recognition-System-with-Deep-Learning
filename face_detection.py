import argparse
from myPackage import tools as tl
from myPackage import model
import os


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    args = vars(ap.parse_args())

    # Configuration
    batch_size = 32*2
    n_epoch = 15      # 5 for no generator, 60 for generator and 15 for fine tuning
    val_split = None
    generator = True
    trainable = 249    # 249
    modelName = str(batch_size) + "_" + str(n_epoch) + "_" + str(val_split) + "_" + str(generator) + "_" + str(trainable)
    train_model = False

    # Prepare data for training
    X_tr, y_tr, dict_categories = tl.prepareTrainData(args["path"])
    if train_model:
        loaded_model = model.loadModel(len(dict_categories), trainable)
        trained_model = model.trainModel(loaded_model, batch_size, n_epoch, X_tr, y_tr, val_split, generator)
        model.saveModel(modelName, trained_model)
    else:
        current_path = os.getcwd()
        all_jsons = tl.natSort(tl.getSamples(current_path, ext= '.json'))
        all_weights = tl.natSort(tl.getSamples(current_path, ext='.h5'))
        # X_te, y_te = tl.prepateTestDataEval(args["path"], dict_categories)
        X_te, y_te = tl.prepateTestDataPredict(args["path"], dict_categories)
        for i, (json, weights) in enumerate(zip(all_jsons, all_weights)):
            # model.testModel(json, weights, X_te, y_te)
            model.testModel(json, weights, batch_size, X_te, y_te)
