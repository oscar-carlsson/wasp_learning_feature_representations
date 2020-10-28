import struct as st
import numpy as np
import os
import gzip
import pickle
import sys


def get_mnist(use_keras=False):

    print("Loading MNIST dataset...")

    if use_keras:
        from tensorflow.keras.datasets import mnist

        ((train_images, train_labels), (test_images, test_labels)) = mnist.load_data()

        data_dict = {}

        data_dict["train-images"] = train_images
        data_dict["train-labels"] = train_labels
        data_dict["test-images"] = test_images
        data_dict["test-labels"] = test_labels

        print("Done.")
        return data_dict

    datasets_dir = os.environ.get("DATASETS", "..")
    if datasets_dir == ".":
        datasets_dir = os.path.dirname(os.path.realpath(__file__)) + "/datasets"

    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)

    dataset_dir = datasets_dir + "/mnist"
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    os.chdir(dataset_dir)

    if os.path.isfile("mnist.gz"):
        with gzip.open("mnist.gz", "rb") as f:
            data_dict = pickle.load(f)
        print("Done.")
        return data_dict

    filename = {
        "train-images": "train-images-idx3-ubyte",
        "train-labels": "train-labels-idx1-ubyte",
        "test-images": "t10k-images-idx3-ubyte",
        "test-labels": "t10k-labels-idx1-ubyte",
    }

    file_types = ["-images", "-labels"]
    data_types = ["train", "test"]

    data_dict = {}

    for data_type in data_types:
        for file_type in file_types:

            if not os.path.exists(filename[data_type + file_type]):
                os.system(
                    "wget -N http://yann.lecun.com/exdb/mnist/"
                    + filename[data_type + file_type]
                    + ".gz"
                )
                os.system("gzip -df " + filename[data_type + file_type] + ".gz")

            with open(filename[data_type + file_type], "rb") as data_file:

                data_file.seek(0)

                magic = st.unpack(">4B", data_file.read(4))

                if file_type == "-images":
                    nImg = st.unpack(">I", data_file.read(4))[0]  # num of images
                    nR = st.unpack(">I", data_file.read(4))[0]  # num of rows
                    nC = st.unpack(">I", data_file.read(4))[0]  # num of column

                    nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte

                    data_array = np.asarray(
                        st.unpack(">" + "B" * nBytesTotal, data_file.read(nBytesTotal))
                    ).reshape((nImg, nR, nC))

                elif file_type == "-labels":
                    nLab = st.unpack(">I", data_file.read(4))[0]  # num of images

                    nBytesTotal = nLab

                    data_array = np.asarray(
                        st.unpack(">" + "B" * nBytesTotal, data_file.read(nBytesTotal))
                    ).reshape((nLab,))

            data_dict[data_type + file_type] = data_array

    print("writing pickle")
    with gzip.open("mnist.gz", "wb") as f:
        pickle.dump(data_dict, f)

    print("Done.")

    return data_dict


def get_s2_mnist(filename="s2_mnist.gz"):

    print("Loading MNIST on S2 dataset...")

    datasets_dir = os.environ.get("DATASETS", "..")
    if datasets_dir == ".":
        print("Please set the DATASETS environment variable!")
        sys.exit(1)

    datasets_dir = datasets_dir + "/s2_mnist"
    if not os.path.isfile(datasets_dir + "/" + filename):
        print("Could not find file", filename, "in", datasets_dir)
        sys.exit(1)

    with gzip.open(datasets_dir + "/" + filename, "rb") as f:
        dataset = pickle.load(f)

    print("Done.")

    return dataset


# Local Variables:
# fill-column: 88
# ispell-local-dictionary: "en_US"
# eval: (blacken-mode)
# End:
