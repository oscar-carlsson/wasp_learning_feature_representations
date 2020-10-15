import struct as st
import numpy as np
import os


def get_mnist():

    os.system("wget -N http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    os.system("wget -N http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    os.system("wget -N http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    os.system("wget -N http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

    os.system("gzip -df train-images-idx3-ubyte.gz")
    os.system("gzip -df train-labels-idx1-ubyte.gz")
    os.system("gzip -df t10k-images-idx3-ubyte.gz")
    os.system("gzip -df t10k-labels-idx1-ubyte.gz")

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
            data_file = open(filename[data_type + file_type], "rb")

            data_file.seek(0)
            magic = st.unpack(">4B", data_file.read(4))
            if file_type is "-images":
                nImg = st.unpack(">I", data_file.read(4))[0]  # num of images
                nR = st.unpack(">I", data_file.read(4))[0]  # num of rows
                nC = st.unpack(">I", data_file.read(4))[0]  # num of column

                nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte

                data_array = np.asarray(
                    st.unpack(">" + "B" * nBytesTotal, data_file.read(nBytesTotal))
                ).reshape((nImg, nR, nC))

            elif file_type is "-labels":
                nLab = st.unpack(">I", data_file.read(4))[0]  # num of images

                nBytesTotal = nLab

                data_array = np.asarray(
                    st.unpack(">" + "B" * nBytesTotal, data_file.read(nBytesTotal))
                ).reshape((nLab,))

            data_dict[data_type + file_type] = data_array

    return data_dict
