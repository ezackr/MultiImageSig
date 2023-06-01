# MultiImageSig
A generalization of the [ImageSig](https://arxiv.org/pdf/2205.06929.pdf) binary image classifier to the multi-class classification task.  ImageSig is an ultra-lightweight image pre-processing method that  splits an input image into chunks, which are interpreted as streams of information. The streams are transformed into low-dimensionals representation, which are then aggregated to form a unique signature of the original image. In the original paper, the authors analyze the efficiency of this preprocessing step applied to neural networks for binary classification. Our approach extends and analyzes this methodology to the multi-class classification setting.

This repository uses PyTorch, rather than the TensorFlow (which was used in the original ImageSig implementation), to implement MultiImageSig.

## Layout
This repository has the following layout:

- ``imagesig``: a package containing all functionality related to the image signature preprocessing calculations. 
- ``model``: a package containing implementations of the neural networks used to evaluate the performance of ImageSig.
- ``util``: general-purpose utility methods.

## Setup
The code base was designed to be cross-compatible for both Windows and UNIX-based operating systems. The requirements for this project are stored in `requirements.txt` and can be installed with ``pip install -r requirements.txt``

In addition, the root directory needs to be added to the Python path. For example, this can be done with the following command in `bash` from the root directory of the project:
```
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

## Usage
### Training a Model
Training a model will train the supplied network according to the given parameters and store resulting checkpoints in the given directory.

To train a model, refer to the following usage guide.
```
usage: train.py [-h] -m {fc,cnn,attn} -ds {cifar10,concretecracks} [-d DEPTH]
                [-b BATCHSIZE] [-n EPOCHS] [-lr LEARNING_RATE]
                [-w WEIGHT_DECAY] [-chkpts CHECKPOINTS_PATH]
                [-ichkpt INITIAL_CHECKPOINT_NAME]
```

As an example, to train a CNN model on concrete cracks, with checkpoints stored in the folder `checkpoints/concretecracks/cnn/` (relative to the project root directory; this folder will be created if it doesn't exist), and depth 4 on the signature transformation with default epochs, learning rate, and weight decay:
```
python src/main/train.py -m cnn -ds concretecracks -d 4 -chkpts checkpoints/concretecracks/cnn/
```

Checkpoints will be stored in the directory provided, named according to the following convention:
`chkpt-[MODEL_NAME]-depth-[DEPTH]-epoch-[CURRENT_EPOCH].pt`.

### Evaluating a Model
Evaluation will compute the test accuracy and F1 score, as well as model parameters and FLOPs.

To evaluate a model given a checkpoint file used during training, refer to the following usage guide to change further parameters for evaluation. Note that these parameters are the same as for train, without the specific training parameters.
```
usage: evaluate.py [-h] -m {fc,cnn,attn} -ds {cifar10,concretecracks}
                   [-d DEPTH] -chkpts CHECKPOINTS_PATH -ichkpt CHECKPOINT_NAME
```

As an example, to evaluate a CNN model trained on concrete cracks, with a checkpoint named `chkpt-CNN-depth-4-epoch-14.pt`, stored in the folder `checkpoints/concretecracks/cnn/` (relative to the project root directory), with signature transformations of depth 4:
```
python src/main/train.py -m cnn -ds concretecracks -d 4 -chkpts checkpoints/concretecracks/cnn/ -ichkpt chkpt-CNN-depth-4-epoch-14.pt
```

### Test Suite
Unit tests are stored under `src/test`.

## Datasets
We use two datasets, CIFAR-10 and Concrete Cracks:
- CIFAR-10 is available via PyTorch ([dataset information](https://www.cs.toronto.edu/~kriz/cifar.html))
- Concrete Cracks is available for download via [Mendeley](https://data.mendeley.com/datasets/5y9wdsg2zt/2). Un-compressed data (`Negative` and `Positive` folders) should be placed into a folder named `data/concrete-crack` in the root directory. 

## Related Resources
[ImageSig](https://github.com/urbanist-ai/ImageSig): The original implementation of ImageSig.

[Signatory](https://github.com/patrick-kidger/signatory): The package used by ImageSig to create image signatures.

[IISignature](https://github.com/bottler/iisignature): Additional library used by ImageSig with other helpful methods for creating signatures. We use this library for computing signatures.

## License
See [LICENSE](LICENSE) for details.
