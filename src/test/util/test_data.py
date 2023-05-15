import math

from src.main.util.data import load_cifar10, load_concrete_cracks, get_data_loaders


def test_load_cifar10():
    train_data, test_data = load_cifar10()
    assert len(train_data) == 50000
    assert len(test_data) == 10000
    assert hasattr(train_data, "classes")
    assert train_data.classes == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]


def test_concrete_cracks():
    train_data, test_data = load_concrete_cracks()
    assert len(train_data) == 34000
    assert len(test_data) == 6000


def test_data_loaders_cifar10():
    num_classes, data_shape, train_dl, val_dl, test_dl = get_data_loaders("cifar10", 4, 64)
    assert num_classes == 10
    assert len(train_dl) + len(val_dl) == 50000
    assert len(test_dl) == 10000
    assert hasattr(train_dl.dataset, "classes")
    assert train_dl.dataset.classes == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]


def test_data_loaders_cifar10():
    num_classes, data_shape, train_dl, val_dl, test_dl = get_data_loaders("cifar10", 4, 64)
    assert num_classes == 10
    assert len(train_dl) == math.ceil(0.9*50000/64)  # check 90% split for train
    assert len(val_dl) == math.ceil(0.1*50000/64)  # check 10% split for val
    assert len(test_dl) == math.ceil(10000/64)
    for subset in [train_dl.dataset, val_dl.dataset, test_dl]:
        assert hasattr(subset.dataset, "classes")  # dl -> subset -> CIFARSignature dataset
        assert subset.dataset.classes == [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]


def test_data_loaders_concretecracks():
    num_classes, data_shape, train_dl, val_dl, test_dl = get_data_loaders("concretecracks", 4, 64)
    assert num_classes == 2
    assert len(train_dl) == math.ceil(0.9*34000/64)  # check 90% split for train
    assert len(val_dl) == math.ceil(0.1*34000/64)  # check 10% split for val
    assert len(test_dl) == math.ceil(6000/64)