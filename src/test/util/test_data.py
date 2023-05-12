from src.main.util.data import load_cifar10


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
