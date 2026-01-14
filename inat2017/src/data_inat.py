import tensorflow as tf
import tensorflow_datasets as tfds

def load_inat2017(split: str, img_size=(299, 299), shuffle=True, seed=0):
    """
    split: 'train' or 'validation' (TFDS naming)
    returns tf.data.Dataset yielding (image, label)
    """
    ds = tfds.load("i_naturalist2017", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=False)

    def _pp(img, y):
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(y, tf.int32)

    return ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)

def take_subset(ds, n: int):
    """Take first n examples deterministically (after shuffle seed fixed)."""
    return ds.take(n)

def batchify(ds, batch_size=64):
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
