import argparse
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str)
    parser.add_argument('--save_path', type=str, default="./weight/model_v1.keras")

    return parser.parse_args()


def main():
    args = get_args()

    inputs = args.inputs

    generator = tf.keras.models.load_weights(args.save_path)
    result = generator.predict(inputs)

    name = extract_name(result)

    return name


if __name__ == "__main__":
    main()
