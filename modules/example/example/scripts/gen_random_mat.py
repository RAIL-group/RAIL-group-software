"""Generate matrix of random numbers and optionally write image to file."""
import argparse
import example
import matplotlib.pyplot as plt


def main(args):
    mat = example.core.get_random_matrix((5, 5))
    plt.imshow(mat)
    if args.image_name is not None:
        plt.savefig(args.image_name, dpi=150)
    else:
        print(mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-name", required=False)
    args = parser.parse_args()
    main(args)
