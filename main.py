import argparse
from processor import process_file
import logging


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", default='/media/aleksander/hd2/data/hackthehack/rbk/telecast_576p.mp4',
                        type=str, help="input mp4 file")

    parser.add_argument("--text-embeddings", default=True, type=str2bool, help="add text embeddings to layers")
    parser.add_argument("--find-peoples", default=True, type=str2bool, help="find persons in video stream")
    parser.add_argument("--sentiment", default=True, type=str2bool, help="add sentiment analysis to layers")
    parser.add_argument("--summarize", default=True, type=str2bool, help="add sber-gpt3 summarization layer")
    parser.add_argument("--verbose", default=True, type=str2bool, help="verbosity to video parsing")
    parser.add_argument("--max-length", default=60, type=int, help="maximum parsing time in seconds")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    result = process_file(args.input_file, args)

    print(result)
