import argparse
import struct
from nvidia_tao_tf1.encoding import encoding

parser = argparse.ArgumentParser(description='ETLT Decode Tool')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Path to the etlt file.')
parser.add_argument('-o',
                    '--uff',
                    required=True,
                    type=str,
                    help='The path to the uff file.')
parser.add_argument('-k',
                    '--key',
                    required=True,
                    type=str,
                    help='encryption key.')
args = parser.parse_args()
print(args)

with open(args.uff, 'wb') as temp_file, open(args.model, 'rb') as encoded_file:
    size = encoded_file.read(4)
    size = struct.unpack("<i", size)[0]
    input_node_name = encoded_file.read(size)
    encoding.decode(encoded_file, temp_file, args.key.encode())
print("Decode successfully.")