import heapq
from collections import defaultdict
from collections import Counter
from collections import deque

import numpy as np
import cv2


def rgb_to_ycbcr(img):
    # Convert the RGB data to YCbCr color space
    ycbcr_data = np.zeros_like(img)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    ycbcr_data[:, :, 0] = np.uint8(0.299 * r + 0.587 * g + 0.114 * b)
    ycbcr_data[:, :, 1] = np.uint8(-0.1687 * r - 0.3313 * g + 0.5 * b + 128)
    ycbcr_data[:, :, 2] = np.uint8(0.5 * r - 0.4187 * g - 0.0813 * b + 128)
    return ycbcr_data


def subsampling_422(data):
    height, width, _ = data.shape
    Y = data[:, :, 0]
    Cb = data[:, ::2, 1]
    Cr = data[:, ::2, 2]
    return Y, Cb, Cr


def dct_2d(x):
    # Apply 2D DCT to a 8x8 block
    dct_block = cv2.dct(np.float32(x), cv2.DCT_INVERSE)
    return dct_block


def quantize(dct_data, quality, num_channel):
    # Define the quantization matrix for luminance component
    luminance_quant_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                       [12, 12, 14, 19, 26, 58, 60, 55],
                                       [14, 13, 16, 24, 40, 57, 69, 56],
                                       [14, 17, 22, 29, 51, 87, 80, 62],
                                       [18, 22, 37, 56, 68, 109, 103, 77],
                                       [24, 35, 55, 64, 81, 104, 113, 92],
                                       [49, 64, 78, 87, 103, 121, 120, 101],
                                       [72, 92, 95, 98, 112, 100, 103, 99]])

    # Define the quantization matrix for chrominance component
    chrominance_quant_matrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                         [18, 21, 26, 66, 99, 99, 99, 99],
                                         [24, 26, 56, 99, 99, 99, 99, 99],
                                         [47, 66, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99]])

    # Initialize the quantization matrix
    quant_matrix = np.zeros((8, 8, 3))

    # Assign the quantization matrix based on the component
    quant_matrix[:, :, 0] = luminance_quant_matrix
    quant_matrix[:, :, 1] = chrominance_quant_matrix
    quant_matrix[:, :, 2] = chrominance_quant_matrix

    # Quantize the DCT coefficients
    quantized_data = dct_data / (quant_matrix[:, :, num_channel] * quality / 100)

    return quantized_data.astype(int)


def zigzag_encode(block):
    # Perform zigzag scan on a block
    points = zigzag_points(8)
    return np.array([block[point[0], point[1]] for point in points])


def zigzag_points(n):
    # Initialize the output array with zeros
    points = np.zeros((n * n, 2), dtype=int)
    row, col = 0, 0
    for i in range(n * n):
        points[i] = row, col
        if (row + col) % 2 == 0:
            # Even stripes
            if col == n - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:
            # Odd stripes
            if row == n - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1
    return points


def rle_encode(zigzag_encoded):
    rl_data = ""
    # Run-length encode
    rl_block = rle_encode_block(zigzag_encoded.flatten())
    rl_data += rl_block

    return rl_data


def rle_encode_block(zigzag_encoded):
    # Run-length encode differentials
    rl_block = ""
    rl_val = zigzag_encoded[0]
    count = 1
    for val in zigzag_encoded[1:]:
        if val == rl_val:
            count += 1
        else:
            rl_block = rl_block + str(count) + "," + str(rl_val) + ","
            rl_val = val
            count = 1
    rl_block = rl_block + str(count) + "," + str(rl_val)

    return rl_block


def huffman_encoding(rl_data):
    # Count the frequencies of characters
    freq_dict = Counter(rl_data)

    # Initialize the Huffman tree as a deque of (frequency, character) tuples
    tree = deque(sorted((freq, char) for char, freq in freq_dict.items()))

    # Build the Huffman tree by merging nodes
    while len(tree) > 1:
        left_freq, left_char = tree.popleft()
        right_freq, right_char = tree.popleft()
        node = (left_freq + right_freq, (left_char, right_char))
        tree.append(node)

    # Create the Huffman encoding table by traversing the Huffman tree
    encoding_table = {}
    def traverse(node, prefix):
        if isinstance(node, str):
            encoding_table[node] = prefix
        else:
            traverse(node[0], prefix + "0")
            traverse(node[1], prefix + "1")
    traverse(tree[0][1], "")

    # Encode the input string using the Huffman encoding table
    encoded_data = [encoding_table[char] for char in rl_data]
    encoded_data = "".join(encoded_data)

    return encoded_data, encoding_table


def block_split(image, block):
    # Split image into non-overlapping blocks
    height, width = image.shape
    bh, bw = block.shape
    blocks_per_row = width // bw
    blocks_per_col = height // bh
    blocks = np.zeros((blocks_per_col, blocks_per_row, bh, bw), dtype=np.int64)
    for row in range(blocks_per_col):
        for col in range(blocks_per_row):
            blocks[row, col] = image[row * bh:(row + 1) * bh, col * bw:(col + 1) * bw]
    return blocks


def jpeg_encode(channel, quality=50):
    # Split the image into 8x8 blocks and apply JPEG encoding to each block
    block = np.zeros([8, 8])
    blocks = block_split(channel, block)
    # Apply DCT to each 8x8 block of image
    dct_data = np.zeros_like(blocks)
    quant_data = np.zeros_like(blocks)
    bh = blocks.shape[0]
    bw = blocks.shape[1]
    encode_data = [[[] for j in range(bw)] for i in range(bh)]
    code_table = [[[] for j in range(bw)] for i in range(bh)]
    rl_data = [[[] for j in range(bw)] for i in range(bh)]
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            # Apply DCT to each 8x8 block of image
            dct_data[i, j, :, :] = dct_2d(blocks[i, j, :, :])

            # Quantize the DCT coefficients
            quant_data[i, j, :, :] = quantize(dct_data[i, j, :, :], quality, 0)

            # Perform zigzag scan on a block
            zigzag_encoded = zigzag_encode(quant_data[i, j, :, :])

            # Run-length encode the quantized coefficients
            rl_data[i][j] = rle_encode(zigzag_encoded)

            # Huffman encode the run-length data
            encode_data[i][j], code_table[i][j] = huffman_encoding(rl_data[i][j])
    return encode_data, code_table
