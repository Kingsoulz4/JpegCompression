import heapq
import numpy as np
import cv2
import time


def rgb_to_ycbcr(rgb_data):
    # Convert the RGB data to YCbCr color space
    r, g, b = np.dsplit(rgb_data, 3)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    return np.dstack((y, cb, cr))


def apply_dct(block):
    # Apply DCT to 8x8 blocks of image
    dct_data = np.zeros_like(block)
    for i in range(0, block.shape[0], 8):
        for j in range(0, block.shape[1], 8):
            dct_data[i:i + 8, j:j + 8] = np.real(dct_2d(block[i:i + 8, j:j + 8]))
    return dct_data


def dct_2d(x):
    # Apply 2D DCT to a 8x8 block
    return np.fft.fftn(x, norm="ortho")


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
    quantized_data = np.round(
        dct_data / (quant_matrix[:, :, num_channel] * quality / 100))

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
    rl_data = []
    # Run-length encode
    rl_block = rle_encode_block(zigzag_encoded.flatten())
    rl_data.append(tuple(rl_block))

    return rl_data


def rle_encode_block(block):
    # Run-length encode differentials
    rl_block = []
    rl_val = block[0]
    count = 0
    for val in block[1:]:
        if val == rl_val:
            count += 1
        else:
            rl_block.append((count, rl_val))
            rl_val = val
            count = 1
    rl_block.append((count, rl_val))

    return rl_block


def huffman_encode(rl_data):
    # Huffman encode the run-length data
    huff_data = b""
    for rl_block in rl_data:
        code, table = encode_block(rl_block)
        huff_data += code
    return huff_data, table


def encode_block(rl_block):
    # Build Huffman-coding table
    freq = {}
    for run, val in rl_block:
        if (run, val) not in freq:
            freq[(run, val)] = 0
        else:
            freq[(run, val)] += 1
    tree = build_huffman_tree(freq)
    codes = assign_huffman_codes(tree)
    # Encode a run-length block using Huffman coding
    encoded_block = b''
    for run, val in rl_block:
        code = codes[(run, val)]
        code_str = "{0:b}".format(code)
        code_bytes = int(code_str, 2).to_bytes((len(code_str) + 7) // 8, byteorder='big')
        encoded_block += code_bytes
    return encoded_block, codes


def build_huffman_tree(freq):
    # Build a Huffman tree from frequency dictionary
    heap = [[wt, [sym, ""]] for sym, wt in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0][1:]


def assign_huffman_codes(tree):
    # Assign Huffman codes to each symbol in the tree
    codes = {}
    for pair in tree:
        sym, code = pair
        codes[sym] = int(code, 2)
    return codes


def block_split(image, block):
    # Split image into non-overlapping blocks
    height, width, _ = image.shape
    bh, bw = block.shape
    blocks_per_row = width // bw
    blocks_per_col = height // bh
    blocks = np.zeros((blocks_per_col, blocks_per_row, bh, bw, 3), dtype=np.int64)
    for row in range(blocks_per_col):
        for col in range(blocks_per_row):
            blocks[row, col] = image[row * bh:(row + 1) * bh, col * bw:(col + 1) * bw]
    return blocks


def jpeg_encode(image, quality=50):
    # Convert the image to YCbCr color space
    ycbcr_image = rgb_to_ycbcr(image)

    # Subsample Cb and Cr components using 4:2:0 sampling
    subsampled_image = ycbcr_image.copy()
    subsampled_image[::2, ::2, 1] = subsampled_image[1::2, ::2, 1]
    subsampled_image[::2, ::2, 2] = subsampled_image[1::2, ::2, 2]

    # Split the image into 8x8 blocks and apply JPEG encoding to each block
    block = np.zeros([8, 8])
    blocks = block_split(subsampled_image, block)

    dct_data = np.zeros_like(blocks)
    quant_data = np.zeros_like(blocks)
    bh = blocks.shape[0]
    bw = blocks.shape[1]
    encode_data = [[[j + k for k in range(3)] for j in range(bw)] for i in range(bh)]
    code_table = [[[j + k for k in range(3)] for j in range(bw)] for i in range(bh)]
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for k in range(3):
                # Apply DCT to each 8x8 block of image
                dct_data[i, j, :, :, k] = apply_dct(blocks[i, j, :, :, k])

                # Quantize the DCT coefficients
                quant_data[i, j, :, :, k] = quantize(dct_data[i, j, :, :, k], quality, k)

                # Perform zigzag scan on a block
                zigzag_encoded = zigzag_encode(quant_data[i, j, :, :, k])

                # Run-length encode the quantized coefficients
                rl_data = rle_encode(zigzag_encoded)

                # Huffman encode the run-length data
                encode_data[i][j][k], code_table[i][j][k] = huffman_encode(rl_data)

    return encode_data, code_table


start = time.time()
img_path = 'old_street.jpg'
img = cv2.imread(img_path)
print(img.shape)
img1 = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
huff, huff_table = jpeg_encode(img1, quality=50)
end = time.time()
final = end - start
print(final)
print(huff[0][0][0])
print(huff_table[0][0][0])
