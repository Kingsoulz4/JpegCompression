import heapq
import numpy as np
import cv2


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
    return np.fft.fft2(x, norm="ortho")


def quantize(dct_data, quality):
    # Quantize the coefficients using JPEG standard quantization matrix
    quant_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                             [12, 12, 14, 19, 26, 58, 60, 55],
                             [14, 13, 16, 24, 40, 57, 69, 56],
                             [14, 17, 22, 29, 51, 87, 80, 62],
                             [18, 22, 37, 56, 68, 109, 103, 77],
                             [24, 35, 55, 64, 81, 104, 113, 92],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99]])
    # Quantization
    quant_matrix = quant_matrix.reshape((8, 8, 1, 1))
    for i in range(0, dct_data.shape[0], 8):
        for j in range(0, dct_data.shape[1], 8):
            for k in range(0, 3):
                dct_data[i:i + 8, j:j + 8, :, :, k] = np.round(
                    dct_data[i:i + 8, j:j + 8, :, :, k] / (quant_matrix * quality / 100))

    return dct_data


def rle_encode(quant_data):
    # Run-length encode the quantized coefficients
    rl_data = []
    for i in range(0, quant_data.shape[0], 8):
        for j in range(0, quant_data.shape[1], 8):
            block = quant_data[i:i + 8, j:j + 8]
            zigzag = zigzag_encode(block)
            rl_block = rle_encode_block(zigzag)
            rl_data.append(rl_block)
    return rl_data


def zigzag_encode(block):
    # Perform zigzag scan on a block
    return np.array([block[point] for point in zigzag_points(8)])


def zigzag_points(n):
    # Generate zigzag scan points
    points = np.empty((n * n, 2), dtype=np.intp)
    for i in range(n):
        for j in range(n):
            k = i + j
            if k % 2 == 0:
                points[k // 2] = i, j
            else:
                points[n * n - 1 - (k // 2)] = i, j
    return points


def rle_encode_block(block):
    # Run-length encode a zigzag-scanned block
    rl_block = []
    i = 0
    while i < len(block):
        run = 0
        while i < len(block) and block[i] == 0:
            run += 1
            i += 1
        if i < len(block):
            rl_block.append((run, block[i]))
            i += 1
    return rl_block


def huffman_encode(rl_data):
    # Huffman encode the run-length data
    huff_data = b""
    for rl_block in rl_data:
        code = encode_block(rl_block)
        huff_data += code
    return huff_data


def encode_block(rl_block):
    # Encode a run-length block using Huffman coding
    freq = {}
    for run, val in rl_block:
        if (run, val) not in freq:
            freq[(run, val)] = 0
        freq[(run, val)] += 1
    tree = build_huffman_tree(freq)
    codes = assign_huffman_codes(tree)
    encoded_block = b""
    for run, val in rl_block:
        code = codes[(run, val)]
        encoded_block += code.to_bytes((len(code) + 7) // 8, byteorder='big')
    return encoded_block


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
    blocks = np.zeros((blocks_per_col, blocks_per_row, bh, bw, 3), dtype=np.float32)
    for row in range(blocks_per_col):
        for col in range(blocks_per_row):
            blocks[row, col] = image[row*bh:(row+1)*bh, col*bw:(col+1)*bw]
    return blocks


def jpeg_encode(image, quality=50):
    # Convert the image to YCbCr color space
    ycbcr_image = rgb_to_ycbcr(image)

    # Subsample Cb and Cr components using 4:2:0 sampling
    subsampled_image = ycbcr_image.copy()
    subsampled_image[::2, ::2, 1] = subsampled_image[1::2, ::2, 1]
    subsampled_image[::2, ::2, 2] = subsampled_image[1::2, ::2, 2]

    # Split the image into 8x8 blocks and apply DCT to each block
    block = np.zeros([8, 8])
    blocks = block_split(subsampled_image, block)

    # Apply DCT to each 8x8 block of image
    dct_data = apply_dct(blocks)
    print(dct_data.shape)
    # Quantize the DCT coefficients
    quant_data = quantize(dct_data, quality)

    # Run-length encode the quantized coefficients
    rl_data = rle_encode(quant_data)

    # Huffman encode the run-length data
    encode_data = huffman_encode(rl_data)

    return encode_data


img_path = 'Data/old_street.jpg'
img = cv2.imread(img_path)
huff = jpeg_encode(img, quality=50)
