import math
import numpy as np
import cv2
import time

from Encode import jpeg_encode, rgb_to_ycbcr, subsampling_422, zigzag_points


def huffman_decoding(encoded_data, huffman_dict):
    # Tạo bảng giải mã từ bảng mã hóa Huffman
    reverse_huffman_dict = {v: k for k, v in huffman_dict.items()}

    # Giải mã dữ liệu đầu vào bằng bảng giải mã Huffman
    decoded_data = ""
    temp_str = ""
    for bit in encoded_data:
        temp_str += bit
        if temp_str in reverse_huffman_dict:
            decoded_data += reverse_huffman_dict[temp_str]
            temp_str = ""

    return decoded_data


def zigzag_decode(rle_decoded_data):
    # Perform zigzag scan on a block
    block_size = int(math.sqrt(len(rle_decoded_data)))
    # print(len(rle_decoded_data))
    zigzag_decoded_data = np.zeros([block_size, block_size], dtype=np.float32)
    points = zigzag_points(8)
    i = 0
    for point in points:
        zigzag_decoded_data[point[0], point[1]] = rle_decoded_data[i]
        i = i + 1
    zigzag_decoded_data = np.array(zigzag_decoded_data)
    return zigzag_decoded_data


def rle_decode(rl_data):
    flat_data = []
    for i in range(len(rl_data)):
        if i % 2 != 0:
            value = rl_data[i]
            for n in range(length):
                flat_data.append(value)
        else:
            length = rl_data[i]
    rle_decoded_data = np.array(flat_data, dtype=np.float32)
    return rle_decoded_data


def idct_2d(x):
    # Apply 2D DCT to a 8x8 block
    tmp = cv2.idct(x)
    idct_block = np.uint8(tmp)
    return idct_block


def decode_quantize(quantized_data, quality, num_channel):
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

    # Compute the dequantized DCT coefficients
    dequantized_data = quantized_data * (quant_matrix[:, :, num_channel] * quality / 100)

    return dequantized_data


def ycbcr_to_rgb(ycbcr_data):
    # Lấy chiều cao và chiều rộng của ảnh
    height, width = ycbcr_data.shape[:2]

    # Khởi tạo mảng numpy cho kênh R, G, B
    r_channel = np.zeros((height, width), dtype=np.uint8)
    g_channel = np.zeros((height, width), dtype=np.uint8)
    b_channel = np.zeros((height, width), dtype=np.uint8)

    # Chuyển đổi từ YCbCr sang RGB
    for i in range(height):
        for j in range(width):
            y, cb, cr = ycbcr_data[i, j]
            r_channel[i, j] = np.clip(y + 1.402 * (cr - 128), 0, 255)
            g_channel[i, j] = np.clip(y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128), 0, 255)
            b_channel[i, j] = np.clip(y + 1.772 * (cb - 128), 0, 255)

    # Gộp 3 kênh màu lại thành ảnh RGB
    rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_data[:, :, 0] = b_channel
    rgb_data[:, :, 1] = g_channel
    rgb_data[:, :, 2] = r_channel

    return rgb_data


def de_block_split(blocks):
    blocks = np.array(blocks)
    bh, bw, _, _ = blocks.shape
    channel = np.zeros([bh * 8, bw * 8])
    for row in range(bh):
        for col in range(bw):
            channel[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = blocks[row, col]
    return channel


def jpeg_decode(encoded_data, codes, quality=50):
    bw = len(encoded_data)
    bh = len(encoded_data[0])
    rl_data = [[[] for j in range(bh)] for i in range(bw)]
    rle_de_data = [[[] for j in range(bh)] for i in range(bw)]
    idct_data = [[[] for j in range(bh)] for i in range(bw)]
    for i in range(bw):
        for j in range(bh):
            # Huffman encode the run-length data
            tmp = huffman_decoding(encoded_data[i][j], codes[i][j])
            # Split data by "," to get array of rl_data
            arr = tmp.split(",")
            for n in range(len(arr)):
                arr[n] = int(arr[n])
            rl_data[i][j] = arr
            # print(rl_data[i][j][k])
            # RLE decode
            rle_de_data[i][j] = rle_decode(rl_data[i][j])
            # print(rle_de_data[i][j][k])
            # zig zag decode
            zigzag_decoded = zigzag_decode(rle_de_data[i][j])
            # print(zigzag_decoded)
            # Dequantize
            deq_data = decode_quantize(zigzag_decoded, quality, 0)
            # IDCT
            idct_data[i][j] = idct_2d(deq_data)
    channel = de_block_split(idct_data)
    return channel
