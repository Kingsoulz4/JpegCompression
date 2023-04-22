#CODE LỖI: ENCODE
import cv2
import numpy as np

# Đọc tệp DNG
img = cv2.imread("old_street.jpg", cv2.IMREAD_UNCHANGED)

# Chuyển đổi không gian màu sang YCbCr
img_YCbCr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

print(img_YCbCr.shape)

# Ma trận quantization
quantization_matrix = np.array([[16,11,10,16,24,40,51,61],
                               [12,12,14,19,26,58,60,55],
                               [14,13,16,24,40,57,69,56],
                               [14,17,22,29,51,87,80,62],
                               [18,22,37,56,68,109,103,77],
                               [24,35,55,64,81,104,113,92],
                               [49,64,78,87,103,121,120,101],
                               [72,92,95,98,112,100,103,99]])

# Mức độ nén
compression_level = 5

# Thực hiện phép biến đổi DCT và quantization cho từng khối 8x8
dct_blocks = np.zeros_like(img_YCbCr, dtype=np.float32)
for i in range(0, img_YCbCr.shape[0], 8):
    for j in range(0, img_YCbCr.shape[1], 8):
        for k in range(0, img_YCbCr.shape[2]):
            dct_blocks[i:i+8,j:j+8, k] = cv2.dct(np.float32(img_YCbCr[i:i+8,j:j+8, k]))
            
            # Quantization
            
        dct_blocks[i:i+8, j:j+8, 0] = np.round(dct_blocks[i:i+8, j:j+8, 0]/(quantization_matrix*compression_level))
            
        dct_blocks[i:i+8, j:j+8, 1] = np.round(dct_blocks[i:i+8, j:j+8, 1]/(quantization_matrix*(compression_level/2)))
        dct_blocks[i:i+8, j:j+8, 2] = np.round(dct_blocks[i:i+8, j:j+8, 2]/(quantization_matrix*(compression_level/2)))
        
# Thứ tự Zig-Zag
zigzag_order = np.array([[0, 1, 5, 6,14,15,27,28],
                         [2, 4, 7,13,16,26,29,42],
                         [3, 8,12,17,25,30,41,43],
                         [9,11,18,24,31,40,44,53],
                         [10,19,23,32,39,45,52,54],
                         [20,22,33,38,46,51,55,60],
                         [21,34,37,47,50,56,59,61],
                         [35,36,48,49,57,58,62,63]])

# Mã hóa Huffman
huffman_encoded = []
for i in range(0, img_YCbCr.shape[0], 8):
    for j in range(0, img_YCbCr.shape[1], 8):
        for k in range(3):
            block = dct_blocks[i:i+8, j:j+8, k].reshape(-1)

            # Thực hiện thứ tự Zig-Zag
            zigzag_block = block[zigzag_order.reshape(-1)]

            # Mã hóa Huffman
            huffman_encoded.extend(zigzag_block.tolist())

# Tạo tệp JPEG
huffman_encoded = np.array(huffman_encoded)
huffman_encoded_bytes = bytearray(np.packbits(huffman_encoded > 0))

with open('output.jpg', 'wb') as f:
    f.write(huffman_encoded_bytes)



#DECODE (LỖI)
import cv2
import numpy as np

# Đọc tệp JPEG đã nén
with open('output.jpg', 'rb') as f:
    data = bytearray(f.read())

# Giải nén Huffman-encoded data
huffman_decoded_bits = np.unpackbits(data)
huffman_decoded = []
for i in range(0, len(huffman_decoded_bits), 64):
    block = huffman_decoded_bits[i:i+64]
    nonzero_indices = np.where(block)[0]
    decoded_block = np.zeros_like(block)
    zigzag_order = np.array([[0, 1, 5, 6,14,15,27,28],
                             [2, 4, 7,13,16,26,29,42],
                             [3, 8,12,17,25,30,41,43],
                             [9,11,18,24,31,40,44,53],
                             [10,19,23,32,39,45,52,54],
                             [20,22,33,38,46,51,55,60],
                             [21,34,37,47,50,56,59,61],
                             [35,36,48,49,57,58,62,63]])
    decoded_block[zigzag_order.reshape(-1)[nonzero_indices]] = block[nonzero_indices]
    huffman_decoded.append(decoded_block.reshape(8, 8))

# Sử dụng quantization matrix
quantization_matrix = np.array([[16,11,10,16,24,40,51,61],
                               [12,12,14,19,26,58,60,55],
                               [14,13,16,24,40,57,69,56],
                               [14,17,22,29,51,87,80,62],
                               [18,22,37,56,68,109,103,77],
                               [24,35,55,64,81,104,113,92],
                               [49,64,78,87,103,121,120,101],
                               [72,92,95,98,112,100,103,99]])

# Mức độ nén
compression_level = 5

# Thực hiện phép biến đổi IDCT và dequantization cho từng khối 8x8
idct_blocks = np.zeros((len(huffman_decoded)*8, len(huffman_decoded[0])*8, 3))
for i in range(0, len(huffman_decoded)):
    for j in range(0, len(huffman_decoded[0])):
        idct_blocks[i*8:i*8+8,j*8:j*8+8,:] = cv2.idct(np.float32(huffman_decoded[i][j])).clip(0, 255)

        # Dequantization
        idct_blocks[i*8:i*8+8,j*8:j*8+8,0] *= (quantization_matrix*compression_level)
        idct_blocks[i*8:i*8+8,j*8:j*8+8,1] *= (quantization_matrix*(compression_level/2))
        idct_blocks[i*8:i*8+8,j*8:j*8+8,2] *= (quantization_matrix*(compression_level/2))

# Chuyển đổi không gian màu từ YCbCr sang RGB
img_RGB = cv2.cvtColor(idct_blocks.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# Lưu ảnh đã giải nén
cv2.imwrite('output.png', img_RGB)




