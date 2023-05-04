import time
import cv2
import numpy as np
import rawpy

from Decode import jpeg_decode, ycbcr_to_rgb, jpeg_decode_full
from Encode import rgb_to_ycbcr, subsampling_422, jpeg_encode


def calculate_ratio_compression(huff_y, huff_u, huff_v, huff_table_y, huff_table_u, huff_table_v):
    y_bits_transmit = 0 #total bits of y channel
    u_bits_transmit = 0 #total bits of u channel
    v_bits_transmit = 0 #total bits of v channel

    for i in range(0, len(huff_y)):
        for j in range (0, len(huff_y[0])):
            y_bits_transmit += len(huff_y[i][j])

    for i in range(0, len(huff_table_y)):
        for j in range (0, len(huff_table_y[0])):
            y_bits_transmit += len(huff_table_y[i][j])

    for i in range(0, len(huff_u)):
        for j in range (0, len(huff_u[0])):
            u_bits_transmit += len(huff_u[i][j])

    for i in range(0, len(huff_table_u)):
        for j in range (0, len(huff_table_u[0])):
            u_bits_transmit += len(huff_table_u[i][j])

    for i in range(0, len(huff_v)):
        for j in range (0, len(huff_v[0])):
            v_bits_transmit += len(huff_v[i][j])

    for i in range(0, len(huff_table_v)):
        for j in range (0, len(huff_table_v[0])):
            v_bits_transmit += len(huff_table_v[i][j])

    #bit encoding
    total_ble = y_bits_transmit + u_bits_transmit + v_bits_transmit
    # bit original
    total_blo = len(y) * len(y[0]) * 8 + len(u) * len(u[0]) * 8 + len(v) * len(v[0]) * 8


    # ratio
    return total_blo/ total_ble


start = time.time()
# Read image
img_path = 'data/PhotoTraces_Free_RAW_Photos_05_Spring_Tree.dng'
img = cv2.imread(img_path)
#img1 = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)

with rawpy.imread(img_path) as raw:
    rgb = raw.postprocess()

img1 = rgb

#img1 = rgb

# img1 = cv2.resize(img, (3000, 1000), interpolation=cv2.INTER_LINEAR)
# cv2.imwrite('image_resize.jpg', img1)

print("Image shape: ", img1.shape)

# Convert the image to YCbCr color spaces
width, height, _ = img1.shape
ycbcr_image = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
ycbcr_image = ycbcr_image
y, u, v = cv2.split(ycbcr_image)
# Subsample Cb and Cr components using 4:2:2 sampling
subsampled_y, subsampled_u, subsampled_v = subsampling_422(ycbcr_image)
# Encoding each channel
print("Encoding channel y")
huff_y, huff_table_y = jpeg_encode(subsampled_y, quality=50)
end1 = time.time()
print("Encoding channel Cb")
huff_u, huff_table_u = jpeg_encode(subsampled_u, quality=50)
end2 = time.time()
print("Encoding channel Cr")
huff_v, huff_table_v = jpeg_encode(subsampled_v, quality=50)

# Ration compression
ratio = calculate_ratio_compression(huff_y, huff_u, huff_v, huff_table_y, huff_table_u, huff_table_v)
print("Ratio Compression: " , ratio)

huff_y_key = 'huff_y'
huff_u_key = 'huff_u'
huff_v_key = 'huff_v'
huff_table_y_key = 'huff_table_y'
huff_table_u_key = 'huff_table_u'
huff_table_v_key = 'huff_table_v'

print("huff_table y shape", np.array(huff_table_y).shape)

file_data = 'data/data_huff.npz'

# np.savez_compressed(file_data, huff_y = huff_y, huff_u = huff_u, huff_v = huff_v, 
#                     huff_table_y = np.array(huff_table_y), huff_table_u = np.array(huff_table_u), huff_table_v = np.array(huff_table_v))
np.savez_compressed(file_data, huff_y = huff_y, huff_u = huff_u, huff_v = huff_v, 
                    huff_table_y = huff_table_y, huff_table_u = huff_table_u, huff_table_v = huff_table_v)



end3 = time.time()
#Decoding
print("Decoding channel y")
y_decoded = jpeg_decode(huff_y, huff_table_y, quality=50)
end4 = time.time()
print("Decoding channel Cb")
u_decoded = jpeg_decode(huff_u, huff_table_u, quality=50)
end5 = time.time()
print("Decoding channel Cr")
v_decoded = jpeg_decode(huff_v, huff_table_v, quality=50)
end6 = time.time()
# Show result
h, w = y_decoded.shape
u_decoded = cv2.resize(u_decoded, (w, h), interpolation=cv2.INTER_LINEAR)
v_decoded = cv2.resize(v_decoded, (w, h), interpolation=cv2.INTER_LINEAR)
img_decoded = cv2.merge([y_decoded, u_decoded, v_decoded])
img_decoded = img_decoded
img_decoded = img_decoded.astype(np.uint8)

#img_decoded = jpeg_decode_full(file_data)

# Convert image to RGB
img_decoded_rgb = cv2.cvtColor(img_decoded, cv2.COLOR_YCR_CB2BGR) #ycbcr_to_rgb(img_decoded)
print(end1 - start)

cv2.imshow('input', img1)
cv2.imshow('JPEG img', img_decoded_rgb)
cv2.imwrite('data/output.jpg', img_decoded_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()





