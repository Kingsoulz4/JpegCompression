import time
import cv2
import numpy as np

from Decode import jpeg_decode, ycbcr_to_rgb
from Encode import rgb_to_ycbcr, subsampling_422, jpeg_encode

start = time.time()
# Read image
img_path = 'old_street.jpg'
img = cv2.imread(img_path)
img1 = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
# Convert the image to YCbCr color spaces
width, height, _ = img1.shape
ycbcr_image = rgb_to_ycbcr(img1)
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
end3 = time.time()
# Decoding
print("Decoding channel y")
y_decoded = jpeg_decode(huff_y, huff_table_y)
end4 = time.time()
print("Decoding channel Cb")
u_decoded = jpeg_decode(huff_u, huff_table_u)
end5 = time.time()
print("Decoding channel Cr")
v_decoded = jpeg_decode(huff_v, huff_table_v)
end6 = time.time()
# Show result
h, w = y_decoded.shape
u_decoded = cv2.resize(u_decoded, (w, h), interpolation=cv2.INTER_LINEAR)
v_decoded = cv2.resize(v_decoded, (w, h), interpolation=cv2.INTER_LINEAR)
img_decoded = cv2.merge([y_decoded, u_decoded, v_decoded])
img_decoded = img_decoded
img_decoded = img_decoded.astype(np.uint8)
# Convert image to RGB
img_decoded_rgb = ycbcr_to_rgb(img_decoded)
print(end1 - start)
print(end4 - start)
cv2.imshow('input', img1)
cv2.imshow('JPEG img', img_decoded_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
