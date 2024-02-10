import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('imu_1.csv')

mag_field_x = data['field.MagField.magnetic_field.x'].to_numpy()
mag_field_y = data['field.MagField.magnetic_field.y'].to_numpy()

mean_x = np.mean(mag_field_x)
mean_y = np.mean(mag_field_y)

c_mag_field_x = mag_field_x - mean_x + 0.016
c_mag_field_y = mag_field_y - mean_y

X_major = float(c_mag_field_x[2000])
Y_major = float(c_mag_field_y[2000])

r = math.sqrt((X_major**2) + (Y_major**2))
print('radius = ', r)
theta = np.arcsin((Y_major/r))
print('theta = ', theta)

R = [[np.cos(theta), np.sin(theta)], [np.sin(-theta), np.cos(theta)]]
v = [c_mag_field_x, c_mag_field_y]

matrix = np.matmul(R, v)
print(np.shape(matrix))

# plt.grid(color='grey', linestyle='--', linewidth=1)
# plt.plot(matrix[0], matrix[1], label = 'Soft-Iron Calibrated')

# plt.title('Soft_Iron_Calibration Of Magnetic Field X vs Y')
# plt.xlabel('Soft_Iron_X (Guass)')
# plt.ylabel('Soft_Iron_Y (Guass)')
# plt.legend()
# plt.axis("equal")
# plt.show()
q = 1.55
R = 1.75
sigma = q/R
print('sigma = ', sigma)

matrix_2 = [[1, 0 ],[0, sigma]]
mat = np.matmul(matrix_2, matrix)

mag_x = mat[0] 
mag_y = mat[1]  


# Calculate yaw angles from magnetometer data
normalized_mag_vecs = np.array([(x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)) for x, y in zip(mag_x, mag_y)])
yaw_angles_cali_mag = np.array([math.atan2(y, x) * -1 for x, y in normalized_mag_vecs])
yaw_cali_unwrapped = np.unwrap(yaw_angles_cali_mag)

normalized_mag_vecs_2= np.array([(x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)) for x, y in zip(mag_field_x, mag_field_y)])
yaw_angles_raw_mag = np.array([math.atan2(y, x) * -1 for x, y in normalized_mag_vecs])

yaw_angles_from_mag = np.array([math.atan2(mag_field_y[i], mag_field_x[i]) * -1 for i in range(len(mag_field_x))])
yaw_unwrapped = np.unwrap(yaw_angles_from_mag)

# # Plot raw magnetometer data
# plt.figure(figsize=(10, 5))
# plt.plot(mag_field_x, label='Raw MagField.x')
# plt.plot(mag_field_y, label='Raw MagField.y')
# plt.xlabel('Samples')
# plt.ylabel('Magnetic Field (Gauss)')
# plt.title('Raw Magnetometer Data')
# plt.legend()
# plt.grid(True)

# Plot yaw angles calculated from raw magnetometer data
plt.figure(figsize=(10, 5))
plt.plot(yaw_unwrapped, label='Yaw from calibrated Magnetometer Data')
# plt.plot(yaw_angles_raw_mag, label='Yaw from raw Magnetometer Data')
plt.xlabel('Samples')
plt.ylabel('Yaw (radians)')
plt.title('Yaw from Raw Magnetometer Data')
plt.legend()
plt.grid(True)

plt.show()
