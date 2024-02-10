import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
import scipy.integrate as integrate
from scipy.signal import butter
from scipy import signal

scaling_factor=45
data = pd.read_csv('imu_1.csv')

mag_field_x = data['field.MagField.magnetic_field.x'].to_numpy()
mag_field_y = data['field.MagField.magnetic_field.y'].to_numpy()

mean_x = np.mean(mag_field_x)
mean_y = np.mean(mag_field_y)

# plt.plot(mag_field_x, mag_field_y, label='Centered MagField.x vs Centered MagField.y')
# plt.legend()

# plt.axis("equal")
# plt.show()


c_mag_field_x = mag_field_x - mean_x + 0.016
c_mag_field_y = mag_field_y - mean_y


# plt.grid(color='grey', linestyle='--', linewidth=1)

# plt.plot(c_mag_field_x, c_mag_field_y, label='Centered MagField.x vs Centered MagField.y')

# plt.title(' Hard iron calibtration')
# plt.xlabel(' MagField.x ')
# plt.ylabel(' MagField.y ')
# plt.legend()

# plt.axis("equal")
# plt.show()

X_major = float(c_mag_field_x[2000])
Y_major = float(c_mag_field_y[2000])
print(X_major)
print(Y_major)
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
q = 1.55   #minor axis
R1 = 1.75
sigma = q/R1
print('sigma = ', sigma)

matrix_2 = [[1, 0 ],[0, sigma]]
mat = np.matmul(matrix_2, matrix)
theta = -theta
R1 = [[np.cos(theta), np.sin(theta)], [np.sin(-theta), np.cos(theta)]]
v1 = np.matmul(R1, mat)
# plt.figure(figsize=(10, 12))
# plt.grid(color='grey', linestyle='--', linewidth=1)
# plt.plot(mat[0], mat[1], label = ' Calibrated')
# plt.plot(mag_field_x, mag_field_y, label='raw MagField.x vs raw MagField.y')
# plt.title('Calibrated vs Raw Magnetic Field X vs Y')
# plt.legend()
# plt.axis("equal")
# plt.show()

w = data['field.IMU.orientation.w'].to_numpy()
x = data['field.IMU.orientation.x'].to_numpy()
y = data['field.IMU.orientation.y'].to_numpy()
z = data['field.IMU.orientation.z'].to_numpy()
Time =data['%time']- data['%time'].min()
Time = Time.to_numpy()

t0 = +2.0 * (w * x + y * z)
t1 = +1.0 - 2.0 * (x * x + y * y)
roll = np.arctan2(t0, t1)

t2 = +2.0 * (w * y - z * x)
pitch = np.arcsin(t2)

t3 = +2.0 * (w * z + x * y)
t4 = +1.0 - 2.0 * (y * y + z * z)
yaw = np.arctan2(t3, t4)
yaw_unwrapped = np.unwrap(yaw)

mag_x = v[0] 
mag_y = v[1]  

#calibrated mag yaw
yaw_a = np.arctan2(mag_y, mag_x)
yaw_an = np.unwrap(yaw_a)

# yaw_angle=[]
# # yaw_angle=np.arctan(mag_y/mag_x)
# for i in range(len(mag_x)):
#      yaw_angle.append(np.arctan(mag_y[i]/mag_x[i]))
# yaw_angle_unwrap = np.unwrap(yaw_angle)


# plt.figure(figsize=(10, 5))
# plt.plot(yaw, label='Yaw')
# plt.plot(yaw_unwrapped, label='Yaw w')
# plt.xlabel('Timestamp')
# plt.ylabel('Yaw')
# plt.title('Yaw from IMU Data')
# plt.legend()
# plt.grid(True)
# plt.show()

#raw yaw
yaw_raw = np.arctan2(mag_field_y, mag_field_x)
yaw_raw_unwrapped = np.unwrap(yaw_raw)

gyro = data['field.IMU.angular_velocity.z'].to_numpy()
gyro_int = cumtrapz(gyro, initial=0)


normalized_mag_vecs = np.array([(x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)) for x, y in zip(mag_x, mag_y)])
yaw_angles_from_mag = np.array([math.atan2(y, x) * -1 for x, y in normalized_mag_vecs])
yaw_cali_unwrapped = np.unwrap(yaw_angles_from_mag)*scaling_factor 


# #yaw plot
# plt.plot(gyro_int, label='Gyro Yaw', c='palevioletred')
# plt.plot(yaw_unwrapped, label='IMU yaw')
# #plt.plot(yaw_raw_unwrapped, label='mag yaw')

# plt.plot(yaw_cali_unwrapped, label='Yaw from Calibrated Mag Data')

# plt.legend(loc='upper right', fontsize='x-large')
# plt.grid(color='grey', linestyle='--', linewidth=1)
# plt.title('Estimation of Yaw for Magnetometer')
# plt.xlabel('samples')
# plt.ylabel('Yaw')
# plt.show()



# # raw mag yaw plot
# plt.figure(figsize=(10, 5))
# plt.plot(yaw_raw_unwrapped, label='Yaw from Non-Calibrated Mag Data')
# plt.xlabel('Samples')
# plt.ylabel('Yaw (radians)')
# plt.title('Yaw from Non-Calibrated Magnetometer Data')
# plt.legend()
# plt.grid(True)
# plt.show()

lpf = signal.filtfilt(*butter(3, 0.1, "lowpass",fs = 40, analog=False), yaw_cali_unwrapped)
hpf = signal.filtfilt(*butter(3, 0.0001, 'highpass', fs = 40, analog=False), gyro_int)



# complimetary filter 
alpha = 0.75
omega = data['field.IMU.angular_velocity.z'].to_numpy()
yaw_filtered = []
yaw_filtered = np.append(yaw_filtered,0)
for i in range(len(hpf)-1):
  j = i+1
  yaw_filtered = np.append(yaw_filtered, alpha*(yaw_filtered[i] + hpf[j]*0.05) + ((1-alpha)*lpf[j]))
# lpf1 = 1 - hpf1
# yaw_filtered = (hpf1*hpf) + (lpf1*lpf)

# plt.figure(figsize=(16, 8))
# plt.plot(yaw_filtered, label='Complementary Filter')
# plt.plot(yaw_unwrapped, label='IMU yaw')
# plt.legend(loc='lower right', fontsize='x-large')
# plt.grid(color='grey', linestyle='--', linewidth=1)
# plt.xlabel('Samples @ 40 Hz')
# plt.ylabel('Yaw')
# plt.title(' Complementary Filter Yaw')
# plt.show()

# plt.figure(figsize = (16,8))
# plt.plot(lpf, label='LPF')
# plt.plot(hpf, label='HPF')
# plt.plot(yaw_filtered, label='Complementary Filter')
# plt.legend(loc='upper right', fontsize='x-large')
# plt.grid(color='grey', linestyle='--', linewidth=1)
# # plt.plot(hpf, label = 'HPF Gyro Yaw', c='seagreen')
# plt.legend(loc='upper right', fontsize='x-large')
# plt.grid(color='grey', linestyle='--', linewidth=1)
# plt.title('LPF for Magnetic Yaw and HPF for Gyro Yaw')
# plt.xlabel('Samples @ 40Hz')
# plt.ylabel('Yaw (degrees)')
# plt.show()


data = pd.read_csv('imu_1.csv')
gpsdata = pd.read_csv('gps_1.csv')

raw_val_imu = data['field.IMU.linear_acceleration.x'].to_numpy()
meanx = np.mean(raw_val_imu)
linear_acc = raw_val_imu - meanx
Forward_velocity_ = integrate.cumtrapz(linear_acc, initial=0)
Forward_velocity_ = Forward_velocity_/40
Forward_velocity_ = Forward_velocity_[1:]

# try avg(40 step)
averages = np.array([]) 
step = 40
for i in range(0, len(linear_acc), step):
    # Calculate the average of every 1st and 40th point
    average_value = np.mean(linear_acc[i:i+step])
    averages = np.append(averages, average_value)

print('avg=', len(averages))  
Forward_velocity_raw_2 = cumtrapz(averages, initial=0)
Forward_velocity_[Forward_velocity_<0]=0

data['%time'] = data['%time'].astype(float)
data['time_in_seconds'] = (data['%time'] - data['%time'].min()) / 1e9
time_s= data['time_in_seconds'].to_numpy()
print(len(time_s))

# tym1 =str(float(str(time))-1.698267706*1e+18)
# sec.append(float(tym1[:3]+'.'+tym1[-11:-2])-100.673790976)

UTMeast = gpsdata['field.UTM_Easting'].to_numpy()
UTMnorth = gpsdata['field.UTM_Northing'].to_numpy()
# Latitude = gpsdata['field.Latitude'].to_numpy()
# Longitude = gpsdata['field.Longitude'].to_numpy()
distance=[]
for i in range(len(UTMeast)-1):
  distance = np.append(distance, math.sqrt(((UTMnorth[i + 1] - UTMnorth[i]) ** 2) + (UTMeast[i + 1] - UTMeast[i]) ** 2))
print(len(distance))
gps_vel= distance 


difference = []
for i in range(len(linear_acc)-1):
  difference = np.append(difference, (linear_acc[i + 1] - linear_acc[i]) / (0.025))
print(difference)
print( ' dif =', len(difference))

adjustment = linear_acc[1:] - difference
lpf_acc = signal.filtfilt(*butter(12, 1, "lowpass",fs = 40, analog=False), adjustment)

Forward_velocity_adjusted = cumtrapz(lpf_acc, initial=0)
Forward_velocity_adjusted[Forward_velocity_adjusted<0] = 0
Forward_velocity_adjusted = Forward_velocity_adjusted/40

print ( 'lenadj=' , len(Forward_velocity_))
# cutoff_frequency = 1
# sampling_frequency = 40.0
# filter_order = 12
# filtered_acceleration = butter_lowpass_filter(acceleration[:, 0], cutoff_frequency, sampling_frequency,filter_order)
# vel_from_acc = cumtrapz(filtered_acceleration, dx=time_intervals, initial=0.0)

# print(distance)
plt.plot(gps_vel, label = "gps velocity") 
plt.plot( time_s[1:], Forward_velocity_adjusted, label='IMU corrected Velocity', c='palevioletred')
plt.plot( Forward_velocity_raw_2, label='IMU Raw Velocity', c='teal')
plt.legend(loc='upper right', fontsize='x-large')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.title('Forward velocity from IMU and GPS after adjustment')
plt.xlabel('Time (secs)')
plt.ylabel('Velocity (m/sec)')
plt.show()

Forward_velocity_adjusted[Forward_velocity_adjusted<10] = 0
w = data['field.IMU.orientation.w'].to_numpy()
x = data['field.IMU.orientation.x'].to_numpy()
y = data['field.IMU.orientation.y'].to_numpy()
z = data['field.IMU.orientation.z'].to_numpy()



t3 = +2.0 * (w * z + x * y)
t4 = +1.0 - 2.0 * (y * y + z * z)
yaw = np.arctan2(t3, t4)
yaw = yaw[1:]

gps_heading = np.arctan2((UTMeast[0] - UTMeast[1]), (UTMnorth[0] - UTMnorth[1]))
yaw = yaw_an.ravel()
# yaw = yaw * -1 + gps_heading

yaw = yaw[1:]


rotation_degrees = 100
rotation_radians = np.deg2rad(rotation_degrees)

# Rotate the yaw angles
r_yaw = yaw + rotation_radians

# Ensure the result is within the range of -π to π (or -180° to 180°)
r_yaw = np.where(r_yaw > np.pi, r_yaw - 2 * np.pi, r_yaw)
r_yaw = np.where(r_yaw < -np.pi, r_yaw + 2 * np.pi, r_yaw)

# forward_vel = np.unwrap(Forward_velocity_adjusted)
forward_vel = np.unwrap(Forward_velocity_)
# forward_vel = gps_vel
time_intervals = 0.025
v_e = [] #velocity easting
v_n = [] #velocity northing

for i in range(len(forward_vel)):
    v_e.append(forward_vel[i]*np.cos(r_yaw[i]))
    v_n.append(forward_vel[i]*np.sin(r_yaw[i]))

initial_x_position = 560
initial_y_position = 1400


delta_x = cumtrapz(v_e, dx=time_intervals, initial=0)
delta_y = cumtrapz(v_n, dx=time_intervals, initial=0)


trajectory_x = [initial_x_position + dx for dx in delta_x]
trajectory_y = [initial_y_position + dy for dy in delta_y]



offset_easting = 332000
offset_northing = 4680750

UTMeast_1= UTMeast - offset_easting
UTMnorth_1 = UTMnorth - offset_northing
plt.figure(figsize = (12,12))
plt.plot(trajectory_x,trajectory_y, label = "IMU Estimated Trajectory")
plt.plot(UTMeast_1,UTMnorth_1, label = "gps Estimated Trajectory")
plt.title("Estimated Trajectory")
plt.xlabel("X position (meter)")
plt.ylabel("Y position (meter)")
plt.legend()
plt.grid(True)
plt.show()


