import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import butter
from scipy import signal

data = pd.read_csv('imu_1.csv')
gpsdata = pd.read_csv('gps_1.csv')

raw_val_imu = data['field.IMU.linear_acceleration.x'].to_numpy()
meanx = np.mean(raw_val_imu)
linear_acc = raw_val_imu - meanx
Forward_velocity_raw = cumtrapz(linear_acc, initial=0)
Forward_velocity_raw = Forward_velocity_raw/40

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

adjustment = linear_acc[1:] - difference
lpf = signal.filtfilt(*butter(12, 1, "lowpass",fs = 40, analog=False), adjustment)

Forward_velocity_adjusted = cumtrapz(lpf, initial=0)
Forward_velocity_adjusted[Forward_velocity_adjusted<0] = 0
Forward_velocity_adjusted = Forward_velocity_adjusted/40


# cutoff_frequency = 1
# sampling_frequency = 40.0
# filter_order = 12
# filtered_acceleration = butter_lowpass_filter(acceleration[:, 0], cutoff_frequency, sampling_frequency,filter_order)
# vel_from_acc = cumtrapz(filtered_acceleration, dx=time_intervals, initial=0.0)

# print(distance)
plt.plot(gps_vel, label = "gps velocity") 

plt.plot( time_s[1:], Forward_velocity_adjusted, label='IMU Raw Velocity', c='palevioletred')
plt.legend(loc='upper right', fontsize='x-large')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.title('Forward velocity from IMU and GPS after adjustment')
plt.xlabel('Time (secs)')
plt.ylabel('Velocity (m/sec)')
plt.show()


w = data['field.IMU.orientation.w'].to_numpy()
x = data['field.IMU.orientation.x'].to_numpy()
y = data['field.IMU.orientation.y'].to_numpy()
z = data['field.IMU.orientation.z'].to_numpy()



t3 = +2.0 * (w * z + x * y)
t4 = +1.0 - 2.0 * (y * y + z * z)
yaw = np.arctan2(t3, t4)
yaw = yaw[1:]

Forward_v = np.unwrap(Forward_velocity_adjusted)
mag = data['field.MagField.magnetic_field.x'].to_numpy()
rot = (-108*np.pi/180)
# print(yaw)
I1 = np.cos(yaw+rot)*Forward_v
I2 = -np.sin(yaw+rot)*Forward_v
I3 = np.cos(yaw+rot)*Forward_v
I4 = np.sin(yaw+rot)*Forward_v
# rads = (180/np.pi)
ve = I1+I2
vn = I3+I4
xe = cumtrapz(ve)
xn = cumtrapz(vn)

plt.figure(figsize = (8,8))
plt.plot((xe),-xn, c='b')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.title('Trajectory of Vehicle')
plt.xlabel('Xe')
plt.ylabel('Xn')
plt.plot()
plt.show()


