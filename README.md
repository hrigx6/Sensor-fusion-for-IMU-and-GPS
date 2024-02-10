# Sensor-fusion-for-IMU-and-GPS

LAB4- Navigation with IMU and Magnetometer
Introduction:  In this LAB we are using two sensors. Firstly, VN-100 IMU sensor: An Inertial Measurement Unit (IMU) is a sensor module that typically combines multiple sensors to measure various aspects of an object's motion and orientation. It commonly includes an accelerometer to measure linear acceleration, a gyroscope to measure angular velocity, and sometimes a magnetometer to detect magnetic fields. Secondly a GPS sensor, GPS (Global Positioning System) sensors work by receiving signals from a network of satellites orbiting the Earth.
Objectives: 
1.	 Collect data from both GPS and IMU sensors during vehicle motion, allowing you to compare their measurements for further analysis.
2.	Correct magnetometer readings for "hard-iron" and "soft-iron" effects during compass calibration and estimate heading (yaw).
3.	Estimate forward velocity by integrating forward acceleration, cross-checking with GPS measurements.
4.	Analyze the results, make necessary adjustments to sensor measurements, and ensure the velocity plot aligns with expectations.
5.	Perform dead reckoning with IMU data to obtain displacement, estimating the vehicle's trajectory and comparing it with GPS data for validation.

MAGNETOMETER DATA CALIBRATION:
RAW DATA: For calibration data was acquired by driving the car in circles and mag_X and mag_Y Values were plotted.
Certainly, here are concise explanations:

Soft Iron Effects:
Soft iron effects refer to the temporary distortion of magnetic fields caused by nearby ferrous materials or magnets. When a magnetic sensor encounters soft iron, it results in a reversible alteration of the magnetic field's strength and direction. This distortion can be corrected by rotation to the correct distorted angle and correcting the ellipse deviation.

Hard Iron Effects:
Hard iron effects involve permanent magnetization of materials, creating a consistent magnetic field offset. These effects result in a fixed and constant deviation in sensor readings, as they introduce a static magnetic bias. Correcting for hard iron effects typically requires calibration to eliminate the persistent offset in sensor data.
RAW MAG DATA(Fig-1):
 
Hard Iron Calibration (Fig-2):
 
For the hard iron calibration mean value was subtracted from x and y so that the graph is centered. The mean still has a little further offset which was subtracted from it.


Soft iron calibration (Fig-3):
 
For the soft iron calibration, rotation matrix was multiplied to correct the rotation by calculation theta and ellipse was translated into a circle by multiplying it with sigma (major axis/minor axis) whose values were calculated by taking the ratio of axis calculated visually. Maintaining that ratio major and minor axis values were estimated. 
Comparing raw and calibrated data(Fig-4):
 
Mag_x vs Mag_y
In this final calibration both hard and soft iron errors were eliminated, and the corrected data is further used to analyze the path data.
YAW ESTIMATION:
Yaw estimation involved multiple steps. Initially, the yaw angle was determined using calibrated magnetometer data by applying the arctan function to the magnetic field components. Then, gyro data was integrated over time to compute another yaw estimate. To enhance accuracy, a complementary filter was employed, combining the magnetometer-based estimate (which excels in stable, low-frequency changes) and the gyro-based estimate (suitable for high-frequency variations). This approach effectively fused the two sources of data, producing a more reliable and robust yaw angle estimation that aligned closely with the IMU's provided yaw readings.
 
Fig -5. Calibrated mag yaw angle estimation
 
Fig -6. Comparison of yaw angle

After applying the filter, the yaw estimation can be seen similar to the yaw calculated by IMU.
More smoothen out than the raw mag yaw and gyro yaw and visually comparable to the IMU yaw. This can say that IMU essentially uses a complimentary filter within for yaw estimation.
 
Fig-7: comparison between IMU yaw and complimentary filter yaw estimation
 
Fig-8: filter comparison.




Low-Pass Filter (LPF) Allows low-frequency components while suppressing high frequencies.
Parameters: Filter Order (3), Cutoff Frequency (0.1 * Nyquist), Filter Type ("lowpass"), Sampling Frequency (40)
High-Pass Filter (HPF) Permits high-frequency components while attenuating low frequencies.
Parameters: Filter Order (3), Cutoff Frequency (Very close to zero), Filter Type ("highpass"), Sampling Frequency (40)
These filters are employed to remove noise and unwanted frequency components, focusing on relevant signal information for analysis and visualization.
The table shows that the complementary filter yaw estimate is the closest to the IMU's yaw reading. This is because the complementary filter fuses the magnetometer-based estimate and the gyro-based estimate in a way that takes advantage of the strengths of each method. The magnetometer-based estimate is accurate at low frequencies, while the gyro-based estimate is accurate at high frequencies. The complementary filter combines these two estimates to produce a more reliable and robust yaw angle estimation.
- Reliability wise sensor fusion techniques, such as a complementary filter or Kalman filter, to combine the strengths of both sensors and mitigate their respective limitations are better. This allows for more reliable and accurate yaw estimation. Magnetometer data can provide an absolute reference, while gyroscope data can enhance responsiveness and help compensate for drift. reframe
Forward velocity:
In this study, we integrated forward acceleration measurements to estimate forward velocity. We also calculated an estimate of the velocity from GPS measurements. We then plotted both velocity estimates and observed that the integrated acceleration velocity estimate was noisy and drifted over time.
There are certain discrepancies between the velocities, which can be due to the Bias which can the rectified. Noisy data collection and unnecessary frequencies are collected. Which are further amplified due to integration. As to calculate velocity we integrate the raw acceleration.
To improve the accuracy of the integrated velocity estimate, we subtracted a bias from the acceleration measurements. The bias was estimated by calculating the average acceleration over a long period of time. After subtracting the bias, the integrated acceleration velocity estimate was closer to the GPS velocity estimate.

 
Fig-9: Raw velocity




GPS velocity (Fig-10):
 
GPS velocity was calculated by calculating  distance between two northing and easting points and dividing it by time.

 
Fig-11: velocity matching after correction
Later, other than the bias, a threshold value was used so that the minor acceleration values under that threshold will be considered as stationary.
Discrepancies between velocity estimates from accelerometers and GPS sensors can arise due to differences in precision, noise, and integration methods. Accelerometers are prone to sensor noise and cumulative integration errors, whereas GPS may have latency and environmental issues. Combining the two sensor outputs using sensor fusion techniques can help mitigate these discrepancies and provide more accurate velocity estimates for navigation.








DEAD Reckoning: We are using Dead reckoning to predict the path and compare it to the gps path. The primary objective is to use the IMU data to do integration and determine displacement. Following that, we will attempt to reconstruct the vehicle's journey using the GPS reference. The process includes determining the vehicle's trajectory using IMU data, matching it with the GPS-derived path, and analyzing the similarities and differences between the two datasets to determine the accuracy and dependability of the IMU-based navigation system.
 For dead reckoning initially the acceleration along x was integrated to calculate forward velocity after subtracting bias and putting a threshold the velocity was corrected. The corrected velocity was then further used. GPS heading was calculated by the starting values of the data( initial data values). It was then added to the Yaw calculated by imu(orientation of z) so that our plot starts from that point and in that direction.
The two components of the velocity vector were calculated by sin (yaw and cos( yaw). This can be said as velocity of easting and velocity of northing and those components were further integrated to obtain displacement which is the x and y values of the plot. These values were plotted to obtain the plot of the trajectory.
Initial points of the trajectory was manually defined from eyeballing the Gps data’s starting point. So that the comparison becomes easier.



 
Fig-12: (Without yaw correction/ uncalibrated yaw)


Actual path:
 
Fig-13: UTM-easting vs UTM-northing
Estimated trajectory:
           
      Fig-14: Trajectory with Calibrated yaw                             Fig-15:Trajectory with Calibrated yaw
               (without heading correction)                                            (with heading correction)  



 
Fig-16 Actual vs predicted path (After offsetting)
The deviation in the path is due to the discrepancies in the forward velocity. If the forward velocity is further corrected, then the path prediction can be improved. Altering the forward velocity resulted in change in the path due to which it can be concluded that further correction can improve the trajectory estimation.
It can be said that VN-100 from vectornav might be able to navigate without a position fix for more than a few minutes using the dead reckoning technique. But it shall only be enforced in emergency situations. This technique might be useful to navigate for the time being and can give decent results considering the velocity calculations are matching. Current data representation doesn’t give 2m accuracy at the moment but can be enforced with further implementations.
Also, the calculations were done with a belief that when calculating Xc, we initially assume that the vehicle's center of mass aligns perfectly with the vehicle frame. However, it's important to recognize that this assumption may not always be held in real-world scenarios. In such cases, we must account for deviations from this ideal alignment and adjust our calculations accordingly. By considering Xc in the calculations, better results can be obtained.
