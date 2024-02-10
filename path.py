import rosbag
import matplotlib.pyplot as plt

bag = rosbag.Bag('1.bag')


easting_data = []
northing_data = []

offset_easting = 332000
offset_northing = 4680800

for _, msg, _ in bag.read_messages():
  if hasattr(msg, 'UTM_Easting') and hasattr(msg, 'UTM_Northing'):
    easting_data.append(msg.UTM_Easting - offset_easting) 
    northing_data.append(msg.UTM_Northing - offset_northing )

bag.close()  

plt.figure(figsize=(12, 8))
plt.scatter(easting_data, northing_data, s=10, c='b', label='GPS Data')

plt.xlabel('UTM_Easting')
plt.ylabel('UTM_Northing')
plt.title('Northing vs Easting')

plt.grid(True)
plt.legend()

plt.xlim(min(easting_data)-15, max(easting_data)+15)  
plt.ylim(min(northing_data)-15, max(northing_data)+15)

plt.show()