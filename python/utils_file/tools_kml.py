import os
import simplekml
import random

def save_coords_to_kml(dictory, filename, coords):
	kml = simplekml.Kml()
	kml.document.name = dictory.split('/')[-1]
	lin = kml.newlinestring(name=dictory.split('/')[-1], description='gps_pgo trajectory', coords=coords)
	##### Constant color
	# lin.style.linestyle.color = simplekml.Color.red
	##### Random color
	lin.style.linestyle.color = simplekml.Color.rgb(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	lin.style.linestyle.width = 6
	kml.save(os.path.join(dictory, filename))
    