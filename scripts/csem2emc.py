from evaluate_csem import evaluate_csem
from netCDF4 import Dataset
import numpy as np
import time
import yaml


#- Input. ---------------------------------------------------------------------

#- Open the input yaml file.

fid=open('parameters.yml','r')
info=yaml.load(fid)
fid.close()


# Name under which the netCDF file will be stored.
filename=info["filename"]
# Short description of the model.
title=info["title"]
# ID of the model
mid=info["id"]
# Reference, e.g., to publication.
reference=info["reference"]
# Reference to a website.
references=info["references"]
# Short summary of the model.
summary=info["summary"]
# Keywords.
keywords=info["keywords"]
# Creator name.
creator_name=info["creator_name"]
# Creator url.
creator_url=info["creator_url"]
# Creator email.
creator_email=info["creator_email"]
# Institution.
institution=info["institution"]
# Acknowledgement.
acknowledgement=info["acknowledgement"]
# Comment.
comment=info["comment"]

# Geographical extent.
lat_min=np.float(info["lat_min"])
lat_max=np.float(info["lat_max"])
lat_increment=np.float(info["lat_increment"])

lon_min=np.float(info["lon_min"])
lon_max=np.float(info["lon_max"])
lon_increment=np.float(info["lon_increment"])

depth_min=np.float(info["depth_min"])
depth_max=np.float(info["depth_max"])
depth_increment=np.float(info["depth_increment"])

# Parameters to include.
include_vsv=np.bool(info["include_vsv"])
include_vsh=np.bool(info["include_vsh"])
include_vpv=np.bool(info["include_vpv"])
include_vph=np.bool(info["include_vph"])
include_eta=np.bool(info["include_eta"])
include_rho=np.bool(info["include_rho"])

#- Make coordinates. ----------------------------------------------------------

lat=np.arange(lat_min,lat_max+lat_increment,lat_increment)
lon=np.arange(lon_min,lon_max+lon_increment,lon_increment)
depth=np.arange(depth_min,depth_max+depth_increment,depth_increment)

#- Initialise netCDF file. ----------------------------------------------------

#- Initialise and set basic information
m=Dataset(filename,"w",format="NETCDF3_CLASSIC")

m.title=title
m.id=mid
m.reference=reference
m.references=references
m.summary=summary
m.keywords=keywords
m.Conventions="CF-1.0"
m.Metadata_Conventions= "Unidata Dataset Discovery v1.0"
m.creator_name=creator_name
m.creator_url=creator_url
m.creator_email=creator_email
m.institution=institution
m.acknowledgement=acknowledgement
m.history="created " + time.ctime(time.time())
m.comment=comment

m.geospatial_lat_min=lat_min
m.geospatial_lat_max=lat_max
m.geospatial_lat_units="degrees_north"
m.geospatial_lat_resolution=lat_increment
m.geospatial_lon_min=lon_min
m.geospatial_lon_max=lon_max
m.geospatial_lon_units="degrees"
m.geospatial_lon_resolution=lon_increment
m.geospatial_vertical_min=depth_min
m.geospatial_vertical_max=depth_max
m.geospatial_vertical_units="km"
m.geospatial_vertical_positive="down"

#- Dimensions. 

m.createDimension("depth",len(depth))
m.createDimension("latitude",len(lat))
m.createDimension("longitude",len(lon))

#- Spatial variables.

depths=m.createVariable("depth","f4",("depth",))
lats=m.createVariable("latitude","f4",("latitude",))
lons=m.createVariable("longitude","f4",("longitude",))

lats.long_name="latitude; positive north"
lats.units="degrees_north"
lats.standard_name="latitude"

lons.long_name="longitude; positive east"
lons.units="degrees_east"
lons.standard_name="longitude"

depths.long_name="depth below earth surface"
depths.units="km"
depths.positive="down"

#- Model variables.

if include_vsv:
	vsv=m.createVariable("vsv","f4",("depth","latitude","longitude",))
	vsv.long_name="SV-wave velocity"
	vsv.display_name="Vsv [km/s]"
	vsv.units="km/s"
	vsv.missing_value=99999.0

if include_vsh: 
	vsh=m.createVariable("vsh","f4",("depth","latitude","longitude",))
	vsh.long_name="SH-wave velocity"
	vsh.display_name="Vsh [km/s]"
	vsh.units="km/s"
	vsh.missing_value=99999.0

if include_vpv: 
	vpv=m.createVariable("vpv","f4",("depth","latitude","longitude",))
	vpv.long_name="PV-wave velocity"
	vpv.display_name="Vpv [km/s]"
	vpv.units="km/s"
	vpv.missing_value=99999.0

if include_vph: 
	vph=m.createVariable("vph","f4",("depth","latitude","longitude",))
	vph.long_name="PH-wave velocity"
	vph.display_name="Vph [km/s]"
	vph.units="km/s"
	vph.missing_value=99999.0

if include_eta: 
	eta=m.createVariable("eta","f4",("depth","latitude","longitude",))
	eta.long_name="eta"
	eta.display_name="eta [1]"
	eta.units="1"
	eta.missing_value=99999.0

if include_rho: 
	rho=m.createVariable("rho","f4",("depth","latitude","longitude",))
	rho.long_name="density"
	rho.display_name="density [kg/m3]"
	rho.units="kg/m3"
	rho.missing_value=99999.0

#- Compute Cartesian coordinates and get material properties. -----------------

#- Cartesian coordinates.

x=[]
y=[]
z=[]

r=6371.0-depth # radius in km
phi=lon*np.pi/180.0 # longitude in rad
theta=(90.0-lat)*np.pi/180.0 

for sr in r:
	for stheta in theta:
		for sphi in phi:

			x.append(sr*np.cos(sphi)*np.sin(stheta))
			y.append(sr*np.sin(sphi)*np.sin(stheta))
			z.append(sr*np.cos(stheta))

#- Get grid_data from CSEM evaluation at Cartesian grid points.


grid_data = evaluate_csem(x,y,z)


#- Assign variables. ----------------------------------------------------------

lats[:]=lat
lons[:]=lon
depths[:]=depth

n=0

for i in range(len(depth)):
	for j in range(len(lat)):
		for k in range(len(lon)):

			if include_vsv: vsv[i,j,k]=grid_data.df['vsv'][n]
			if include_vsh: vsh[i,j,k]=grid_data.df['vsh'][n]
			if include_vpv: vpv[i,j,k]=grid_data.df['vpv'][n]
			if include_vph: vph[i,j,k]=grid_data.df['vph'][n]
			if include_eta: eta[i,j,k]=grid_data.df['eta'][n]
			if include_rho: rho[i,j,k]=grid_data.df['rho'][n]
			n+=1

#- Clean up. ------------------------------------------------------------------

m.close()