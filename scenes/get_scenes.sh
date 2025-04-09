#!/bin/sh

# ---------------------------------------------------------------------------- #
# This script downloads 4 scenes from https://benedikt-bitterli.me/resources   #
# and adapts them to work with our version of mitsuba.                         #
# We rely on the scenes having the same form as when we downloaded them since  #
# we use sed to replace specific lines in the xml files.                       #
# If something breaks, feel free to open an issue and/or pull request.         #
# ---------------------------------------------------------------------------- #
# --------------------- LAST VERIFIED: 28. October 2024 ---------------------- #
# ---------------------------------------------------------------------------- #

### LIVING-ROOM
wget https://benedikt-bitterli.me/resources/mitsuba/living-room.zip
unzip living-room.zip
rm living-room.zip
mv living-room/scene_v0.6.xml living-room/scene.xml

sed -i '1,25c \
<?xml version="1.0" encoding="utf-8"?> \
<!DOCTYPE scene [ \
<!ENTITY header SYSTEM "../_integrators/_header.xml"> \
]> \
\
<scene version="0.5.0"> \
	\&header; \
  \
	<sensor type="perspective"> \
		<float name="fov" value="90" /> \
		<transform name="toWorld"> \
			<matrix value="0.264209 0.071763 -0.961792 5.10518 -2.81996e-008 0.997228 0.074407 0.731065 0.964465 -0.019659 0.263476 -2.31789 0 0 0 1" /> \
		</transform> \
 \
		<sampler type="independent" > \
			<integer name="sampleCount" value="$spp" /> \
		</sampler> \
 \
		<film type="hdrfilm"> \
			<integer name="width" value="1280" /> \
			<integer name="height" value="720" /> \
			<boolean name="banner" value="false" /> \
 \
			<rfilter type="box" /> \
		</film> \
	</sensor> \
' living-room/scene.xml

sed -i '140a\			<string name="channel" value="a" />' living-room/scene.xml


### BEDROOM
wget https://benedikt-bitterli.me/resources/mitsuba/bedroom.zip
unzip bedroom.zip
rm bedroom.zip

mv bedroom/scene_v0.6.xml bedroom/scene.xml
sed -i '1,25c \
<?xml version="1.0" encoding="utf-8"?> \
<!-- Bedroom by SlykDrako; Adapted by Benedikt Bitterli; Public Domain --> \
 \
<!DOCTYPE scene [ \
<!ENTITY header SYSTEM "../_integrators/_header.xml"> \
]> \
 \
<scene version="0.5.0" > \
	\&header; \
 \
	<sensor type="perspective" > \
		<float name="fov" value="65" /> \
		<transform name="toWorld" > \
			<matrix value="-0.653592 -0.0128556 -0.756738 3.45558 2.84986e-009 0.999856 -0.0169858 1.21244 0.756847 -0.0111018 -0.653498 3.29897 0 0 0 1"/> \
		</transform> \
 \
		<sampler type="independent" > \
			<integer name="sampleCount" value="$spp" /> \
		</sampler> \
 \
		<film type="hdrfilm"> \
			<integer name="width" value="1280"/> \
			<integer name="height" value="720"/> \
					<boolean name="banner" value="false"/> \
 \
			<rfilter type="box"/> \
		</film> \
	</sensor> \
' bedroom/scene.xml

sed -i '206,213c \
	<bsdf type="twosided" id="Curtains" > \
		<bsdf type="diffuse" > \
			<rgb name="reflectance" value="0.8"/> \
		</bsdf> \
	</bsdf>' bedroom/scene.xml

sed -i '721c \
      <rgb name="radiance" value="80"/>' bedroom/scene.xml


sed -i '734c \
      <rgb name="radiance" value="80"/>' bedroom/scene.xml


### COUNTRY KITCHEN
wget https://benedikt-bitterli.me/resources/mitsuba/kitchen.zip
unzip kitchen.zip
rm kitchen.zip
mv kitchen/scene_v0.6.xml kitchen/scene.xml

sed -i '1,25c \
<?xml version="1.0" encoding="utf-8"?> \
<!DOCTYPE scene [ \
<!ENTITY header SYSTEM "../_integrators/_header.xml"> \
]> \
<!-- Country Kitchen by Jay-Artist; Adapted by Benedikt Bitterli and us; CC BY 3 --> \
 \
<scene version="0.5.0" > \
	&header; \
 \
	<sensor type="perspective" > \
		<float name="fov" value="60" /> \
		<transform name="toWorld" > \
			<matrix value="-0.89874 -0.0182716 -0.4381 1.211 0 0.999131 -0.0416703 1.80475 0.438481 -0.0374507 -0.89796 3.85239 0 0 0 1"/> \
		</transform> \
		<sampler type="independent" > \
			<integer name="sampleCount" value="$spp" /> \
		</sampler> \
		<film type="hdrfilm" > \
			<integer name="width" value="1280" /> \
			<integer name="height" value="720" /> \
			<boolean name="banner" value="false" /> \
			<rfilter type="box" /> \
		</film> \
	</sensor> \
 \
	<shape type="rectangle" > \
		<transform name="toWorld" > \
			<scale y="0.5" x="0.5" /> \
			<rotate y="1" angle="30" /> \
			<matrix value="0.811764 1.25471e-014 5.48452e-022 0.060314 -9.254e-015 1.10064 6.63724e-009 1.97379 0 5.90839e-008 0.549328 -3.1173 0 0 0 1"/> \
			<translate z="1"/> \
			<translate x="-1.2"/> \
			<translate y="0.6"/> \
		</transform> \
		<bsdf type="conductor" > \
			<string name="material" value="none" /> \
		</bsdf> \
	</shape> \
' kitchen/scene.xml

sed -i '490a \
			<float name="gamma" value="1.0" />' kitchen/scene.xml

sed -i '575a \
			<float name="gamma" value="1.0" />' kitchen/scene.xml

sed -i '829,835d' kitchen/scene.xml
sed -i '2092,2098d' kitchen/scene.xml
sed -i '2706,2713d' kitchen/scene.xml
sed -i '2722,2728d' kitchen/scene.xml

sed -i '1213c \
		<ref id="WineGlasses" />' kitchen/scene.xml
sed -i '2877,2889d' kitchen/scene.xml
sed -i '2878,2916c \
	<emitter type="sunsky"> \
		<transform name="toWorld"> \
			<rotate y="1" angle="90"/> \
		</transform> \
		<float name="hour" value="9"/> \
		<float name="turbidity" value="5"/> \
		<float name="sunRadiusScale" value="4"/> \
		<float name="scale" value="50"/> \
	</emitter>' kitchen/scene.xml

### dining-room
wget https://benedikt-bitterli.me/resources/mitsuba/dining-room.zip
unzip dining-room.zip
rm dining-room.zip
mv dining-room/scene_v0.6.xml dining-room/scene.xml

sed -i '1,25c \
<?xml version="1.0" encoding="utf-8"?> \
<!DOCTYPE scene [ \
<!ENTITY header SYSTEM "../_integrators/_header.xml"> \
]> \
 \
<scene version="0.5.0" > \
  &header; \
 \
	<sensor type="perspective" > \
		<float name="fov" value="60" /> \
		<transform name="toWorld" > \
			<matrix value="-0.999914 0.000835626 0.013058 -0.587317 -5.82126e-011 0.997959 -0.063863 2.7623 -0.0130847 -0.0638576 -0.997873 9.71429 0 0 0 1"/> \
		</transform> \
    \
    <sampler type="independent"> \
			<integer name="sampleCount" value="$spp"/> \
		</sampler> \
    \
    <film type="hdrfilm"> \
			<integer name="width" value="1280"/> \
			<integer name="height" value="720"/> \
			<boolean name="banner" value="false"/> \
 \
			<rfilter type="box"/> \
		</film> \
	</sensor> \
' dining-room/scene.xml
