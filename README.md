<img src="pics/bioverse.png">

# Bioverse Labs (deep-learning module)
The deep-learning module incorporate essential procedures to prepare remote sensing images mostly for automatic classification methods, in this case specifically, deep-learning approaches. 

Beyond the routines, three main methods have been developed: image tiling, vector tiling, annotation images preparation, among other independent preprocessing procedures (available in `/scripts`).

**All modules available here are under construction. Therefore, many errors and malfunctions can occur.**

1. [Setting up your environment](#1-Setting-up-your-environment)
2. [Prepare your virtual environment](#2-Prepare-your-virtual-environment)
3. [Examples ](#3-Examples)
4. [TODO-list](#4-TODO-list)

# Setting up your Python environment
This command-line solution was mainly use for image processing analysis, for specific applications and specific purposes. Most of the methods and libraries to solve these particular events might have alternatives 

## Python version and OS
The `image-processing` was developed using Python version 3.7+, and **Linux 19.10 eoan** operational system. 

# Prepare your virtual environment
First of all, check if you have installed the libraries needed:
```
sudo apt-get install python3-env
```
then, in the
```
python -m venv .venv
```
and activates it:
```
source .venv/bin/activate
```
as soon you have it done, you are ready to install the requirements.

## Preparing your `.env` file

This library uses decoupling, which demands you to set up variables that is only presented locally, for instance, the path you want to save something, or the resources of your project. In summary, your environment variables. So, copy a paste the file `.env-example` and rename it to `.env`. Afterwards, just fill out each of the variables content 
within the file:
```
DL_DATASET=PATH_TO_TRAINING_FOLDER
```

## Installing `requirements.txt`
```
pip install -r requirements.txt
```

## The `settings.py` file:
This file centralized all constants variable used in the code. Until now, only the constants below were required. The shapefiles usually has with it a attribute table, where the geographic geometry and metadata are stored:
<img src="pics/attribute-table.png">

This shapefile should have at least one column for the class name. The class name to be read is indicated by `CLASS_NAME` variable. This shapefile could also have multiple classes, for each one, a color should be specified. The values specified in `CLASSES` gives either the exact name of the class as its respective RGB color.  
```
VALID_RASTER_EXTENSION = (".tif", ".tiff", ".TIF", ".TIFF")
VALID_VECTOR_EXTENSION = ".shp"

CLASS_NAME = 'class' # <---- column name in attribut table
CLASSES = {
    "nut": [102, 153, 0],
    "palm": [153, 255, 153]
}
```
annotation image after processing (with RGB color (102,153,0) for `nut` class):
<img src="pics/annotation.png">

# Examples 
## Running the raster tilling:
The raster tiling consists in crop the full image in small peaces in order to get a required dimension for most of the supervised classifiers. 

Here, the procedure demands 6 arguments, which are the procedure to be executed (`-procedure`), the full image path (TIFF format - `-image`), the output to the tiles (`-output`), the width dimension of the tiles (`-tile_width`), the height dimension of the tiles (`-tile_height`), and a boolean verbose outcomes (`-verbose`). 
```
python main.py -procedure tiling_raster 
               -image PATH_TO_FULL_IMAGE_IN_TIFF_FORMAT  
               -output PATH_TO_OUTPUT_RASTER_TILES
               -tile_width DIMESION_OF_TILES 
               -tile_height DIMESION_OF_TILES 
               -verbose BOOLEAN
```
**In Linux, it should be run in main folder**

## Running the vector tilling:
The vector tiling consists in crop the full shapefile in small peaces according to the raster tiles extends processed before. The procedure saves a tiled shapefile if the full shapefile intersect a raster tile. If it does not intersect, the raster tile is then deleted (in order to save space). 

Here, the procedure demands 6 arguments, which are the procedure to be executed (`-procedure`), the full shapefile path (SHP format - `-image_tiles`), the output to the tiles (`-output`), the shapefile reference (`-shapefile_reference`), and a boolean verbose outcomes (`-verbose`). 
```
python main.py -procedure tiling_vector
               -image_tiles PATH_TO_OUTPUT_RASTER_TILES  
               -output PATH_TO_OUTPUT_VECTOR_TILES
               -shapefile_reference PATH_TO_REFERENCE_SHAPEFILES               
               -verbose BOOLEAN
```
**In Linux, it should be run in main folder**

## Running the SHP to PNG convertion:
The vector tiling consists in crop the full shapefile in small peaces according to the raster tiles extends processed before. The procedure saves a tiled shapefile if the full shapefile intersect a raster tile. If it does not intersect, the raster tile is then deleted (in order to save space). 

Here, the procedure demands 6 arguments, which are the procedure to be executed (`-procedure`), the shapefile tiles path (`-shapefile_folder`), the output to the final annotation images (`-output`), the width dimension of the tiles (`-tile_width`), the height dimension of the tiles (`-tile_height`), and a boolean verbose outcomes (`-verbose`). 
```
python main.py -procedure shp2png
               -shapefile_folder PATH_TO_VECTOR_TILES
               -output PATH_TO_SAVE_ANNOTATION_IMAGES
               -tile_width DIMESION_OF_TILES 
               -tile_height DIMESION_OF_TILES 
               -verbose BOOLEAN
```
**In Linux, it should be run in main folder**

## Bash for sequencial processing `run.sh`
Compiling the three procedures in one, the shellscript `run.sh`, in the main folder, can be then apply for multiple images and shapefiles, generating a consistent amount of samples. So, this file simply summary all processing until the final pair of training samples, which is the pair of image and its correspondent reference (annotation image). So, the only thing needed are the full images (preference in `.tiff` format), and its full correspondent shapefiles (ESRI Shapefile format - `.shp` extension).
```
RASTER_PATH=$1
RASTER_TILE_OUTPUT=$2
SQUARED_DIMENSION=$3
VECTOR_PATH=$4
VECTOR_TILE_OUTPUT=$5
OUTPUT_ANNOTATION=$6

for entry in "$RASTER_PATH"*
do
  if [ -f "$entry" ];then
    filename=$(basename $entry)

    python main.py -procedure tiling_raster -image "$entry" -output "$RASTER_TILE_OUTPUT" -tile_width "$SQUARED_DIMENSION" -tile_height "$SQUARED_DIMENSION" -verbose True &&
    python main.py -procedure tiling_vector -image_tiles "$RASTER_TILE_OUTPUT" -output "$VECTOR_TILE_OUTPUT" -shapefile_reference "$VECTOR_PATH" -verbose True
  fi
done

python main.py -procedure shp2png -image "$RASTER_TILE_OUTPUT" -shapefile_folder "$VECTOR_TILE_OUTPUT" -output "$OUTPUT_ANNOTATION" -tile_width "$SQUARED_DIMENSION" -tile_height "$SQUARED_DIMENSION" -verbose True
```

# TODO-list

This source-code is being released for specific applications, and for those who probably has similar needs. For this reason, we still have a lot to do in terms of unit tests, python conventions, optimization issues, refactoring, so on! So, Feel free to use and any recommendation or PRs will be totally welcome!

