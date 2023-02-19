#!/bin/bash  

read -r -p "Do you have 7z Installed? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo ""
else
    exit 0
fi


read -r -p "You are about to download a 10GB zip file. Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo ""
else
    exit 0
fi

read -r -p "It will then extract the data taking up an additional 80GB of space. Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo ""
else
    exit 0
fi

# # Download the data
wget -O data.zip https://www.dropbox.com/sh/oi00rlkkzy41qek/AADFcjULdYaot9_rtZvg7rYva?dl=0 

# Extract the data
unzip data.zip -x /

# Remove the data file
rm data.zip

# Extract the data
7z x highway.7z
rm highway.7z

7z x beamng.7z
rm beamng.7z

7z x waymo.7z
rm waymo.7z