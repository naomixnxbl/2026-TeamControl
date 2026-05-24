#!/usr/bin/env bash
## this script installs Vision ssl and activates as the USB camera connection

echo "Installing SSL Vision please do not touch until you see the phrase - 'END'"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # updates system
        echo "*** Updating System ***"
        ## SELECT YOUR OWN UPGRADE
        sudo apt update
        # sudo apt upgrade -y
        # sudo apt-get dist-upgrade
        sudo apt-get full-upgrade -y

        # check home directory 
        cd 
        if [ -z "$HOME" ]; then
            echo "HOME variable is not set. Exiting."
            exit 1
        fi
        
        SSL_DIR="$HOME/ssl-software"
        VISION_DIR="$SSL_DIR/ssl-vision"

        echo "searching for Directory :$SSL_DIR"
        
        if [ ! -d "$SSL_DIR" ]; then
            echo "Directory : $SSL_DIR not found. Creating directory . . ."
            mkdir -p "$SSL_DIR" || { echo "Fail to create Directory $SSL_DIR"; exit 1; }
        else 
            echo "Found $SSL_DIR"
        fi

        cd "$SSL_DIR" || { echo "Failed to cd into $SSL_DIR"; exit 1;}
        
        echo "*** Installing ssl Vision --> $VISION_DIR"

        if [ ! -d "$VISION_DIR" ]; then
            echo "cloning SSL vision from GitHub"
            git clone https://github.com/RoboCup-SSL/ssl-vision.git || { echo "Git clone failed"; exit 1; }
        else
            echo "SSL Vision already Cloned."
        fi
        
        cd "$VISION_DIR"

        echo "** Installing Dependencies ** "
        ./InstallPackagesUbuntu.sh || { echo "Vision Dependencies installation failed"; exit 1; }

        ## installing v4l for usb camera
        sudo apt install -y v4l-utils
        
        echo "Building Vision base on using USB camera Input"
        ## Setting vision to work with USB camera
        cmake -B build -DUSE_V4L=true
        ## install
        make

        echo "SSL-Vision has now been installed @ $VISION_DIR"
        echo "To initiate the software, please input ./bin/vision -s within the terminal and directory"
        echo "If you want to know more about operating vision SSL, please visit their website."
        echo "https://github.com/RoboCup-SSL/ssl-vision.git"

    fi

echo "*** - E N D - ***"

