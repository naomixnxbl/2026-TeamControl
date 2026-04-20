#! /usr/bin/bash 

## this installs grsim
## Please use chmod +x installGRSIM.sh before trying to run this script
## only works in Linux

echo "Installing grSim please do not touch until you see the phrase - 'END'";

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        cd 
        ## Verifying default home
        if [ -z "$HOME" ]; then
            echo "HOME variable is not set. Exiting."
            exit 1
        fi

        SSL_DIR="$HOME/ssl-software"
        if [ ! -d "$SSL_DIR" ]; then
            echo "Directory : $SSL_DIR not found. Creating directory . . ."
            mkdir -p "$SSL_DIR" || { echo "Fail to create Directory $SSL_DIR"; exit 1; }
        fi

        cd "$SSL_DIR" || { echo "Failed to cd into $SSL_DIR"; exit 1;}
       

        echo "*** Updating System ***"
        sudo apt update
        sudo apt-get dist-upgrade -y

        echo "*** Installing GRSIM Software Dependency ***"
        sudo apt install -y git build-essential cmake pkg-config qtbase5-dev \
                   libqt5opengl5-dev libgl1-mesa-dev libglu1-mesa-dev \
                   libprotobuf-dev protobuf-compiler libode-dev libboost-dev || { echo "Dependency installation failed"; exit 1; }
 
        ## If grSim is not cloned 
        if [ ! -d "$SSL_DIR/grSim" ]; then 
            echo "*** Cloning git GRSIM repository from https://github.com/RoboCup-SSL/grSim.git ***"
            git clone https://github.com/RoboCup-SSL/grSim.git
        else 
            echo "GRSIM Already Cloned"
        fi
        
        cd grSim || { echo "cannot goto grSim file, check git clone. Abandoning . . . "; exit 1; }
        mkdir -p build && cd build || { echo "Failed to create/navigate to build directory"; exit 1; }

        echo "*** Building and Installing GrSim ***"

        cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. || { echo "CMake configuration failed"; exit 1; }
        make || { echo "Make command failed"; exit 1; }
        
        sudo make install || { echo "Installation failed"; exit 1; }



        echo "*** GRSIM Installation and Setup Complete ***"

        echo "You can now go to $SSL_DIR/grSim to access the grSim Installation"

        echo "To trigger GrSim, do : cd $SSL_DIR/grSim && ./bin/grSim"



    else
        echo "Sorry This installation only supports in Linux(ubuntu)"

    fi

echo "*** END *** "
