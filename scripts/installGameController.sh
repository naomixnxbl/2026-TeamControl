#!/usr/bin/env bash
## This installs the game controller and it only runs on ubuntu linux at the moment.
## when github do a clone, it will grab the repository name and use it as the folder name.
## Therefore, if that changes for any of the git clones, we will have to modify it correspondingly

## Function to compare versions 
version_lt() {
  [ "$1" = "$2" ] && return 1
  dpkg --compare-versions "$1" lt "$2"
}

echo "Installing gameController please do not touch until you see the phrase - 'END'"
    ## Required Software Versions
    NODEJS_VERSION="22.14.0"
    GO_VERSION="1.24.1"
    

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        cd 
        ## Verifying default home
        if [ -z "$HOME" ]; then
            echo "HOME variable is not set. Exiting."
            exit 1
        fi
        ## Sets directory 
        SSL_DIR="$HOME/ssl-software"
        GC_DIR="$SSL_DIR/ssl-game-controller"
        SB_DIR="$SSL_DIR/ssl-status-board"
        TIGERS_DIR="$SSL_DIR/TIGERS"
        ERFORCE_DIR="$SSL_DIR/ERFORCE"
        export NVM_DIR="$HOME/.nvm"
        echo "To make sure you install the right stuff, questions will be asked."
        echo "Only Type 'n' or 'N' if you don't want that software to be installed. "
        read -p "Would you Like to check System Update ? (Recommended yes) : " s 
        read -p "Would you Like to install SSL Status Board ? : " sb
        read -p "Would you Like to install Tigers' AutoReferee ? : " ta
        read -p "Would you Like to install ER-FORCE's AutoRef ? : " ea
        echo "User input received. -- Proceeding --"

        
        if [ ! -d "$SSL_DIR" ]; then
            echo "Directory : $SSL_DIR not found. Creating directory . . ."
            mkdir -p "$SSL_DIR" || { echo "Fail to create Directory $SSL_DIR"; exit 1; }
        fi

        cd "$SSL_DIR" || { echo "Failed to cd into $SSL_DIR"; exit 1;}

        case "$s" in 
            [nN]* ) echo "SKIPPING SYSTEM UPDATE & UPGRADE" ;;
            * ) 
                echo "*** Updating System ***"
                ## SELECT YOUR OWN UPGRADE
                sudo apt update
                # sudo apt upgrade -y 
                sudo apt-get dist-upgrade -y 
                # sudo apt-get full-upgrade -y 
                echo " SYSTEM UPDATE COMPLETED " ;;
        esac 

        echo "*** Installing Software Dependency ***"
        echo "This will install : curl, unzip, NVM, NodeJS, GO"

        sudo apt install curl unzip -y # u need this anyway
        
            
        ## The game controller requires a node JS version above 20, so we will be using the following script.
        # Ensure nvm is installed
        echo "Checking for nvm . . ."
        if [ -z "$NVM_DIR" ] || [ ! -s "$NVM_DIR/nvm.sh" ]; then
            echo "nvm is not installed. Installing now..."
            curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash 
            export NVM_DIR="$HOME/.nvm"
            source "$NVM_DIR/nvm.sh"
        else
            echo "nvm found"
            source "$HOME/.nvm/nvm.sh"
        fi
        source ~/.bashrc #restart terminal for nvm, npm
        echo "self-reloading Terminal"
        ## checks the version of nvm and npm to see if they are installed
        echo "node : nvm version : $(nvm --version) and npm version : $(npm --version) is installed and active"



        ## Locate if there's any existing NodeJs and what is the version
        if command -v node &>/dev/null; then
            CURRENT_NODEJS=$(node -v | sed 's/v//')
            echo "Installed Node.js version: $CURRENT_NODEJS"
        else
            CURRENT_NODEJS="none"
            echo "Node.js is not installed."
        fi
        
        # If there's no NODEJs installed or the version is lower than the required Version get the latest version
        if [ "$CURRENT_NODEJS" = "none" ] || version_lt "$CURRENT_NODEJS" "$NODEJS_VERSION"; then
            echo "Installing/updating Node.js to latest LTS version..."
            nvm install --lts
            nvm use --lts
            # Verify installation
            node -v
        else
            echo "Node.js is up to date. No update needed."
        fi

        echo "** Installing GO **"

        # Locate and get Current (Installed) Go version
        if command -v go &>/dev/null; then
            CURRENT_GO=$(go version | awk '{print $3}' | sed 's/go//')
            echo "Installed Go version: $CURRENT_GO"
        else
            CURRENT_GO="none"
            echo "Go is not installed."
        fi

        # If there is no GO or the version is Lower that required Go Version
        if [ "$CURRENT_GO" = "none" ] || version_lt "$CURRENT_GO" "$GO_VERSION"; then
            ## Gets the NEW Latest GO version
            echo "Updating Go to the latest GO version..."

            # Remove old versions if necessary
            sudo rm -rf /usr/local/go

            # Download and install the latest GO version
            wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
            sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
            rm go${GO_VERSION}.linux-amd64.tar.gz

            # Add Go to PATH
            export PATH=$PATH:/usr/local/go/bin
            echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
            echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.zshrc

            echo "Go has been updated to version $(go version)"
        else
            echo "Go is up to date. No update needed."
        fi
        
        ##########################################################################
        echo "*** Installing ssl-game-controller ***"
        
        echo "*** Cloning Git Repository -> $GC_DIR ***"
        ## IF the game controller folder does not exist
        if [ ! -d "$GC_DIR" ]; then 
            git clone https://github.com/RoboCup-SSL/ssl-game-controller.git || { echo "Git clone failed"; exit 1; }
        else
            echo "Game Controller already Exist"
        fi
        cd $GC_DIR

        ## Activate make install
	    make install || { echo "Fail to make install Game Controller"; exit 1; }
        ## Return to ssl directory
        cd $SSL_DIR

        ##################################################################
        case $sb in 
            [Nn]* ) echo "SKIPPING SSL-Status-Board" ;;
            * )
                echo "*** Installing SSL-Status-Board ***"
                echo "*** Cloning Git Repository -> $SB_DIR ***"
                if [ ! -d "$SB_DIR" ]; then ## IF the folder ssl-status-board does not exist
                    git clone https://github.com/RoboCup-SSL/ssl-status-board.git || { echo "Git clone failed"; exit 1; }
                else
                    echo "Status Board already Cloned from GitHub" 
                fi
                cd $SB_DIR
                make install || { echo "Fail to make install Status Board"; exit 1; }
                ## RETURN
                cd $SSL_DIR
                ;;
        esac



        #######################################################################
        case "$ta" in
            [nN]* ) echo "SKIPPING TIGERS' AUTOREF";;
            * ) 
            
                echo "*** Installing Java SDK for TIGER's AutoRef ***"
                sudo apt install openjdk-21-jdk -y

                echo "*** Installing TIGER's AutoReferee -> $TIGERS_DIR/AutoReferee ***"
                if [ ! -d "$TIGERS_DIR/AutoReferee" ]; then
                    mkdir -p "$TIGERS_DIR"
                    cd "$TIGERS_DIR" || { echo "Failed to cd into $TIGERS_PATH"; exit 1; }
                    echo "*** Cloning Git Repository -> Tiger's AutoReferee ***"
                    git clone https://github.com/TIGERs-Mannheim/AutoReferee.git || { echo "Git clone failed"; exit 1; }
                else
                    echo "TIGER's AutoReferee Already Exists"
                fi

                ## Navigate into AutoReferee directory
                cd "$TIGERS_DIR/AutoReferee" || { echo "Failed to cd into AutoReferee"; exit 1; }

                ## Try to build AutoReferee
                ./build.sh || { echo "ERROR during execution on Tiger's build.sh"; exit 1; }

                ## RETURN
                cd "$SSL_DIR" 
                ;;
        esac

        ################################################################################
        case "$ea" in
            [nN]* ) echo "SKIPPING ER-FORCE'S AUTOREF" ;;
            * )
                echo "*** Installing ERFORCE - AUTOREF --> $ERFORCE_DIR/autoref ***"
                if [ ! -d "$ERFORCE_DIR/autoref" ]; then ## IF the folder ERFORCE does not exist
                    mkdir -p "$ERFORCE_DIR"|| { echo "Failed to create $ERFORCE_DIR"; exit 1; }
                    cd "$ERFORCE_DIR" || { echo "Failed to cd into $ERFORCE_DIR"; exit 1; }
                    git clone https://github.com/robotics-erlangen/autoref.git || { echo "Git clone failed"; exit 1; }
                else
                    echo "$ERFORCE_DIR exists."
                fi

                ## going into ERFORCE's AUTOREF folder
                cd "$ERFORCE_DIR/autoref" || { echo "Failed to cd into $ERFORCE_DIR/autoref"; exit 1; }

                echo "Initialising ERFORCE GitHub Submodule"
                git submodule update --init || { echo "Submodule initialization failed"; exit 1; }
                ## Setting auto pull submodule
                git config submodule.recurse true
                ## Pulling Submodule
                git pull || { echo "Failed to pull updates"; exit 1; }

                echo "Installing Dependencies"
                ./install_ubuntu_deps.sh || { echo "Dependency installation failed"; exit 1; }

                echo "Building ERFORCE AutoRef package"
                ./build.sh || { echo "Build failed"; exit 1; }

                echo "-- Installation Completed , Returning to : $SSL_DIR"
                ## RETURN
                cd "$SSL_DIR"
                ;;
        esac

    fi

echo "*** - E N D - ***"


## Depeciated code storage (temp)
        # installing default go
        # sudo apt install golang-go -y
        # go version
        # echo "- Installing NodeJS version 20"
        # curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh 
        # echo "Reloading Terminal"
        # source ~/.bashrc
        # nvm install v22.14.0
        # nvm list
        # echo "NodeJS has now been installed with Version:"
        # node -v