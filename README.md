# Team Control (2026)

This is the Repository for Our server-side operation

To see the difference between this and last year's version, please see [here](/docs/DifferenceFromLastVersion.md). 
1
---

## Installation 

To install this module, do

```
git clone https://github.com/WSU-TurtleRabbit/2025-teamcontrol.git
```
Then we will have to create a virtual environment. 
In Linux : 
```bash
python3 -m venv .venv
```
In Windows : 
```powershell
python -m venv .venv
```
*if this did not work, it is recommended to get the version of python (>=3.13) from the [python.org](https://www.python.org/downloads/). Then add python to path
in particular : Appdata/Python3.13/ (-> python) and Appdata/Python3.13/Scripts/ (-> pip) 
Then restart vscode or the app with terminal opened.


Then we will have to install the pip packages and our development project. 
To do so, you will first have to activate the environment.
In Linux : 
```bash 
source .venv/bin/activate
```
In Windows : 
```powershell
.\.venv\Scripts\activate.bat
```

Then install the modules and this project as *Editable* project (that uses `pyproject.toml`)
```bash 
pip install -e . 
```

Now you should be able to run and start coding without any problem ! 

If you want to know what was installed (and what version), you can do :
```bash
pip list
```
or you can get a file called `requirement.txt` as a backup for the current modules and stuff.
To do so, in Linux : 
```bash
pip freeze > requirement.txt
```
afterwards, you can do : 
```bash
pip install -r requirement.txt
```
This is then the modules install using `requirement.txt` (which is another way to do it). To learn more, see [how to setup a python project](https://github.com/WSU-TurtleRabbit/how-to/blob/b2daf710f8d522aca7eadc72b78dd3002f60de95/Code/PythonProjectSetup.md)

If got into any error, please copy or screenshot text and post it in Mattermost Chat, and await for reply. 


To deactivate a virtual environment use :
In Linux : 
```bash
deactivate 
```
In Windows : 
```powershell
.\.venv\Scripts\deactivate.bat
```
