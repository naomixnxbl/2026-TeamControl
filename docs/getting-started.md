# Getting Started

## What This Is
Short description: TeamControl is the TurtleRabbits RoboCup SSL team server. It receives SSL-Vision/Game Controller data, builds world state, and sends robot or grSim commands.

## Prerequisites
- Python 3.10+
- Git
- pip / venv
- Optional: grSim
- Optional: SSL-Vision
- Optional: SSL Game Controller
- Windows users: Git Bash recommended for `setup.sh`

## Install
### Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[testing]"

### Git Bash / Linux / macOS
./setup.sh

## Verify Installation
pytest
python -c "import TeamControl; print('TeamControl installed')"

## Configure Network
Point to:
- `src/TeamControl/utils/ipconfig.yaml`
- `docs/SSL-NetworkPorts.md`

### keys in `ipconfig.yaml`:
- `use_grSim_vision` - do we want to use grSim's Vision 
| `true` : grSim Vision settings | `false` : SSL-vision settings |
- `send_to_grSim` - do we want to send to grSim
| `true` : grSim Controls are active | `false` : grSim Controls are NOT active |
- `us_yellow` - default team color
| `true` : yellow | `false` : blue |
- `us_positive` - default team side (x-axis) is on positive
| `true` : positive | `false` : negative |
- grSim IP/port - connection sending address of grSim network `ip` and `port`
| `internal` : '127.0.0.1' | `external` : xxx.xxx.xxx.xxx | 
| `port` See grSim - Command listening port | `default` 20010 |
- vision multicast group/port
| `group` : '224.5.23.2' | `port` 10006 | * check grSim vision port 
- game controller multicast group/port
| `group` : '224.5.23.1' | `port` 10003 |

## Run the UI
python ui_main.py

## Run TeamControl Headless
python main.py

Available modes:
python main.py --mode goalie
python main.py --mode 1v1
python main.py --mode obstacle
python main.py --mode coop
python main.py --mode 6v6

## Run With grSim
To install grSim, please use one of the following. 
Linux - ubuntu: 
```shell
./scripts/installGRSIM.sh
```

Windows 11 :
1. Navgate to : https://github.com/rishieissocool/grSim-Windows/releases/tag/1.0.0 
2. Download `release.zip` and extract it to a destination that you desire.
3. run `grSim.exe`
4. Verify using [WireShark](https://www.wireshark.org/) to see if you receive internal traffic on the grSim Vision data.


Basic flow:
1. Start grSim.
2. Check grSim network ports -> update `grSim IP`as grSim device IP and `port` as grSim Command Listening port
3. Set `send_to_grSim: true`.
4. Set `use_grSim_vision: true` if using grSim vision.
5. Check and Update `grSim Vision Port` and change to `10006` for ease of access
5. Run `python ui_main.py`.

## Troubleshooting
Common checks:
- Is the virtual environment active?
- Did `pip install -e ".[testing]"` succeed?
- Is grSim running?
- Are multicast ports correct?
- Is the firewall blocking UDP?
- Are you using the correct team color?
- Is `ipconfig.yaml` configured for this machine?

## Next Steps
- Developers: `CONTRIBUTING.md`
- Network ports: `docs/SSL-NetworkPorts.md`
- Multiprocessing internals: `docs/Multiprocessing.md`
- Writing code: `https://www.turtlerabbit.org/docs/python-code-standards/`
- RoboCup SSL Official GitHub Repositories: `https://github.com/RoboCup-SSL`
- RoboCup SSL Rulebook Info: `https://ssl.robocup.org/rules/`
