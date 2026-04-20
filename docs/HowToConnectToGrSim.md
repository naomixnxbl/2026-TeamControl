# How to connect to GRSIM ? 

This team control is not only a server to the real time robots but also contains the code to directly connect with GRSIM.

You can skip this document if you already know how to do so. 

---

To establish your connection to grSim, you will need to an Application : GRSIM to be ACTIVE on a device on the SAME network.

Then you will have to ensure that the ports that are using are the same between GRSIM and THIS COMPUTER (the one you are running this team control)

Afterwards, you can run the GRSIM.py in `TeamControl > SSL > grSim.py`. This is a tutorial that provides the basics.

You can also replace the `make_robot_move()` function to other functions and test.

It is recommended to not commit any changes unless it is critical to this file.

GrSim's referee is `refbox`, it is not the `SSL game controller`, however you can still run it separately.

---

If you have further questions, please consult Emma.

Thank you.