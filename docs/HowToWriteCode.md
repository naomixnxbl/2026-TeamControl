You might be new and want to work on some code and don't know where to start.

To begin, here are a few concepts that you need to know. 

1. The Robots are all connected to this central Team Control software.

2. Please be aware that at this point in time, there will be a slight delay in the system due to operation, optimization on traffic, or Calculation. An example might be that robot communication requires a delay of 0.5s per message being received, please note and adjust accordingly.

3. We might be using Multiprocessing to Optimise the Computing power so that multiple operation can happen at the same time at ease.
    However, there might also be issues where if one process fails, it goes unaware of. 

4. We work with Camera Vision software - Vision SSL and simulated GRSIM. These 2 software are similar with slight differences. In particular Vision SSL uses 1 camera, and GRSIM uses 4 cameras. GRSIM Robots also have better velocity control translation, hence, software vs our robots might behave differently.

5. Due to the standard color and pattern provided by the League, we will be following those closely in this software, which means they are being identified and referenced accordingly. examples : setting our Team Color : if our team is yellow : is_yellow => True, if our Team is Blue => is_yellow = False

6. There are existing functions to minimize the confusion of team color, however, if you have referenced it too much and everywhere it may break, please check accordingly. 

7. To Connect to GRSIM or VISION or GAME CONTROLLER, please access the `NETWORK >  SSL_NETWORKING.PY` They have been coded specifically for the job.
    To see examples, please see run_1_robot.py or run_single_process.py . 

8. We tried to make our functions as condensed as possible, if you are not sure, feel free to ask, if you want to test something new, start a new file, so you get a big playground, then optimise it and put it into a category. (You can setup a new one if needed)

9. It is recommended that you utilize comments and doc-strings as they are handy whether you revisit it after a long time or getting things clear while you're coding. Makes it easier for one another. Also put down your name next to the functions that you worked on, so when the code is visited or someone wants to learn more, they can approach you.

10. It is recommended that you are working on a separate branch on a single issue, or things can get messy, and please do not force push origin at any time without permission and it is highly discouraged to do that, please reconsider other alternatives.