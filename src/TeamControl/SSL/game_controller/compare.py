from TeamControl.SSL.game_controller.Message import *
from TeamControl.network.ssl_sockets import GameControl
from multiprocessing import Queue

class GameControllerManager:
    def __init__(self,output_q:Queue,):
        self.recv = GameControl()
        self.output_q = output_q
        self.ref_msg:RefereeMessage = None
        
    def run(self):
        while True:
            new_ref = self.listen()
            self.update(new_ref)
    
    def listen(self):
        data,_ = self.recv.listen()
        ref_msg = self.convert(data)
        return ref_msg
     
    def update(self,ref_msg:RefereeMessage):
        current = self.ref_msg
        if current is None:
            print("initialise")
            self.ref_msg = ref_msg
            self.output_q.put(ref_msg)
            return
        elif current is not None:
            if current.command != ref_msg.command:
                print(f"new Command: {ref_msg.command}")
                self.output_q.put(ref_msg.command)
            if current.stage != ref_msg.stage:
                print(f"new Stage: {ref_msg.stage}")
                self.output_q.put(ref_msg.stage)
            if current.blue_team_on_positive_half != ref_msg.blue_team_on_positive_half:
                print("sides switched")
                self.output_q.put(ref_msg)
            
            # when yellow is no longer yellow, when blue is no longer blue
            elif current.yellow.name != ref_msg.yellow.name or current.blue.name != ref_msg.blue.name:
                print(f"Team {ref_msg.yellow.name} is now yellow, Team {ref_msg.blue.name} is now BLUE, Blue Positive={ref_msg.blue_team_on_positive_half}")
                packet = [ref_msg.yellow, ref_msg.blue]
                self.output_q.put(packet)
            
            else: #when no team color / side change    
                if current.yellow.max_allowed_bots != ref_msg.yellow.max_allowed_bots:
                    print(f"YELLOW ROBOTS ALLOWED : {ref_msg.yellow.max_allowed_bots} ")
                    self.output_q.put(ref_msg.yellow) 

                if current.blue.max_allowed_bots != ref_msg.blue.max_allowed_bots:
                    print(f"BLUE ROBOTS ALLOWED : {ref_msg.blue.max_allowed_bots} ")
                    self.output_q.put(ref_msg.blue)
            self.ref_msg = ref_msg
        else:
            return
        
    def convert(self,ref_msg):
        try:
            ref_msg = RefereeMessage.from_proto(ref_msg)
            return ref_msg
        except Exception as e:
            raise Exception(f"exception encountered : {ExceptionGroup}{e}")


def game_controller_worker(output_q):
    manager = GameControllerManager(output_q)
    manager.run()

if __name__ == "__main__":
    def read(input_q):
        while True:
            if not input_q.empty():
                item = input_q.get_nowait()
                print(item)
    from multiprocessing import Process
    output_q = Queue()
    
    manager = Process(target=game_controller_worker,args=(output_q,))
    reader = Process(target=read,args=(output_q,))
    
    manager.start()
    reader.start()
    manager.join()
    reader.join()