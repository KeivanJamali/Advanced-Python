import simpy


class MovementServerIntersection:
    def __init__(self, 
                 env: simpy.Environment,
                 lane: str,
                 id_: str,
                 type_: str):
        self.env = env
        self.lane = lane
        self.id_ = id_
        self.blue = True if type_.upper() == "AV" else False
        self.green = True if type_.upper() == "HDV" else False
        self.server = simpy.Resource(env, capacity=1)


class Intersection:
    def __init__(self,
                 env: simpy.Environment,
                 node_id: str,
                 neighbors: list):
        self.node_id = node_id
        self.env = env
        self.servers = {}
        for neighbor in neighbors:
            self.servers[neighbor] = []
            for i in range(5):
                if i > 2:
                    self.servers[-1].append(MovementServerIntersection(env, lane=i, id_=f"{node_id}->{neighbor}", type_="AV"))
                else:
                    self.servers[-1].append(MovementServerIntersection(env, lane=i, id_=f"{node_id}->{neighbor}", type_="HDV"))
        

class Vehicle:
    def __init__(self, 
                 id_: str,
                 initial_path: str,
                 initial_lane: str,
                 intersection: Intersection,
                 env: simpy.Environment,
                 type_: str,
                 arrival_time: float,
                 service_time: float,
                 stats: dict,
                 _callbacks: dict):
        """Here we have the Customer object that will process.

        Args:
            env (simpy.Environment): We pass the environment we are working in.
            id (int): The id of this customer which is a number.
            server (simpy.Resource): We pass the resource or server.
            arrival_time (float): We pass the calculated arrival time here.
            service_time (float): We pass the calculated service time here.
            stats (dict): All the parameters we want to monitor.
            _callback (dict, private variable): Do not change this variable, it will call Engine whenever is needed. Warning!!!
        """
        self.env = env
        self.id_ = id_
        self.AV = True if type_.upper() == "AV" else False
        self.HDV = True if type_.upper() == "HDV" else False
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.stats = stats
        self._callbacks = _callbacks
        self.action = None

        self.current_path = initial_path
        self.current_lane = initial_lane
        self.current_intersection = intersection

    def active(self):
        """Activate vehicles. Vehicles starts to do it's job."""
        self.action = self.env.process(self.process())

    def update_path(self, new_path: str, intersection: Intersection):
        self.current_location = new_path
        self.current_intersection = intersection
    
    def update_lane(self, new_lane: int):
        self.current_lane = new_lane

    def process(self):
        """The behaviour of a vehicle who shoud go to the server."""
        o, i, d = self.current_path.split("->")
        server = self.current_intersection.servers[d][int(self.current_lane)]
        with server.request() as req:
            yield req
            yield self.env.timeout(self.service_time)
        