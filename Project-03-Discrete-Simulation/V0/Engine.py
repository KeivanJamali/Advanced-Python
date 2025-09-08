import simpy
import Players
import pandas as pd
import DataLoader
from tqdm.auto import tqdm
from pathlib import Path

class Clock:
    """A simulation clock that manages the traffic simulation environment.
    
    This class serves as the main controller for the traffic simulation. It manages:
    - The simulation environment (simpy)
    - Vehicle generation and tracking
    - Traffic statistics
    - Network graph representation
    - Traffic light control
    
    Attributes:
        env (simpy.Environment): The simulation environment
        stats (pd.DataFrame): Statistics tracking frame with columns:
            - time: Current simulation time
            - vehicle_id: Unique identifier for each vehicle
            - origin: Starting intersection
            - destination: Target intersection
            - lane: Current lane number (0-4)
            - block: Current block position in lane
            - arrival_time: Time vehicle arrived at current position
            - stuck_time: Time vehicle has been stuck
            - active: Whether vehicle is still in simulation
        graph (DataLoader.Graph_Generator): Network representation
        vehicles (dict): Dictionary of active vehicles
        demand (pd.DataFrame): Vehicle demand schedule
        
    Example:
        >>> # Initialize simulation with a 500m dedicated lane and change zone
        >>> sim = Clock(
        ...     network_file='network.csv',
        ...     dedicated_lane_length=500,
        ...     lane_changing_zone_length=500
        ... )
        >>> # Load vehicle demand and run for 60 time steps
        >>> sim.generate_vehicles('demand.csv')
        >>> sim.run()
    """
    
    def __init__(self, network_files: list, output_directory: str, dedicated_lane_length: int, 
                 lane_changing_zone_length: int, each_block_length: int = 100):
        """Initialize the traffic simulation environment.
        
        Args:
            network_files (list): Path to CSV file containing network structure and pos
            dedicated_lane_length (int): Length of dedicated AV lanes in meters
            lane_changing_zone_length (int): Length of lane changing zones in meters
            each_block_length (int, optional): Length of each road block. Defaults to 100 meters
        """
        self.output_directory = Path(output_directory)
        self.env = simpy.Environment()
        self.stats = pd.DataFrame(columns=["time", "vehicle_id", "origin", 
                                         "destination", "lane", "block", 
                                         "arrival_time", "stuck_time", "active", "light", "type"])
        self.graph = DataLoader.Graph_Generator(
            network_files=network_files,
            dedicated_lane_length=dedicated_lane_length,
            lane_changing_zone_length=lane_changing_zone_length,
            each_block_length=each_block_length
        )
        self.vehicles = {}
        self.demand = {}
        print(f"[INFO] Initializing finished. Graph had been generated.")

    def generate_vehicles(self, file_path: str) -> None:
        """Load vehicle demand data from a CSV file.
        
        The CSV file should contain columns:
        - ID: Unique vehicle identifier
        - departure: Time step when vehicle enters simulation
        - Origin: Starting intersection ID
        - Destination: Target intersection ID
        - lane: Initial lane number (0-4)
        - type: Vehicle type (1 for HDV, 2 for AV)
        
        Args:
            file_path (str): Path to the CSV file containing vehicle demand data
            
        Example:
            >>> sim.generate_vehicles('demand.csv')
        """
        self.demand = pd.read_csv(file_path)
        self.demand.sort_values(by="departure", inplace=True)
        print(f"[INFO] Vehicles read in the data and generated correctly.")

    def _sort_vehicles(self) -> list:
        """Sort active vehicles by their arrival time plus stuck time.
        
        This private method is used to determine vehicle processing order,
        prioritizing vehicles that have been waiting longer.
        
        Returns:
            list: Sorted list of vehicle IDs
            
        Note:
            This is a private method used internally by run_gen()
        """
        active_vehicles = self.stats[self.stats["active"]]
        sorted_active_vehicles = active_vehicles.sort_values(
            by=["arrival_time"], 
            key=lambda x: active_vehicles["arrival_time"] + active_vehicles["stuck_time"]
        )
        vehicles = sorted_active_vehicles["vehicle_id"].to_list()
        return vehicles

    def _run_gen(self) -> simpy.events.Generator:
        """Generator function that implements the main simulation logic.
        
        This private method:
        1. Generates new vehicles according to demand schedule
        2. Updates traffic lights every 5 time steps
        3. Processes vehicle movements
        4. Advances simulation time
        
        Note:
            This is a private method used internally by run()
        """
        while True:
            time = self.env.now
            if not self.demand.empty:
                queue = self.demand.iloc[0]

                # Generate new vehicles at their departure time
                if abs(time - int(queue["departure"])) < 0.01:
                    self.vehicles[str(int(queue["ID"]))] = Players.Vehicle(
                        env=self.env,
                        id=str(int(queue["ID"])),
                        initial_path=[str(int(queue["Origin"])), str(int(queue["Destination"]))],
                        initial_lane=str(int(queue["lane"])-1),
                        type_="HDV" if int(queue["type"])==1 else "AV",
                        graph=self.graph,
                        stats=self.stats, 
                        track=0
                    )
                    self.demand = self.demand.drop(self.demand.index[0]).reset_index(drop=True)
                    print(f"[INFO] We are adding a new car into the system at time {time}.")
            
            # Every 5 time steps: update lights and process vehicles
            if time % 5 == 0:
                # Update lights for all intersections
                for node in self.graph.graph.nodes:
                    intersection = self.graph.graph.nodes[node]["intersection"]
                    intersection.update_lights(self.stats)
                    # print(f"[INFO] Traffic lights at intersection {node} have been updated at time {time}.")
                
                # Sort vehicles by waiting time
                vehicles_id = self._sort_vehicles() if self.vehicles else []
                print(f"[INFO] There are {len(vehicles_id)} sorted vehicles in the system.")
                # Process all vehicles
                for id in vehicles_id:
                    self.vehicles[id].process()
            
            yield self.env.timeout(1)
            
    def run(self, until: int = 60) -> None:
        """Run the simulation for a specified number of time steps.
        
        Displays a progress bar showing simulation time progression.
        
        Args:
            until (int, optional): Number of time steps to simulate. Defaults to 60.
            
        Example:
            >>> # Run simulation for 100 time steps
            >>> sim.run(until=100)
        """
        try:
            # Create progress tracking process
            def progress_monitor():
                with tqdm(total=until, desc="Simulation Progress", 
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} [Time elapsed: {elapsed}]") as pbar:
                    last_time = 0
                    while True:
                        current_time = self.env.now
                        if current_time > last_time:
                            pbar.update(current_time - last_time)
                            last_time = current_time
                        if current_time >= until:
                            break
                        yield self.env.timeout(1)
            
            # Start both simulation and progress monitoring
            self.env.process(self._run_gen())
            self.env.process(progress_monitor())
            self.env.run(until=until)
            
        except ImportError:
            print("Note: Install tqdm package for progress bar visualization")
            # Run without progress bar if tqdm not available
            self.env.process(self._run_gen())
            self.env.run(until=until)

        self.save_log()

    def save_log(self, file_path: str = "simulation_log.csv") -> None:
        """Save the simulation log to a CSV file.

        Args:
            file_path (str, optional): Path to the output CSV file. Defaults to "simulation_log.csv".
        """
        self.stats.to_csv(self.output_directory / file_path, index=False)
        print(f"[INFO] Simulation log saved to {file_path}")

    def draw_network(self) -> None:
        """Visualize the road network using the graph's draw method.
        
        This method calls the draw function of the Graph_Generator instance
        to display the road network with intersections and road segments.
        
        Example:
            >>> sim.draw_network()
            # Displays the network visualization
        """
        self.graph.draw()