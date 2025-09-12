import simpy
import networkx as nx
import pandas as pd

class Lane:
    def __init__(self, 
                 id: str,
                 blocks: int,
                 dedicated_lane_length: int,
                 lane_changing_zone_length: int):
        """Initialize a new lane.
        
        Args:
            id (str): Lane identifier (0-4)
            blocks (int): Total number of blocks in the lane
            dedicated_lane_length (int): Length of AV-only section in blocks
            lane_changing_zone_length (int): Length of lane changing section in blocks
        """
        self.id = id
        self.blue = True if int(id) > 2 else False
        self.green = True if int(id) <= 2 else False
        self.blocks = int(blocks)
        self.dedicated_lane_length = dedicated_lane_length
        self.lane_changing_zone_length = lane_changing_zone_length
        self.path = {}
        for i in range(self.blocks):
            self.path[i] = 0  # Initialize all blocks as empty

    def is_available(self, block: int) -> bool:
        """Check if a block has capacity for another vehicle.
        
        Args:
            block (int): The block number to check
            
        Returns:
            bool: True if block has less than 20 vehicles
        """
        return self.path[block] < 20
    
    def leave(self, block: int) -> None:
        """Remove a vehicle from a block.
        
        Args:
            block (int): The block number the vehicle is leaving
        """
        self.path[block] -= 1
    
    def arrive(self, block: int) -> None:
        """Add a vehicle to a block.
        
        Args:
            block (int): The block number the vehicle is entering
        """
        self.path[block] += 1


class Intersection:
    def __init__(self,
                 node_id: str,
                 neighbors: list,
                 lengths: list,
                 dedicated_lane_length: int,
                 lane_changing_zone_length: int,
                 each_block_length: int):
        """Initialize an intersection.
        
        Args:
            node_id (str): Unique identifier for this intersection
            neighbors (list): List of neighboring intersection IDs
            lengths (list): List of road segment lengths (one per neighbor)
            dedicated_lane_length (int): Length of AV-only sections
            lane_changing_zone_length (int): Length of lane changing sections
            each_block_length (int): Length of each road block
        """
        self.node_id = node_id
        self.lanes = {}
        self.lights = {}
        
        # Create lanes and lights for each approaching road segment
        for i in range(len(neighbors)):
            self.lanes[str(neighbors[i])] = []
            self.lights[str(neighbors[i])] = []
            for j in range(5):
                # Create lane with appropriate length in blocks
                self.lanes[str(neighbors[i])].append(Lane(
                    id=j, 
                    blocks=int(lengths[i])/each_block_length, 
                    dedicated_lane_length=int(dedicated_lane_length)/each_block_length,
                    lane_changing_zone_length=int(lane_changing_zone_length/each_block_length)
                ))
                self.lights[str(neighbors[i])].append("red")  # All lights start red
        # print(f"[INFO] Intersection {self.node_id} initialized with these lanes: {self.lanes}.")

    def update_lights(self, stats):
        """Update traffic light states based on comparing blue lanes (3,4) vs green lanes (0,1,2).
        
        Args:
            stats (pd.DataFrame): DataFrame containing vehicle statistics with columns:
                - time, vehicle_id, origin, destination, lane, block, arrival_time, stuck_time, active
        
        Logic:
        1. Compare total vehicles in blue lanes (3,4) vs green lanes (0,1,2) across all directions
        2. If blue lanes have more vehicles:
           - Set all blue lanes (3,4) to green in all directions
        3. If green lanes have more vehicles:
           - Find direction with most vehicles in green lanes
           - Set only that direction's green lanes (0,1,2) to green
        """
        if stats.empty:
            print("[INFO] No active vehicles to update lights.")
            return
            
        # Get active vehicles only
        active_vehicles = stats[stats['active'] == True]
        active_vehicles = active_vehicles.copy()
        active_vehicles['origin'] = active_vehicles['origin'].astype(str)
        active_vehicles['destination'] = active_vehicles['destination'].astype(str)
        active_vehicles['lane'] = active_vehicles['lane'].astype(str)
        
        # Initialize counters for blue vs green lanes
        total_blue_lanes = 0  # lanes 3,4
        total_green_lanes = 0  # lanes 0,1,2
        
        # Dictionary to store green lane counts per direction
        direction_green_counts = {}
        
        # Count vehicles in each direction
        # print(f"[DEBUG] Checking intersection {self.node_id}...")
        for neighbor in self.lanes.keys():
            neighbor = str(neighbor)
            node_id = str(self.node_id)
            # print(f"[DEBUG] Checking neighbor {neighbor}...")
            # Count blue lanes (3,4)
            blue_vehicles = active_vehicles[
                (active_vehicles['destination'] == node_id) & 
                (active_vehicles['origin'] == neighbor) &
                (active_vehicles['lane'].isin(['3', '4']))
            ]
            total_blue_lanes += self.find_in_queue_vehicles_from_lane(blue_vehicles, [neighbor, node_id])
            # print(f"[DEBUG] Blue vehicles: {len(blue_vehicles)}")

            # Count green lanes (0,1,2)
            green_vehicles = active_vehicles[
                (active_vehicles['destination'] == node_id) & 
                (active_vehicles['origin'] == neighbor) &
                (active_vehicles['lane'].isin(['0', '1', '2']))
            ]
            # print(f"[DEBUG] Green vehicles: {len(green_vehicles)}")
            green_v = self.find_in_queue_vehicles_from_lane(green_vehicles, [neighbor, node_id])
            total_green_lanes += green_v
            direction_green_counts[neighbor] = green_v

            # First set all lights to red
            for i in range(5):
                self.lights[neighbor][i] = "red"

        # print(f"[DEBUG] Total blue lanes: {total_blue_lanes}, Total green lanes: {total_green_lanes}, Direction green counts: {direction_green_counts}")
        # If blue lanes have more vehicles
        if total_blue_lanes > total_green_lanes:
            # Set all blue lanes to green in all directions
            for neighbor in self.lanes.keys():
                self.lights[neighbor][3] = "green"
                self.lights[neighbor][4] = "green"
        
        # If green lanes have more vehicles
        else:
            # Find direction with most green lane vehicles
            if direction_green_counts:  # Check if dictionary is not empty
                max_direction = max(direction_green_counts.items(), key=lambda x: x[1])[0]
                # Set only that direction's green lanes to green
                self.lights[max_direction][0] = "green"
                self.lights[max_direction][1] = "green"
                self.lights[max_direction][2] = "green"

        # print(f"[INFO] Traffic lights at node {self.node_id} updated: {self.lights}.")

    def find_in_queue_vehicles_from_lane(self, vehicles: pd.DataFrame, od: list):
        if not vehicles.empty:
            vehicles = vehicles.sort_values(by=["block"], inplace=False)
            blocks = vehicles["block"].values
            count = 0
            current = self.lanes[str(od[0])][0].blocks - 1
            while current in blocks:
                count += 1
                current -= 1
            return count
        else:
            return 0

class Vehicle:
    def __init__(self,
                 env: simpy.Environment,
                 id: str,
                 initial_path: str,
                 initial_lane: str,
                 type_: str,
                 stats: dict,
                 graph: nx.DiGraph,
                 track: int = 0):
        """Initialize a vehicle.
        
        Args:
            env (simpy.Environment): Simulation environment
            id (str): Unique vehicle identifier
            initial_path (list): [start_node, end_node]
            initial_lane (str): Initial lane number (0-4)
            type_ (str): Vehicle type ("HDV" or "AV")
            stats (pd.DataFrame): Simulation statistics
            graph (nx.DiGraph): Road network graph
            track (int, optional): Debug tracking level. Defaults to 0.
        """
        self.env = env
        self.id: str = id
        self.AV: bool = True if type_.upper() == "AV" else False
        self.HDV: bool = True if type_.upper() == "HDV" else False
        self.stats: pd.DataFrame = stats
        self.track = track
        self.initial_path = initial_path
        self.graph = graph
        self.current_path: list = self._shortest_path(str(self.initial_path[0]), str(self.initial_path[1])) # [start node, end node]
        self.current_intersection: Intersection = self.graph.graph.nodes[str(self.current_path[1])]["intersection"]
        self.current_lane: Lane = self.current_intersection.lanes[self.current_path[0]][int(initial_lane)]
        self.arrival_time = self.env.now
        self.stucked_time = 0
        self.current_pos: int = 0
        self.max_pos: int = self.current_lane.blocks - 1
        self._update_stats()
        if self.track:
            print(f"[TRACK{self.track}] Initializing...")
            print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")

    def _shortest_path(self, from_: str, to_: str) -> list:
        """Calculate the shortest path considering current traffic conditions.
        
        Uses a modified Dijkstra algorithm where edge weights are travel times
        calculated using the BPR formula with current traffic counts.
        
        Args:
            from_ (str): Starting node ID
            to_ (str): Target node ID
            
        Returns:
            list: [current_node, next_node] for the optimal path
            
        Note:
            This is a private method used internally for navigation.
        """
        # Get current traffic counts
        active_stats = self.stats[self.stats["active"] == True]
        link_counts = active_stats.groupby(["origin", "destination"]).size().to_dict()
        
        # Calculate travel time based on current traffic
        def travel_time(u, v, data):
            n_cars = link_counts.get((u, v), 0)
            x = data["param"]
            return float(data["expr"].subs(x, n_cars))
        
        # Find shortest path using current travel times
        path = nx.shortest_path(self.graph.graph, source=from_, target=to_, weight=travel_time)
        if len(path) >= 2:
            return [path[0], path[1]]  # Return next step in path
        
    def _update_stats(self) -> None:
        """Update vehicle statistics in the simulation DataFrame.
        
        Updates:
        - Marks previous position as inactive
        - Adds new row with current position and status
        
        Note:
            This is a private method called after any vehicle movement.
        """
        # Mark previous position as inactive
        self.stats.loc[self.stats["vehicle_id"] == self.id, "active"] = False
        light = self.current_intersection.lights[self.current_path[0]][self.current_lane.id]
        light = "none" if self.current_pos != self.max_pos else light
        # Add new row with current state
        self.stats.loc[len(self.stats)] = [
            self.env.now,         # Current time
            self.id,              # Vehicle ID
            self.current_path[0], # Current node
            self.current_path[1], # Next node
            self.current_lane.id, # Current lane
            self.current_pos,     # Block position
            self.arrival_time,    # Arrival time
            self.stucked_time,    # Stuck time
            True,                  # Active flag
            light,                 # Light status
            "AV" if self.AV else "HDV",  # Vehicle type
        ]

    def update_path(self, new_path: list, intersection: Intersection, lane_num: int) -> None:
        """Update vehicle's path when entering a new road segment.
        
        This method updates all necessary attributes when a vehicle moves to
        a new road segment (link), including:
        - Path information
        - Current intersection
        - Lane assignment
        - Timing information
        - Position limits
        
        Args:
            new_path (list): New [from_node, to_node] path segment
            intersection (Intersection): New intersection object
            lane_num (int): New lane number (0-4)
            
        Example:
            >>> # Vehicle moves to new road segment
            >>> vehicle.update_path(
            ...     new_path=["2", "3"],
            ...     intersection=new_intersection,
            ...     lane_num=2
            ... )
        """
        self.current_path = new_path
        self.current_intersection = intersection
        self._update_lane(lane_num)
        self.arrival_time = self.env.now
        self.stucked_time = 0
        self.max_pos = self.current_lane.blocks
        self._update_stats()
        if self.track:
            print(f"[TRACK{self.track}] Updating path...")
            print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")
    
    def _update_lane(self, new_lane: int) -> None:
        """Update vehicle's lane assignment.
        
        This private method handles the lane-level updates when a vehicle
        changes lanes, including:
        - Updating lane reference
        - Updating statistics
        
        Args:
            new_lane (int): New lane number (0-4)
            
        Note:
            This is a private method used internally by movement methods.
        """
        self.current_lane = self.current_intersection.lanes[self.current_path[0]][new_lane]
        self._update_stats()
        if self.track:
            print(f"[TRACK{self.track}] Updating lane...")
            print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")

    def _change_lane_to_left(self) -> bool:
        """Attempt to move to the left lane and forward one block.
        
        The vehicle will:
        1. Check if the target block in the left lane is available
        2. Leave current block
        3. Move to left lane
        4. Move forward one block
        
        Returns:
            bool: True if movement successful, False if blocked
            
        Note:
            This is a private method used by the vehicle's movement logic.
        """
        if self.track:
            print(f"[TRACK{self.track}] Try to go to 'left'...")
            
        # Check if target position is available
        target_lane = self.current_intersection.lanes[self.current_path[0]][int(self.current_lane.id)-1]
        if target_lane.is_available(block=self.current_pos+1):
            self.current_lane.leave(block=self.current_pos)
            self._update_lane(new_lane=int(self.current_lane.id)-1)
            self.current_lane.arrive(block=self.current_pos+1)
            self.current_pos += 1
            
            if self.track:
                print(f"[TRACK{self.track}] Successful!")
                print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")
            return True
            
        if self.track:
            print(f"[TRACK{self.track}] Failed! No change in the stats.")
        return False
    
    def _change_lane_to_right(self) -> bool:
        """Attempt to move to the right lane and forward one block.
        
        The vehicle will:
        1. Check if the target block in the right lane is available
        2. Leave current block
        3. Move to right lane
        4. Move forward one block
        
        Returns:
            bool: True if movement successful, False if blocked
            
        Note:
            This is a private method used by the vehicle's movement logic.
        """
        if self.track:
            print(f"[TRACK{self.track}] Try to go to 'right'...")
            
        # Check if target position is available
        target_lane = self.current_intersection.lanes[self.current_path[0]][int(self.current_lane.id)+1]
        if target_lane.is_available(block=self.current_pos+1):
            self.current_lane.leave(block=self.current_pos)
            self._update_lane(new_lane=int(self.current_lane.id)+1)
            self.current_lane.arrive(block=self.current_pos+1)
            self.current_pos += 1
            
            if self.track:
                print(f"[TRACK{self.track}] Successful!")
                print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")
            return True
            
        if self.track:
            print(f"[TRACK{self.track}] Failed! No change in the stats.")
        return False

    def _move_forward(self) -> bool:
        """Attempt to move forward one block in the current lane.
        
        The vehicle will:
        1. Check if the next block is available
        2. Leave current block
        3. Move to next block
        
        Returns:
            bool: True if movement successful, False if blocked
            
        Note:
            This is a private method used by the vehicle's movement logic.
        """
        if self.track:
            print(f"[TRACK{self.track}] Try to go to 'forward'...")
            
        # Check if next block is available
        if self.current_lane.is_available(block=self.current_pos+1):
            self.current_lane.leave(block=self.current_pos)
            self.current_lane.arrive(block=self.current_pos+1)
            self.current_pos += 1
            
            if self.track:
                print(f"[TRACK{self.track}] Successful!")
                print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")
            return True
            
        if self.track:
            print(f"[TRACK{self.track}] Failed! No change in the stats.")
        return False
    
    def _cant_move(self) -> None:
        """Record that vehicle is stuck in current position.
        
        Increments the stuck time counter by 5 units when a vehicle
        cannot move in the current time step.
        
        Note:
            This is a private method used internally when movement attempts fail.
        """
        if self.track:
            print(f"[TRACK{self.track}] Stucked!...")
        self.stucked_time += 5

    def _action_flr(self, p: list) -> bool:
        """Execute movement actions in priority order.
        
        Attempts movements in the specified order until one succeeds.
        Movement types:
        - 0: Forward
        - 1: Left + Forward
        - 2: Right + Forward
        
        Args:
            p (list): List of movement priorities
                Example: [2,1,0] means try Right, then Left, then Forward
                
        Returns:
            bool: True if any movement succeeded, False if all failed
            
        Example:
            >>> # Try right turn first, then forward
            >>> vehicle._action_flr([2, 0])
            >>> # Try left turn first, then right, then forward
            >>> vehicle._action_flr([1, 2, 0])
            
        Note:
            This is a private method used by the vehicle's movement strategies.
        """
        done = False
        actions = [self._move_forward, self._change_lane_to_left, self._change_lane_to_right]
        
        # Try each action in priority order until one succeeds
        for i in p:
            done = actions[i]() if not done else True
            
        # If no movement possible, update stuck time
        if not done:
            self._cant_move()
            
        return done

    def _simple_process(self) -> bool:
        """Handle vehicle movement when far from intersection.
        
        Movement strategy varies by lane and vehicle type:
        
        Lane 0 (rightmost):
        - All vehicles: Try forward, then right
        
        Lane 4 (leftmost):
        - All vehicles: Try forward, then left
        
        Lanes 1-3 (middle):
        - HDV: Try forward, then left
        - AV: Try forward, then right
        
        Returns:
            bool: True if movement succeeded, False if vehicle is stuck
            
        Note:
            This is a private method used when vehicle is in the simple movement zone.
        """
        if self.track:
            print(f"[TRACK{self.track}] Enter the simple process!")

        # Lane-specific movement strategies
        if int(self.current_lane.id) == 0:
            done = self._action_flr(p=[0, 2])  # Forward then right

        elif int(self.current_lane.id) == 4:
            done = self._action_flr(p=[0, 1])  # Forward then left

        # Middle lanes - different strategies for HDV and AV
        elif int(self.current_lane.id) in [1, 2, 3]:
            done = self._action_flr(p=[0, 1]) if self.HDV else False  # HDV: Forward then left
            done = self._action_flr(p=[0, 2]) if self.AV else False   # AV: Forward then right

        self._update_stats()
        return done

    def _lane_changing_process(self):
        """This is the vehicle action for the range of near to intersection."""
        if self.track:
            print(f"[TRACK{self.track}] Enter the lane changing process!")

        # if in 0: for HDV try to move forward, then go to right, for AV first try for right, then forward.
        if int(self.current_lane.id) == 0:
            done = self._action_flr(p=[0, 2]) if self.HDV else False
            done = self._action_flr(p=[2, 0]) if self.AV else False

        # if in 4: for HDV try to move left, then go to forward, for AV try to move forward then left.
        # warning: if we are in the 2 columns before dedicated area, HDV can only move left.
        elif int(self.current_lane.id) == 4:
            warning_temp = self.max_pos - self.current_lane.dedicated_lane_length - 1
            if not self.current_pos == warning_temp:
                done = self._action_flr(p=[1, 0]) if self.HDV else False
            else:
                done = self._action_flr(p=[1]) if self.HDV else False
            done = self._action_flr(p=[0, 1]) if self.AV else False
        
        # if in 3: for HDV try to move left, then go to forward, for AV try to move forward then right.
        # warning: if we are in the 2 columns before dedicated area, HDV can only move left.
        elif int(self.current_lane.id) == 3:
            warning_temp = self.max_pos - self.current_lane.dedicated_lane_length
            if not self.current_pos == warning_temp:
                done = self._action_flr(p=[1, 0]) if self.HDV else False
            else:
                done = self._action_flr(p=[1]) if self.HDV else False
            done = self._action_flr(p=[0, 2]) if self.AV else False

        # if in 2: for HDV try to go left then forward. for AV try to go right, then forward, and then right.
        elif int(self.current_lane.id) == 2:
            done = self._action_flr(p=[1, 0]) if self.HDV else False
            done = self._action_flr(p=[2, 0, 1]) if self.AV else False

        # if in 1: for HDV try to go left then forward then right. for AV try to go right, then forward, then left.
        elif int(self.current_lane.id) == 1:
            done = self._action_flr(p=[1, 0, 2]) if self.HDV else False
            done = self._action_flr(p=[2, 0, 1]) if self.AV else False

        self._update_stats()
        return done
        
    def _end_process(self):
        """This is the vehicle action for the range of close to intersection."""
        if self.track:
            print(f"[TRACK{self.track}] Enter the end process!")

         # if in 0: for HDV first try to move forward, then go to right. for AV try to move right, then forward.
        if int(self.current_lane.id) == 0:
            done = self._action_flr(p=[0, 2]) if self.HDV else False
            done = self._action_flr(p=[2, 0]) if self.AV else False

        # if in 4: only for AV try to move forward, then go to left. HDV not allowed.
        elif int(self.current_lane.id) == 4:
            assert self.HDV == False
            done = self._action_flr(p=[0, 1]) if self.AV else False
        
        # if in 3: only for AV try to move right, then go to forward.  HDV not allowed.
        elif int(self.current_lane.id) == 3:
            assert self.HDV == False
            done = self._action_flr(p=[2, 0]) if self.AV else False

        # if in 2: for HDV try to move left then forward. for AV try forward then left
        elif int(self.current_lane.id) == 2:
            done = self._action_flr(p=[1, 0]) if self.HDV else False
            done = self._action_flr(p=[2, 0]) if self.AV else False

        # if in 1: for HDV try to move left, then forward, then right. for AV try move right, then forward then left.
        elif int(self.current_lane.id) == 1:
            done = self._action_flr(p=[1, 0, 2]) if self.HDV else False
            done = self._action_flr(p=[2, 0, 1]) if self.AV else False
        
        self._update_stats()
        return done

    def _intersection_process(self) -> None:
        """Handle vehicle behavior at intersection.
        
        Checks traffic light state and either:
        1. Passes through intersection if light is green
        2. Waits at intersection if light is red
        
        Note:
            This is a private method used when vehicle reaches an intersection.
        """
        # Check if light is green for current lane
        if self.current_intersection.lights[self.current_path[0]][self.current_lane.id] == "green":
            self._pass_intersection()
        else:
            self._stay_intersection()
        self._update_stats()

    def _pass_intersection(self) -> None:
        """Move vehicle through intersection to next road segment.
        
        Actions performed:
        1. Leave current position
        2. Calculate next path segment
        3. Update intersection and lane
        4. Enter new road segment at position 0
        
        Note:
            This is a private method used when crossing intersection on green light.
        """
        self._update_stats()
        # Leave current position
        self.current_lane.leave(block=self.current_pos)
        
        # Calculate next path segment
        self.current_path = self._shortest_path(
            str(self.current_path[1]),  # Current intersection becomes start
            str(self.initial_path[1])   # Final destination remains the same
        )
        
        # Update intersection and lane
        self.current_intersection = self.graph.graph.nodes[str(self.current_path[1])]["intersection"]
        self.current_lane = self.current_intersection.lanes[self.current_path[0]][int(self.current_lane.id)]
        
        # Enter new road segment
        self.current_lane.arrive(block=0)

        # update arrival time
        self.arrival_time = self.env.now
        self.stucked_time = 0
        self.current_pos = 0
        self.max_pos = self.current_lane.blocks - 1
        if self.track:
            print(f"[TRACK{self.track}] Passing the intersection...")
            print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")

    def _stay_intersection(self) -> None:
        """Wait at intersection on red light.
        
        Note:
            This is a private method used when waiting at red light.
        """
        self._cant_move()

    def _exit_the_system(self) -> None:
        """Remove vehicle from simulation when reaching final destination.
        
        Actions performed:
        1. Leave current position
        2. Mark vehicle as inactive in statistics
        
        Note:
            This is a private method used when vehicle reaches its final destination.
        """
        # Leave current position
        self.current_lane.leave(block=self.current_pos)
        
        # Mark vehicle as inactive
        self.stats.loc[self.stats["vehicle_id"] == self.id, "active"] = False
        if self.track:
            print(f"[TRACK{self.track}] Exiting the system...")
            print(f"[TRACK{self.track}] [ID={self.id}] [path={self.current_path[0]} to {self.current_path[1]}] [lane={self.current_lane.id}] [block={self.current_pos}] [arrive_time={self.arrival_time}] [stuck_time={self.stucked_time}]")

    def process(self) -> None:
        """Execute the vehicle's movement behavior based on its position.
        
        The vehicle's behavior changes as it moves along a road segment:
        
        1. Simple Process (Far from intersection):
           - Basic lane changing and forward movement
           - No special restrictions
           
        2. Lane Changing Process (Approaching intersection):
           - Vehicles begin preparing for intersection
           - HDVs must exit AV lanes
           - AVs may optimize lane position
           
        3. End Process (Close to intersection):
           - Final lane positioning
           - HDVs not allowed in AV lanes
           - AVs align for efficient intersection crossing
           
        4. Intersection Process (At intersection):
           - Wait for green light
           - Cross intersection
           - Update path for next segment
           
        Example:
            >>> # Move vehicle according to its position
            >>> vehicle.process()  # Called every time step
        
        Raises:
            ValueError: If vehicle position exceeds maximum
        """
        # Determine zone based on position
        if self.current_pos < self.max_pos - self.current_lane.lane_changing_zone_length - self.current_lane.dedicated_lane_length:
            self._simple_process()  # Far from intersection
            
        elif self.current_pos < self.max_pos - self.current_lane.dedicated_lane_length:
            self._lane_changing_process()  # Approaching intersection
            
        elif self.current_pos < self.max_pos:
            self._end_process()  # Close to intersection
            
        elif self.current_pos == self.max_pos and self.current_path[1] == self.initial_path[1]:
            self._exit_the_system()

        elif self.current_pos == self.max_pos:
            self._intersection_process()  # At intersection
 
        else:
            raise ValueError("Vehicle position exceeds maximum allowed position")

