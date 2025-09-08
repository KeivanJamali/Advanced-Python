from Engine import Clock

demand_path = r"/mnt/Data1/Python_Projects/Machine-Learning-Projects/Project 13 - Saghar Ahmadi/data/demand.csv"
network_file = r"/mnt/Data1/Python_Projects/Machine-Learning-Projects/Project 13 - Saghar Ahmadi/data/Network.csv"
pos_file = r"/mnt/Data1/Python_Projects/Machine-Learning-Projects/Project 13 - Saghar Ahmadi/data/SiouxFalls_node_xy.tntp"
output_directory = r"/mnt/Data1/Python_Projects/Machine-Learning-Projects/Project 13 - Saghar Ahmadi/Results"
network_files = [network_file, pos_file]
world = Clock(network_files=network_files, dedicated_lane_length=500, lane_changing_zone_length=500, output_directory=output_directory)
# world.draw_network()
world.generate_vehicles(demand_path)
world.run(until=10000)