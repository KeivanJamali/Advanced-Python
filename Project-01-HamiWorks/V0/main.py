from data_analysis import Pre_Analysis, Post_Analysis
from data_visulazation import Visulaze
from pathlib import Path

output_folder = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/output")
input_folder = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/input")
plot_folder = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/plots")


obj = Pre_Analysis(input_path=input_folder, output_path=output_folder)
obj.fit_to_get_data()
obj.fit_to_analysis()


obj = Post_Analysis(output_path=output_folder)
obj.fit()

obj = Visulaze(output_path=output_folder, plot_path=plot_folder)
obj.analyze_conversation(hist=True, bar=True, scatter=True, pair=True, heatmap=True)
obj.analyze_students()