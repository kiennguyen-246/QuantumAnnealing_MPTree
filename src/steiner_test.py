import os
import unittest
import networkx as nx

from src.steinerTree import ilp


class MyTestCase(unittest.TestCase):
    def read_input(self, file_path):
        g = nx.DiGraph()
        terminals = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line.split()) == 0:
                    continue
                type = line.split()[0]
                if type == "E":
                    u, v, w = line.split()[1:]
                    g.add_edge(int(u), int(v), weight=float(w))
                    g.add_edge(int(v), int(u), weight=float(w))
                elif type == "T":
                    terminals.append(int(line.split()[1]))
        return g, terminals

    def read_answer(self, file_path):
        ans_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                line = line.strip()
                ans_dict[line.split(",")[0].strip()] = int(line.split(",")[1].strip())
        return ans_dict

    def test_ilp(self):
        for track_id in [1, 2, 3]:
            directory = f"../steiner_dataset/Track{track_id}"
            ans_dict = self.read_answer(f"../steiner_dataset/track{track_id}.csv")
            for file in os.listdir(directory):
                if file.endswith(".gr"):
                    g, terminals = self.read_input(os.path.join(directory, file))
                    if len(g.nodes) > 100:
                        continue
                    ans = ilp(g=g, terminals=terminals, root=terminals[0])["objective"]
                    self.assertEqual(ans, ans_dict[file])

        # self.assertEqual(True, True)  # add assertion here
        # self.assertEqual(True, True)  # add assertion here
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
