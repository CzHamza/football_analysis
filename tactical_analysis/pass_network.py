from matplotlib import pyplot as plt
import networkx as nx

class PassNetwork:
    def construct_pass_network(self, player_tracks, team_ball_control):
        graph = nx.DiGraph()
        for frame_tracks in player_tracks:
            for player_id, track in frame_tracks.items():
                if "has_ball" in track and track["has_ball"]:
                    team = track["team"]
                    if team not in graph:
                        graph.add_node(team)
                    for target_player, target_track in frame_tracks.items():
                        if target_player != player_id and target_track["team"] == team:
                            graph.add_edge(player_id, target_player, weight=1)
        return graph

    def visualize(self, graph, output_path):
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.title("Pass Network")
        plt.savefig(output_path)
        plt.close()
