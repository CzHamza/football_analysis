import numpy as np
import cv2

class SpaceOccupancyAnalyzer:
    def __init__(self, field_width=68, field_length=105, grid_size=10):
        """
        Initialize the space occupancy analyzer.
        :param field_width: Width of the field in meters.
        :param field_length: Length of the field in meters.
        :param grid_size: Size of each grid cell in meters.
        """
        self.field_width = field_width
        self.field_length = field_length
        self.grid_size = grid_size
        
        # Compute grid dimensions
        self.grid_rows = int(np.ceil(self.field_length / self.grid_size))
        self.grid_cols = int(np.ceil(self.field_width / self.grid_size))

    def analyze_space_control(self, player_tracks):
        """
        Analyze space occupancy based on player positions.
        :param player_tracks: List of player positions by frame.
        :return: A 2D numpy array representing space control per grid cell.
        """
        space_control = np.zeros((self.grid_rows, self.grid_cols, 2))  # Two teams

        for frame in player_tracks:
            for player_id, player_info in frame.items():
                team = player_info.get('team')
                position = player_info.get('position_transformed')

                if team is not None and position is not None:
                    row = int(position[1] / self.grid_size)
                    col = int(position[0] / self.grid_size)

                    if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                        space_control[row, col, team - 1] += 1

        return space_control

    def visualize_space_occupancy(self, space_control, output_path):
        """
        Visualize space control and save as an image.
        :param space_control: A 2D numpy array representing space control per grid cell.
        :param output_path: Path to save the output visualization.
        """
        max_control = np.sum(space_control, axis=-1)
        control_ratios = space_control / (max_control[..., None] + 1e-5)  # Avoid division by zero

        # Debug: Print control ratios
        print("Control Ratios:")
        print(control_ratios)

        # Create an empty field visualization
        img_height, img_width = self.grid_rows * 20, self.grid_cols * 20
        field_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                team1_ratio = control_ratios[row, col, 0]
                team2_ratio = control_ratios[row, col, 1]

                # Normalize colors for better visibility
                team1_color = int(np.clip(team1_ratio * 255, 0, 255))
                team2_color = int(np.clip(team2_ratio * 255, 0, 255))

                color = (
                    team2_color,  # Blue for Team 2
                    0,            # Green is unused
                    team1_color   # Red for Team 1
                )

                cv2.rectangle(
                    field_img,
                    (col * 20, row * 20),
                    ((col + 1) * 20, (row + 1) * 20),
                    color,
                    -1
                )

        # Overlay field lines
        for i in range(self.grid_rows + 1):
            cv2.line(field_img, (0, i * 20), (img_width, i * 20), (255, 255, 255), 1)
        for j in range(self.grid_cols + 1):
            cv2.line(field_img, (j * 20, 0), (j * 20, img_height), (255, 255, 255), 1)

        # Add grid labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                label = f"{row},{col}"
                cv2.putText(
                    field_img,
                    label,
                    (col * 20 + 5, row * 20 + 15),
                    font,
                    0.3,
                    (255, 255, 255),
                    1
                )


        # Add legend
        legend_height = 50
        legend = np.zeros((legend_height, img_width, 3), dtype=np.uint8)
        cv2.rectangle(legend, (10, 10), (30, 30), (0, 0, 255), -1)
        cv2.putText(legend, "Team 1 (Red)", (40, 25), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (150, 10), (170, 30), (255, 0, 0), -1)
        cv2.putText(legend, "Team 2 (Blue)", (180, 25), font, 0.5, (255, 255, 255), 1)

        # Combine legend with field image
        combined_img = np.vstack((legend, field_img))

        # Save visualization
        cv2.imwrite(output_path, combined_img)
        print(f"Space occupancy visualization saved to {output_path}")