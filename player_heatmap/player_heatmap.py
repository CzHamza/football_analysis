import cv2
import numpy as np

class PlayerHeatmap:
    def __init__(self, frame_size, heatmap_resolution=(1080, 1920), pitch_image_path=None):
        #Initialize the PlayerHeatmap class.
        self.frame_size = frame_size
        self.heatmap_resolution = heatmap_resolution
        self.pitch_image_path = pitch_image_path
        self.blur_radius = max(1, int(min(self.heatmap_resolution) / 20) | 1)
        self.heatmap = np.zeros(heatmap_resolution, dtype=np.float32)

        # Scaling factors to map coordinates to heatmap resolution
        self.h_scale = heatmap_resolution[0] / frame_size[0]
        self.w_scale = heatmap_resolution[1] / frame_size[1]

        self.pitch_image = self.load_pitch_image()
    
    def reset_heatmap(self):
        self.heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)

    # Load the football pitch image and resize it to match the heatmap resolution.
    def load_pitch_image(self):
        if self.pitch_image_path:
            pitch_img = cv2.imread(self.pitch_image_path)
            if pitch_img is not None:
                return cv2.resize(pitch_img, (self.heatmap_resolution[1], self.heatmap_resolution[0]))
            else:
                print("Warning: Pitch image not found. Using black background.")
        return np.zeros((self.heatmap_resolution[0], self.heatmap_resolution[1], 3), dtype=np.uint8)

    # Update the heatmap with player positions from the tracks.
    def update_heatmap(self, tracks, team=None):
        player_radius = 18
        for frame_tracks in tracks['players']:
            for player_data in frame_tracks.values():
                if team is not None and player_data.get('team') != team:
                    continue 
                bbox = player_data['bbox']
                center_x = int(((bbox[0] + bbox[2]) / 2) * self.w_scale)
                center_y = int(((bbox[1] + bbox[3]) / 2) * self.h_scale)
                if 0 <= center_x < self.heatmap_resolution[1] and 0 <= center_y < self.heatmap_resolution[0]:
                    cv2.circle(self.heatmap, (center_x, center_y), player_radius, 1, thickness=-1)


    #Normalize the heatmap to range [0, 255] and apply Gaussian blur.
    def get_normalized_heatmap(self):
        if self.blur_radius % 2 == 0:
            self.blur_radius += 1

        heatmap_blurred = cv2.GaussianBlur(self.heatmap, (self.blur_radius, self.blur_radius), 0)
        heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return heatmap_normalized
        
    # Save the heatmap overlaid on the football pitch with vivid heatmap colors and a green background.
    def save_heatmap_on_pitch(self, output_path="output_heatmap_pitch.png", colormap=cv2.COLORMAP_JET, alpha=0.5):
        heatmap_normalized = self.get_normalized_heatmap()
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
        heatmap_colored = cv2.convertScaleAbs(heatmap_colored, alpha=2.0, beta=30)
        
        mask = heatmap_normalized > 10  # Include low-intensity areas
        pitch_overlay = self.pitch_image.copy()
        pitch_overlay[mask] = cv2.addWeighted(self.pitch_image[mask], 1 - alpha, heatmap_colored[mask], alpha, 0)

        cv2.imwrite(output_path, pitch_overlay)
        print(f"Heatmap with vivid colors saved to {output_path}")

