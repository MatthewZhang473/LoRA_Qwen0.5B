import h5py
import numpy as np
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent


def encode(trajectories, decimal_points=3):
    text_data = []
    for system in trajectories:
        system_text = ";".join([f"{point[0]:.{decimal_points}f},{point[1]:.{decimal_points}f}" for point in system])
        text_data.append(system_text)
    
    return text_data

def decode(text_data):
    trajectories = []
    for system_text in text_data:
        
        points = system_text.split(";")
        system = []
        for point in points:
            try:
                prey_str, predator_str = point.split(",")
                system.append([float(prey_str), float(predator_str)])
            except ValueError:
                continue
        trajectories.append(system)
    
    trajectories = np.array(trajectories)
    
    return trajectories

class LotkaVolterraDataset:
    def __init__(self, normalize=True, file_path=str(DATA_DIR / "lotka_volterra_data.h5")):
        with h5py.File(file_path, "r") as f:
            self.trajectories = f["trajectories"][:]
            self.time_points = f["time"][:]
        if normalize:
            self.trajectories, self.mean, self.std = self._normalize(self.trajectories)

    def _normalize(self, data):
        mean = np.mean(data, axis=(0,1), keepdims=True)
        std = np.std(data, axis=(0,1), keepdims=True)
        return (data - mean) / std, mean, std


if __name__ == "__main__":
    dataset = LotkaVolterraDataset()
    trajectories, time_points = dataset.trajectories, dataset.time_points
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Time points shape: {time_points.shape}")

    texts = encode(trajectories)
    print(f"maximum text length: {max(len(t) for t in texts)}, minimum text length: {min(len(t) for t in texts)}")
    
    decoded_trajectories = decode(texts)
    print(f"Decoded trajectories shape: {decoded_trajectories.shape}")


    
