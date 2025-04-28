import cv2
import random
from ultralytics import YOLO

class Points_Calculator:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = max(len(row) for row in grid)
        self.visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.terrain_base = {
            'wheat_field': 'wheatfield',
            'wheatfield_crown1': 'wheatfield',
            'lake_crown1': 'lake',
            'swamp_crown1': 'swamp',
            'swamp_crown2': 'swamp',
            'mine_crown1': 'mine',
            'mine_crown2': 'mine',
            'mine_crown3': 'mine',
            'forest_crown1': 'forest',
            'grassland_crown1': 'grassland',
            'grassland_crown2': 'grassland'
        }

    def get_base_terrain_type(self, label):
        return self.terrain_base.get(label.split('_crown')[0], label.split('_crown')[0])

    def get_crown_count(self, tile):
        if '_crown' in tile:
            crown = tile.split('_crown')[-1]
            if crown.isdigit():
                return int(crown)
        return 0

    def depth_first_search(self, row, col, region_id):
        if not (0 <= row < self.rows and 0 <= col < len(self.grid[row])):
            return None, 0, 0, []
        terrain_type = self.get_base_terrain_type(self.grid[row][col])
        if terrain_type in ['castle', 'table']:
            return None, 0, 0, []
        stack = [(row, col)]
        connected_tiles = crowns = 0
        while stack:
            r, c = stack.pop()
            if not (0 <= r < self.rows and 0 <= c < len(self.grid[r])) or self.visited[r][c]:
                continue
            if self.get_base_terrain_type(self.grid[r][c]) != terrain_type:
                continue
            self.visited[r][c] = region_id
            connected_tiles += 1
            crowns += self.get_crown_count(self.grid[r][c])
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        if connected_tiles >= 2:
            return terrain_type, connected_tiles, crowns, connected_tiles * crowns
        return None, 0, 0, 0

    def calculate_total_points(self):
        total = 0
        region_id = 1
        for r in range(self.rows):
            for c in range(len(self.grid[r])):
                if not self.visited[r][c] and self.get_base_terrain_type(self.grid[r][c]) != 'castle':
                    _, tiles, crowns, score = self.depth_first_search(r, c, region_id)
                    if tiles >= 2 and crowns > 0:
                        total += score
                        region_id += 1
        return total

def is_inside(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    return x1a >= x1b and y1a >= y1b and x2a <= x2b and y2a <= y2b

def calculate_iou(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    x_left = max(x1a, x1b)
    y_top = max(y1a, y1b)
    x_right = min(x2a, x2b)
    y_bottom = min(y2a, y2b)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)
    return intersection / (area1 + area2 - intersection)

def main():
    model = YOLO("bestv3.pt")

    ground_truth = {str(i+1): v for i, v in enumerate([
        36, 43, 52, 42, 36, 43, 41, 42, 45, 48, 49, 22,
        44, 38, 49, 22, 40, 60, 29, 50, 40, 66, 36, 52,
        44, 48, 67, 65, 44, 48, 67, 65, 21, 36, 46, 51,
        21, 36, 46, 40, 33, 43, 66, 33, 38, 43, 66, 42,
        26, 34, 35, 42, 23, 34, 37, 44, 66, 27, 38, 44,
        66, 36, 38, 66, 80, 124, 99, 66, 124, 99, 66, 80,
        124, 99
    ])}

    keys = list(ground_truth.keys())
    random.seed(42)
    random.shuffle(keys)

    split_idx = int(len(keys) * 0.8)
    train_keys = keys[:split_idx]
    test_keys = keys[split_idx:]

    def evaluate(keys_subset):
        correct = 0
        for key in keys_subset:
            img_path = f"King_Domino_dataset/Cropped_and_perspective_corrected_boards/{key}.jpg"
            img = cv2.imread(img_path)
            result = model.predict(img, conf=0.3)[0]

            filtered_boxes = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                if (x2 - x1) < 24 or (y2 - y1) < 24:
                    continue

                new_box = (x1.item(), y1.item(), x2.item(), y2.item())

                skip = any(
                    is_inside(new_box, kept_box) or is_inside(kept_box, new_box) or calculate_iou(new_box, kept_box) > 0.3
                    for kept_box, _, _ in filtered_boxes
                )

                if not skip:
                    filtered_boxes.append((new_box, box, result.names[int(box.cls[0].item())]))

            filtered_boxes.sort(key=lambda x: ((x[0][1] + x[0][3]) / 2))
            rows = [[] for _ in range(5)]
            row_height = img.shape[0] / 5

            for box in filtered_boxes:
                center_y = (box[0][1] + box[0][3]) / 2
                row_idx = min(int(center_y // row_height), 4)
                rows[row_idx].append(box)

            for row in rows:
                row.sort(key=lambda x: (x[0][0] + x[0][2]) / 2)

            sorted_boxes = [box for row in rows for box in row]

            grid = [['' for _ in range(5)] for _ in range(5)]

            for idx, (_, _, label) in enumerate(sorted_boxes):
                r, c = idx // 5, idx % 5
                if r < 5 and c < 5:
                    grid[r][c] = label

            calculator = Points_Calculator(grid)
            total_points = calculator.calculate_total_points()

            if total_points == ground_truth[key]:
                correct += 1

        return correct / len(keys_subset) * 100

    train_acc = evaluate(train_keys)
    test_acc = evaluate(test_keys)

    print(f"\nTrain Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()