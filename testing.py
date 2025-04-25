from sokoban import Warehouse
from mySokobanSolver import solve_weighted_sokoban, check_elem_action_seq
from scipy.optimize import linear_sum_assignment
from time import time
import os
import re


def verify_box_assignments(initial_boxes, final_boxes, targets, weights):
    cost_matrix = []
    for box in initial_boxes:
        row = []
        for target in targets:
            dist = abs(box[0] - target[0]) + abs(box[1] - target[1])
            row.append(dist)
        cost_matrix.append(row)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    expected_assignments = {tuple(initial_boxes[i]): targets[j] for i, j in zip(row_ind, col_ind)}

    moved = dict(zip(initial_boxes, final_boxes))
    success = all(moved[b] == expected_assignments[b] for b in moved)
    return success, expected_assignments


def test_solve_weighted_sokoban():
    warehouse_dir = "./warehouses"
    test_files = sorted(
    [f for f in os.listdir(warehouse_dir) if f.endswith('.txt')],
    key=lambda name: int(re.search(r'\d+', name).group())
)

    for fname in test_files:
        print(f"\n<< Testing {fname} >>")
        w = Warehouse()
        w.load_warehouse(os.path.join(warehouse_dir, fname))
        initial_boxes = sorted(w.boxes)

        time1 = time()
        actions, cost = solve_weighted_sokoban(w)
        time2 = time()
        print(f"⏱️ Time taken: {time2 - time1:.4f} seconds")

        if actions == 'Impossible':
            print("❌ No solution found.")
            continue
            
        result_str = check_elem_action_seq(w, actions)
        if result_str == "Impossible":
            print("❌ Invalid action sequence.")
            continue
        else:
            print(f"✅ Valid path — {len(actions)} steps, cost: {cost}")

        final_w = Warehouse()
        final_w.from_string(result_str)
        final_boxes = sorted(final_w.boxes)

        if w.weights and any(w.weights):
            correct_assignment, expected = verify_box_assignments(
                initial_boxes, final_boxes, list(w.targets), w.weights
            )
            if correct_assignment:
                print("✅ Correct box-to-target assignment (weight-aware)")
            else:
                print("⚠️ Boxes were not pushed to optimal targets based on weight.")
                print("Expected assignments:")
                for b, t in expected.items():
                    print(f"  Box at {b} → Target at {t}")
                print("But final boxes were:")
                for i, b in enumerate(final_boxes):
                    print(f"  Box {i} at {b}")
        else:
            print("✅ Correct final box positions (weights not used)")

        print("Full action sequence ({} steps):".format(len(actions)))
        print(", ".join(actions))


if __name__ == "__main__":
    test_solve_weighted_sokoban()
