from sokoban import Warehouse
from mySokobanSolver import solve_weighted_sokoban, check_elem_action_seq
from scipy.optimize import linear_sum_assignment


def verify_box_assignments(initial_boxes, final_boxes, targets, weights):
    """
    Uses the Hungarian algorithm to verify if the final box placement
    matches the minimal weighted cost assignment.
    """
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
    test_files = [
        "warehouse_03.txt",
        "warehouse_8a.txt",
        "warehouse_13.txt",
        "warehouse_47.txt",
        "warehouse_49.txt",
        "warehouse_103.txt",
        "warehouse_109.txt",
        "warehouse_127.txt",
        "warehouse_131.txt",
        "warehouse_199.txt"
    ]

    for fname in test_files:
        print(f"\n<< Testing {fname} >>")
        w = Warehouse()
        w.load_warehouse(f"./warehouses/{fname}")
        initial_boxes = sorted(w.boxes)

        actions, cost = solve_weighted_sokoban(w)

        if actions == 'Impossible':
            print("❌ No solution found.")
            continue

        # Step 1: Check the action sequence is valid
        result_str = check_elem_action_seq(w, actions)
        if result_str == "Impossible":
            print("❌ Invalid action sequence.")
            continue
        else:
            print(f"✅ Valid path — {len(actions)} steps, cost: {cost}")

        # Step 2: Parse the final warehouse state
        final_w = Warehouse()
        final_w.from_string(result_str)
        final_boxes = sorted(final_w.boxes)

        # Only use weight-aware matching if weights are non-zero
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


        # Optional: Show first few actions
        #print("First few actions:", actions[:10], "..." if len(actions) > 10 else "")

        print("Full action sequence ({} steps):".format(len(actions)))
        print(", ".join(actions))

if __name__ == "__main__":
    test_solve_weighted_sokoban()