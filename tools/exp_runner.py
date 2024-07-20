import rospy


class ExperimentRunner:
    def __init__(self, planner) -> None:
        self.planner = planner
        self.planner_step = 0
        self.planning_budget = self.planner.planning_budget

    def start(self):
        print("---------- experiment start ---------- \n")

        while not rospy.is_shutdown():
            if self.planner_step < self.planning_budget:
                steps = self.planner.planning()
                self.planner_step += steps
                print(f"--- step {self.planner_step} --- \n")

                if self.planner.map_update_for_planning:
                    # self.planner.reset()
                    print("training network")
                    outputs, iteration_step = self.planner.update(
                        self.planner.iteration_per_step
                    )
                    print(f"to {iteration_step} iterations")
                    self.planner.update_end()

            else:
                # save experiment results
                self.planner.save_experiment()
                print("---------- experiment done ---------- \n")
                break
